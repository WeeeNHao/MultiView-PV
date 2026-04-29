from __future__ import annotations

from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from postprocess.nms import nms_features


def _score_of(feature: Dict[str, Any], score_field: str) -> float:
    return float(feature.get(score_field, 0.0))





def _cluster_by_iou(
    features: List[Dict[str, Any]],
    score_field: str,
    iou_threshold: float,
) -> List[List[Dict[str, Any]]]:
    ordered = sorted(features, key=lambda x: _score_of(x, score_field), reverse=True)
    clusters: List[List[Dict[str, Any]]] = []

    pbar = tqdm(ordered, desc="Clustering features", leave=False)
    for item in pbar:
        attached = False
        for cluster in clusters:
            head = cluster[0]
            iou = nms_features([head, item], score_field=score_field, iou_threshold=iou_threshold, use_geometry_iou=True)
            # If only one survives, they overlap enough to be treated as same cluster.
            if len(iou) == 1:
                cluster.append(item)
                attached = True
                break
        if not attached:
            clusters.append([item])
        pbar.set_postfix(clusters=len(clusters))
    return clusters


def _weighted_average(cluster: List[Dict[str, Any]], key: str, score_field: str) -> float:
    weights = [max(_score_of(item, score_field), 1e-6) for item in cluster]
    values = [float(item.get(key, 0.0)) for item in cluster]
    wsum = sum(weights)
    return sum(v * w for v, w in zip(values, weights)) / wsum


def fuse_multiview_features(
    features: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not features:
        return []

    score_field = str(cfg.get("score_field", "con_weight"))
    iou_threshold = float(cfg.get("iou_threshold", 0.2))
    strategy = str(cfg.get("strategy", "nms_keep_max")).lower()

    normalized = features

    if strategy == "nms_keep_max":
        return nms_features(
            normalized,
            score_field=score_field,
            iou_threshold=iou_threshold,
            use_geometry_iou=True,
        )

    if strategy == "cluster_weighted":
        clusters = _cluster_by_iou(normalized, score_field=score_field, iou_threshold=iou_threshold)
        fused: List[Dict[str, Any]] = []
        for cluster in clusters:
            base = dict(max(cluster, key=lambda x: _score_of(x, score_field)))
            base["con_sem"] = _weighted_average(cluster, "con_sem", score_field)
            base["con_pv"] = _weighted_average(cluster, "con_pv", score_field)
            base["con_weight"] = _weighted_average(cluster, "con_weight", score_field)
            base["score"] = base["con_weight"]
            srcs = sorted({str(item.get("src", "")) for item in cluster if item.get("src")})
            base["src"] = ";".join(srcs) if srcs else base.get("src", "")
            fused.append(base)
        return fused

    raise ValueError(f"Unsupported multiview strategy: {strategy}")


def _best_match_index(
    feature: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    iou_threshold: float,
    score_field: str,
) -> int:
    best_idx = -1
    best_score = -1.0

    for idx, cand in enumerate(candidates):
        pair = nms_features([feature, cand], score_field=score_field, iou_threshold=iou_threshold, use_geometry_iou=True)
        if len(pair) == 1:
            score = _score_of(cand, score_field)
            if score > best_score:
                best_score = score
                best_idx = idx
    return best_idx


def merge_image_with_dom_features(
    image_features: List[Dict[str, Any]],
    dom_features: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    strategy = str(cfg.get("strategy", "confidence")).lower()
    score_field = str(cfg.get("score_field", "con_weight"))
    iou_threshold = float(cfg.get("iou_threshold", 0.2))

    merged = list(image_features)
    dom_norm = dom_features
    added = 0
    replaced = 0

    pbar = tqdm(dom_norm, desc="DOM merge", leave=False)
    for dom_item in pbar:
        best_idx = _best_match_index(dom_item, merged, iou_threshold=iou_threshold, score_field=score_field)

        if best_idx < 0:
            merged.append(dom_item)
            added += 1
            pbar.set_postfix(added=added, replaced=replaced)
            continue

        if strategy == "union":
            merged.append(dom_item)
            added += 1
            pbar.set_postfix(added=added, replaced=replaced)
            continue

        if strategy == "prefer_dom":
            merged[best_idx] = dom_item
            replaced += 1
            pbar.set_postfix(added=added, replaced=replaced)
            continue

        if strategy == "confidence":
            if _score_of(dom_item, score_field) > _score_of(merged[best_idx], score_field):
                merged[best_idx] = dom_item
                replaced += 1
            pbar.set_postfix(added=added, replaced=replaced)
            continue

        raise ValueError(f"Unsupported dom merge strategy: {strategy}")

    return merged
