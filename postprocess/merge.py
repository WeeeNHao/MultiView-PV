from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Tuple

from tqdm import tqdm

from io_flow.shp_io import read_features_from_shapefile
from postprocess.nms import _geometry_iou, _rtree_index, nms_features
from postprocess.pixel_fusion import fuse_pixel_vote


def _bbox_tuple(feature: Dict[str, Any]) -> Tuple[float, ...]:
    return tuple(float(v) for v in feature["bbox"])


def _score_of(feature: Dict[str, Any], score_field: str) -> float:
    return float(feature.get(score_field, 0.0))


def _cluster_by_iou(
    features: List[Dict[str, Any]],
    score_field: str,
    iou_threshold: float,
) -> List[List[Dict[str, Any]]]:
    ordered = sorted(features, key=lambda x: _score_of(x, score_field), reverse=True)
    clusters: List[List[Dict[str, Any]]] = []

    # Match each item against existing cluster *heads* only (cluster[0], which is
    # fixed once the cluster is created), pruned by a single in-memory R-tree over
    # those heads. Scanning the bbox-overlap candidates in ascending cluster order
    # preserves the original "first cluster whose head overlaps wins" behavior,
    # while dropping the per-pair disk-backed nms_features call that turned this
    # into O(n^2) filesystem round-trips.
    use_index = _rtree_index is not None
    idx = _rtree_index.Index() if use_index else None

    try:
        pbar = tqdm(ordered, desc="Clustering features", leave=False)
        for item in pbar:
            item_bbox = _bbox_tuple(item)
            candidate_ids = idx.intersection(item_bbox) if use_index else range(len(clusters))

            attached = -1
            for c in sorted(candidate_ids):
                # Only one survives NMS iff their IoU exceeds the threshold.
                if _geometry_iou(clusters[c][0], item) > iou_threshold:
                    attached = c
                    break

            if attached >= 0:
                clusters[attached].append(item)
            else:
                new_id = len(clusters)
                clusters.append([item])
                if use_index:
                    idx.insert(new_id, item_bbox)
            pbar.set_postfix(clusters=len(clusters))
    finally:
        if use_index:
            idx.close()
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

    if strategy == "pixel_vote":
        # Baselines M2 / M3: fuse in raster space instead of by instance.
        # extra_features_shp folds a second source (the TDOM result) into the
        # same vote grid, which is what makes M3 a *pixel-level* late fusion
        # rather than an object-level merge bolted on afterwards.
        extra_shp = str(cfg.get("extra_features_shp", "")).strip()
        if extra_shp:
            if not os.path.exists(extra_shp):
                raise FileNotFoundError(
                    f"multiview.extra_features_shp not found: {extra_shp}"
                )
            extra = read_features_from_shapefile(shp_path=extra_shp)
            normalized = list(normalized) + list(extra)
        return fuse_pixel_vote(normalized, cfg)

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
    candidate_ids: Iterable[int],
    iou_threshold: float,
    score_field: str,
) -> int:
    # Among the (bbox-pruned) candidates, return the index of the highest-scoring
    # one whose geometry IoU with ``feature`` exceeds the threshold. This matches
    # the old ``len(nms_features([feature, cand])) == 1`` test: NMS suppresses the
    # pair exactly when their IoU is above ``iou_threshold``. Iterating the ids in
    # sorted order keeps the original tie-break (lowest index wins on equal score).
    best_idx = -1
    best_score = -1.0

    for idx in sorted(candidate_ids):
        cand = candidates[idx]
        if _geometry_iou(feature, cand) > iou_threshold:
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

    # Index the running merged set once in an in-memory R-tree, then refine only
    # the few bbox-overlapping candidates per DOM feature with a geometry IoU.
    # The old code spun up a fresh *disk-backed* R-tree inside nms_features for
    # every (dom, candidate) pair -- O(N*M) index files created and deleted on
    # disk. That filesystem churn, not the IoU math, is what made merging slow.
    # bbox overlap is necessary for geometry-IoU overlap, so pruning by the
    # R-tree never discards a real match.
    use_index = _rtree_index is not None
    idx = _rtree_index.Index() if use_index else None
    if use_index:
        for i, feat in enumerate(merged):
            idx.insert(i, _bbox_tuple(feat))

    pbar = tqdm(dom_norm, desc="DOM merge", leave=False)
    for dom_item in pbar:
        dom_bbox = _bbox_tuple(dom_item)
        candidate_ids = idx.intersection(dom_bbox) if use_index else range(len(merged))
        best_idx = _best_match_index(
            dom_item,
            merged,
            candidate_ids,
            iou_threshold=iou_threshold,
            score_field=score_field,
        )

        if best_idx < 0 or strategy == "union":
            new_id = len(merged)
            merged.append(dom_item)
            if use_index:
                idx.insert(new_id, dom_bbox)
            added += 1
            pbar.set_postfix(added=added, replaced=replaced)
            continue

        if strategy == "prefer_dom" or (
            strategy == "confidence"
            and _score_of(dom_item, score_field) > _score_of(merged[best_idx], score_field)
        ):
            if use_index:
                idx.delete(best_idx, _bbox_tuple(merged[best_idx]))
                idx.insert(best_idx, dom_bbox)
            merged[best_idx] = dom_item
            replaced += 1
            pbar.set_postfix(added=added, replaced=replaced)
            continue

        if strategy == "confidence":
            # Overlap found but the image feature already scores higher: keep it.
            pbar.set_postfix(added=added, replaced=replaced)
            continue

        raise ValueError(f"Unsupported dom merge strategy: {strategy}")

    return merged
