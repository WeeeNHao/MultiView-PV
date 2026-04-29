from __future__ import annotations

import os
import uuid
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from tqdm import tqdm

from osgeo import ogr

try:
    from rtree import index as _rtree_index
except Exception:
    _rtree_index = None


def _bbox_iou(b1: Sequence[float], b2: Sequence[float]) -> float:
    x1 = max(float(b1[0]), float(b2[0]))
    y1 = max(float(b1[1]), float(b2[1]))
    x2 = min(float(b1[2]), float(b2[2]))
    y2 = min(float(b1[3]), float(b2[3]))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)
    a1 = max(float(b1[2]) - float(b1[0]), 0.0) * max(float(b1[3]) - float(b1[1]), 0.0)
    a2 = max(float(b2[2]) - float(b2[0]), 0.0) * max(float(b2[3]) - float(b2[1]), 0.0)
    union = a1 + a2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def _flat_to_pairs(coords: Iterable[float]) -> List[Tuple[float, float]]:
    vals = list(coords)
    if len(vals) < 6 or len(vals) % 2 != 0:
        return []
    return [(float(vals[i]), float(vals[i + 1])) for i in range(0, len(vals), 2)]


def _feature_to_geometry(feature: Dict[str, Any]) -> ogr.Geometry | None:
    if "geom" in feature and feature["geom"] is not None:
        return feature["geom"]

    seg = feature.get("segmentation")
    if not seg:
        feature["geom"] = None
        return None

    poly = ogr.Geometry(ogr.wkbPolygon)
    rings = seg if isinstance(seg, list) else []
    for ring_data in rings:
        pts = _flat_to_pairs(ring_data)
        if len(pts) < 3:
            continue

        ring = ogr.Geometry(ogr.wkbLinearRing)
        for x, y in pts:
            ring.AddPoint(x, y)
        if pts[0] != pts[-1]:
            ring.AddPoint(pts[0][0], pts[0][1])

        if ring.GetPointCount() >= 4:
            poly.AddGeometry(ring)

    if poly.IsEmpty():
        feature["geom"] = None
        return None
    feature["geom"] = poly
    return poly


def _geometry_iou(f1: Dict[str, Any], f2: Dict[str, Any]) -> float:
    g1 = _feature_to_geometry(f1)
    g2 = _feature_to_geometry(f2)
    if g1 is None or g2 is None:
        return _bbox_iou(f1["bbox"], f2["bbox"])

    inter = g1.Intersection(g2)
    if inter is None or inter.IsEmpty():
        return 0.0
    inter_area = inter.Area()
    if inter_area <= 0:
        return 0.0

    g1_area = g1.Area()
    g2_area = g2.Area()
    union_area = g1_area + g2_area - inter_area
    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


def nms_features(
    features: List[Dict[str, Any]],
    score_field: str,
    iou_threshold: float,
    use_geometry_iou: bool = False,
    backend: str = "rtree",
) -> List[Dict[str, Any]]:
    backend_name = str(backend).strip().lower()
    if backend_name == "rtree":
        return _nms_features_rtree(
            features=features,
            score_field=score_field,
            iou_threshold=iou_threshold,
            use_geometry_iou=use_geometry_iou,
        )
    if backend_name == "naive":
        return _nms_features_naive(
            features=features,
            score_field=score_field,
            iou_threshold=iou_threshold,
            use_geometry_iou=use_geometry_iou,
        )
    if backend_name == "auto":
        if _rtree_index is not None:
            return _nms_features_rtree(
                features=features,
                score_field=score_field,
                iou_threshold=iou_threshold,
                use_geometry_iou=use_geometry_iou,
            )
        return _nms_features_naive(
            features=features,
            score_field=score_field,
            iou_threshold=iou_threshold,
            use_geometry_iou=use_geometry_iou,
        )
    raise ValueError(f"Unsupported NMS backend: {backend}")


def _nms_features_naive(
    features: List[Dict[str, Any]],
    score_field: str,
    iou_threshold: float,
    use_geometry_iou: bool,
) -> List[Dict[str, Any]]:
    if not features:
        return []

    ordered = sorted(features, key=lambda x: float(x.get(score_field, 0.0)), reverse=True)
    kept: List[Dict[str, Any]] = []
    suppressed_count = 0

    pbar = tqdm(ordered, desc="NMS", leave=False)
    for cand in pbar:
        suppressed = False
        for base in kept:
            if use_geometry_iou:
                iou = _geometry_iou(cand, base)
            else:
                iou = _bbox_iou(cand["bbox"], base["bbox"])
            if iou > iou_threshold:
                suppressed = True
                break
        if not suppressed:
            kept.append(cand)
        else:
            suppressed_count += 1
        pbar.set_postfix(kept=len(kept), drop=suppressed_count)
    return kept


def _nms_features_rtree(
    features: List[Dict[str, Any]],
    score_field: str,
    iou_threshold: float,
    use_geometry_iou: bool,
) -> List[Dict[str, Any]]:
    if not features:
        return []
    if _rtree_index is None:
        return _nms_features_naive(
            features=features,
            score_field=score_field,
            iou_threshold=iou_threshold,
            use_geometry_iou=use_geometry_iou,
        )

    ordered = sorted(features, key=lambda x: float(x.get(score_field, 0.0)), reverse=True)
    kept: List[Dict[str, Any]] = []
    suppressed_count = 0

    tmp_dir = os.path.join(os.path.dirname(__file__), "..", "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    index_path = os.path.join(tmp_dir, f"nms_rtree_{uuid.uuid4().hex}")

    idx = _rtree_index.Index(index_path)
    try:
        pbar = tqdm(ordered, desc="NMS", leave=False)
        for cand in pbar:
            suppressed = False
            cand_bbox = tuple(float(v) for v in cand["bbox"])
            candidate_kept_ids = list(idx.intersection(cand_bbox))

            for kept_id in candidate_kept_ids:
                base = kept[kept_id]
                if use_geometry_iou:
                    iou = _geometry_iou(cand, base)
                else:
                    iou = _bbox_iou(cand["bbox"], base["bbox"])
                if iou > iou_threshold:
                    suppressed = True
                    break

            if not suppressed:
                kept.append(cand)
                idx.insert(len(kept) - 1, cand_bbox)
            else:
                suppressed_count += 1
            pbar.set_postfix(kept=len(kept), drop=suppressed_count)
    finally:
        idx.close()
        for ext in (".dat", ".idx"):
            fpath = index_path + ext
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                except OSError:
                    pass
    return kept
