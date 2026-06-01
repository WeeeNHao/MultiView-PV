from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from osgeo import ogr

from utils.common import Feature

# Reuse the geometry-construction logic that the rest of the pipeline relies on,
# so a feature dict read by ``read_features_from_shapefile`` is interpreted
# identically here.
from postprocess.nms import _bbox_iou, _feature_to_geometry


def feature_geometry(feature: Feature) -> Optional[ogr.Geometry]:
    """Return the OGR polygon for a feature, building it from ``segmentation``
    if a cached ``geom`` is not present (same rules as NMS / merge)."""
    return _feature_to_geometry(feature)


def geometry_area(feature: Feature) -> float:
    """Planimetric area of a feature.

    Prefers a stored ``area`` field (written by the pipeline) and falls back to
    the OGR geometry area, then to the bbox area.
    """
    val = feature.get("area")
    if isinstance(val, (int, float)) and val > 0:
        return float(val)

    geom = feature_geometry(feature)
    if geom is not None and not geom.IsEmpty():
        a = float(geom.Area())
        if a > 0:
            return a

    bbox = feature.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        w = max(float(bbox[2]) - float(bbox[0]), 0.0)
        h = max(float(bbox[3]) - float(bbox[1]), 0.0)
        return w * h
    return 0.0


def pair_iou(f1: Feature, f2: Feature) -> float:
    """IoU between two features using true polygon geometry when available,
    otherwise axis-aligned bbox IoU."""
    g1 = feature_geometry(f1)
    g2 = feature_geometry(f2)
    if g1 is None or g2 is None:
        b1 = f1.get("bbox")
        b2 = f2.get("bbox")
        if b1 and b2:
            return _bbox_iou(b1, b2)
        return 0.0

    try:
        inter = g1.Intersection(g2)
    except Exception:
        # Self-intersecting / invalid ring -> attempt to repair via Buffer(0).
        g1 = g1.Buffer(0)
        g2 = g2.Buffer(0)
        inter = g1.Intersection(g2)

    if inter is None or inter.IsEmpty():
        return 0.0
    inter_area = float(inter.Area())
    if inter_area <= 0:
        return 0.0

    a1 = float(g1.Area())
    a2 = float(g2.Area())
    union = a1 + a2 - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _bbox(feature: Feature) -> Optional[Tuple[float, float, float, float]]:
    bbox = feature.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    geom = feature_geometry(feature)
    if geom is not None and not geom.IsEmpty():
        env = geom.GetEnvelope()  # (minX, maxX, minY, maxY)
        return (float(env[0]), float(env[2]), float(env[1]), float(env[3]))
    return None


def _bbox_overlap(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def build_iou_matrix(
    preds: Sequence[Feature],
    gts: Sequence[Feature],
) -> List[List[float]]:
    """Dense IoU matrix ``[len(preds)][len(gts)]``.

    Candidate pairs are pre-filtered by bbox overlap so that non-overlapping
    pairs skip the (expensive) polygon intersection entirely.
    """
    pred_boxes = [_bbox(p) for p in preds]
    gt_boxes = [_bbox(g) for g in gts]

    matrix: List[List[float]] = [[0.0] * len(gts) for _ in preds]
    for i, p in enumerate(preds):
        pb = pred_boxes[i]
        for j, g in enumerate(gts):
            gb = gt_boxes[j]
            if pb is not None and gb is not None and not _bbox_overlap(pb, gb):
                continue
            matrix[i][j] = pair_iou(p, g)
    return matrix
