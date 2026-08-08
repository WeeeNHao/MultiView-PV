from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

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


def pair_overlap(f1: Feature, f2: Feature) -> Tuple[float, float]:
    """Return ``(iou, intersection_area)`` for two features.

    Both come out of the same polygon intersection, so callers that need the raw
    overlap (over/under-segmentation, which are containment ratios rather than
    IoU) get it without paying for a second pass over the geometry.

    Falls back to axis-aligned bbox IoU when a geometry cannot be built; the
    intersection area is then unavailable and reported as 0.
    """
    g1 = feature_geometry(f1)
    g2 = feature_geometry(f2)
    if g1 is None or g2 is None:
        b1 = f1.get("bbox")
        b2 = f2.get("bbox")
        if b1 and b2:
            return _bbox_iou(b1, b2), 0.0
        return 0.0, 0.0

    try:
        inter = g1.Intersection(g2)
    except Exception:
        # Self-intersecting / invalid ring -> attempt to repair via Buffer(0).
        g1 = g1.Buffer(0)
        g2 = g2.Buffer(0)
        inter = g1.Intersection(g2)

    if inter is None or inter.IsEmpty():
        return 0.0, 0.0
    inter_area = float(inter.Area())
    if inter_area <= 0:
        return 0.0, 0.0

    a1 = float(g1.Area())
    a2 = float(g2.Area())
    union = a1 + a2 - inter_area
    if union <= 0:
        return 0.0, inter_area
    return inter_area / union, inter_area


def pair_iou(f1: Feature, f2: Feature) -> float:
    """IoU between two features using true polygon geometry when available,
    otherwise axis-aligned bbox IoU."""
    return pair_overlap(f1, f2)[0]


def feature_centroid(feature: Feature) -> Optional[Tuple[float, float]]:
    """Centroid of a feature in map units, or None if it has no geometry.

    Used for the centroid RMSE: an instance can score a high IoU while sitting
    systematically off-centre, which is what a positional error metric exposes.
    """
    geom = feature_geometry(feature)
    if geom is not None and not geom.IsEmpty():
        try:
            c = geom.Centroid()
        except Exception:
            c = geom.Buffer(0).Centroid()
        if c is not None and not c.IsEmpty():
            return (float(c.GetX()), float(c.GetY()))

    bbox = feature.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return (
            (float(bbox[0]) + float(bbox[2])) / 2.0,
            (float(bbox[1]) + float(bbox[3])) / 2.0,
        )
    return None


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


def _polygon_parts(geom: ogr.Geometry) -> List[ogr.Geometry]:
    """Flatten a geometry to its polygon components.

    A single polygon yields itself; a MultiPolygon or GeometryCollection yields
    its polygon members and drops degenerate line/point debris.
    """
    gtype = ogr.GT_Flatten(geom.GetGeometryType())
    if gtype == ogr.wkbPolygon:
        return [geom]
    if gtype in (ogr.wkbMultiPolygon, ogr.wkbGeometryCollection):
        parts: List[ogr.Geometry] = []
        for i in range(geom.GetGeometryCount()):
            child = geom.GetGeometryRef(i)
            if child is not None and not child.IsEmpty():
                parts.extend(_polygon_parts(child))
        return parts
    return []


def dissolve_features(feats: Sequence[Feature]) -> Optional[ogr.Geometry]:
    """Union all feature polygons into a single (multi)polygon geometry.

    Overlapping panels are dissolved so shared area is counted once -- matching
    a rasterised "semantic mask" view of the scene. Invalid self-intersecting
    rings are repaired with ``Buffer(0)`` before union.
    """
    coll = ogr.Geometry(ogr.wkbMultiPolygon)
    n = 0
    for f in feats:
        g = feature_geometry(f)
        if g is None or g.IsEmpty():
            continue
        if not g.IsValid():
            g = g.Buffer(0)
            if g is None or g.IsEmpty():
                continue
        # Buffer(0) on a bow-tie ring splits it: the repaired geometry comes back
        # as a MultiPolygon (or a GeometryCollection), and adding either to a
        # wkbMultiPolygon raises "OGR Error: Unsupported geometry type" -- which
        # aborted the whole variant, not just the offending feature. Flatten to
        # the polygon parts instead.
        for part in _polygon_parts(g):
            coll.AddGeometry(part)
            n += 1
    if n == 0:
        return None
    try:
        return coll.UnionCascaded()
    except Exception:
        return coll.Buffer(0)


def area_overlap_metrics(
    pred_union: Optional[ogr.Geometry],
    gt_union: Optional[ogr.Geometry],
) -> dict:
    """Scene-level (semantic) area overlap metrics between two dissolved masks.

    Returns IoU / Dice / precision / recall plus the raw areas. This mirrors the
    "Area Results" reported by raster/area-based evaluation tools: a single
    intersection-over-union over the whole footprint rather than per-instance.
    """
    pa = float(pred_union.Area()) if pred_union is not None and not pred_union.IsEmpty() else 0.0
    ga = float(gt_union.Area()) if gt_union is not None and not gt_union.IsEmpty() else 0.0

    inter = 0.0
    if pred_union is not None and gt_union is not None and pa > 0 and ga > 0:
        try:
            ig = pred_union.Intersection(gt_union)
        except Exception:
            ig = pred_union.Buffer(0).Intersection(gt_union.Buffer(0))
        if ig is not None and not ig.IsEmpty():
            inter = float(ig.Area())

    union = pa + ga - inter
    return {
        "area_iou": inter / union if union > 0 else 0.0,
        "area_dice": 2.0 * inter / (pa + ga) if (pa + ga) > 0 else 0.0,
        "area_precision": inter / pa if pa > 0 else 0.0,
        "area_recall": inter / ga if ga > 0 else 0.0,
        "area_pred": pa,
        "area_gt": ga,
        "area_inter": inter,
    }


def build_overlap_matrices(
    preds: Sequence[Feature],
    gts: Sequence[Feature],
) -> Tuple[List[List[float]], List[List[float]]]:
    """Dense ``(iou, intersection_area)`` matrices, each ``[len(preds)][len(gts)]``.

    Candidate pairs are pre-filtered by bbox overlap so that non-overlapping
    pairs skip the (expensive) polygon intersection entirely.
    """
    pred_boxes = [_bbox(p) for p in preds]
    gt_boxes = [_bbox(g) for g in gts]

    iou_matrix: List[List[float]] = [[0.0] * len(gts) for _ in preds]
    inter_matrix: List[List[float]] = [[0.0] * len(gts) for _ in preds]
    for i, p in enumerate(preds):
        pb = pred_boxes[i]
        for j, g in enumerate(gts):
            gb = gt_boxes[j]
            if pb is not None and gb is not None and not _bbox_overlap(pb, gb):
                continue
            iou_matrix[i][j], inter_matrix[i][j] = pair_overlap(p, g)
    return iou_matrix, inter_matrix


def build_iou_matrix(
    preds: Sequence[Feature],
    gts: Sequence[Feature],
) -> List[List[float]]:
    """Dense IoU matrix ``[len(preds)][len(gts)]``."""
    return build_overlap_matrices(preds, gts)[0]
