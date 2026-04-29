from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

from osgeo import ogr

from utils.common import Feature, FeatureList, GeoMeta
from projection.oblique_projector import ObliqueProjector
from projection.scoring import compute_pv_geometry_score


_OBLIQUE_PROJECTOR_CACHE: Dict[str, ObliqueProjector] = {}


def _flat_to_pairs(flat_xy: Sequence[float]) -> List[Tuple[float, float]]:
    if len(flat_xy) < 6 or len(flat_xy) % 2 != 0:
        return []
    return [(float(flat_xy[i]), float(flat_xy[i + 1])) for i in range(0, len(flat_xy), 2)]


def _pairs_to_flat(points: Sequence[Tuple[float, float]]) -> List[float]:
    out: List[float] = []
    for x, y in points:
        out.append(float(x))
        out.append(float(y))
    return out


def _pixel_to_map_point(x: float, y: float, gt: Sequence[float]) -> Tuple[float, float]:
    gx = float(gt[0]) + x * float(gt[1]) + y * float(gt[2])
    gy = float(gt[3]) + x * float(gt[4]) + y * float(gt[5])
    return gx, gy


def _transform_segmentation(segmentation: List[List[float]], gt: Sequence[float]) -> List[List[float]]:
    transformed: List[List[float]] = []
    for ring in segmentation:
        pts = _flat_to_pairs(ring)
        if len(pts) < 3:
            continue
        mapped = [_pixel_to_map_point(x, y, gt) for x, y in pts]
        transformed.append(_pairs_to_flat(mapped))
    return transformed


def _bbox_from_segmentation(segmentation: List[List[float]]) -> List[float]:
    xs: List[float] = []
    ys: List[float] = []
    for ring in segmentation:
        pts = _flat_to_pairs(ring)
        for x, y in pts:
            xs.append(x)
            ys.append(y)
    if not xs or not ys:
        return [0.0, 0.0, 0.0, 0.0]
    return [min(xs), min(ys), max(xs), max(ys)]


def _feature_to_geometry(feature: Feature) -> Optional[ogr.Geometry]:
    if "geom" in feature and feature["geom"] is not None:
        return feature["geom"]

    seg = feature.get("segmentation")
    if not isinstance(seg, list) or not seg:
        feature["geom"] = None
        return None

    poly = ogr.Geometry(ogr.wkbPolygon)
    for ring_data in seg:
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


def _project_dom_feature(feature: Feature, gt: Sequence[float]) -> Feature:
    out = dict(feature)
    out["segmentation"] = _transform_segmentation(
        segmentation=feature.get("segmentation", []),
        gt=gt,
    )
    out["bbox"] = _bbox_from_segmentation(out["segmentation"])
    out["projection_method"] = "dom_geotransform"
    return out


def _get_oblique_projector(oblique_cfg: Dict[str, Any]) -> ObliqueProjector:
    pose_csv = str(oblique_cfg.get("pose_csv", "")).strip()
    dsm_path = str(oblique_cfg.get("dsm_path", "")).strip()
    cache_key = f"{pose_csv}::{dsm_path}"
    if cache_key not in _OBLIQUE_PROJECTOR_CACHE:
        _OBLIQUE_PROJECTOR_CACHE[cache_key] = ObliqueProjector(oblique_cfg)
    return _OBLIQUE_PROJECTOR_CACHE[cache_key]


def _score_one_feature(feature: Feature, score_cfg: Dict[str, Any]) -> Feature:
    out = dict(feature)
    con_sem = float(out.get("con_sem", out.get("score", 0.0)))

    geom = _feature_to_geometry(out)
    if geom is None:
        out["con_pv"] = 0.0
        out["con_sem"] = con_sem
        out["con_weight"] = con_sem
        out["score"] = out["con_weight"]
        return out

    pv_score = compute_pv_geometry_score(geom, score_cfg)

    w_sem = float(score_cfg.get("w_sem", 0.4))
    w_pv = float(score_cfg.get("w_pv", 0.6))
    wsum = max(w_sem + w_pv, 1e-6)

    con_pv = float(pv_score["con_pv"])
    con_weight = (w_sem * con_sem + w_pv * con_pv) / wsum

    out["area"] = float(pv_score["area"])
    out["aspect_ratio"] = float(pv_score["aspect_ratio"])
    out["area_score"] = float(pv_score["area_score"])
    out["ratio_score"] = float(pv_score["ratio_score"])
    out["shape_score"] = float(pv_score["shape_score"])
    out["con_pv"] = con_pv
    out["con_sem"] = con_sem
    out["con_weight"] = con_weight
    out["score"] = con_weight
    return out


def project_and_score_features(
    features: FeatureList,
    geo_meta: GeoMeta,
    projection_cfg: Dict[str, Any],
    image_path: Optional[str] = None,
    progress_position: int = 1,
) -> FeatureList:
    if not features:
        return []

    score_cfg = projection_cfg.get("score", {})
    project_coordinates = bool(projection_cfg.get("project_coordinates", True))
    mode = str(projection_cfg.get("mode", "auto")).lower()
    oblique_cfg = projection_cfg.get("oblique", {})
    score_threshold = float(score_cfg.get("score_threshold", 0.0))

    transformed: FeatureList = []
    filtered = 0
    pbar = tqdm(
        features,
        desc="Projecting feature",
        leave=False,
        position=progress_position,
    )
    for feature in pbar:
        out = dict(feature)
        gt = geo_meta.geotransform

        use_dom = project_coordinates and (mode == "dom" or (mode == "auto" and gt is not None))
        use_oblique = project_coordinates and (mode == "oblique" or (mode == "auto" and gt is None))

        if use_dom and gt is not None:
            out = _project_dom_feature(feature=feature, gt=gt)
        elif use_oblique:
            if not image_path:
                raise ValueError("image_path is required when using oblique projection")
            projector = _get_oblique_projector(oblique_cfg)
            out = projector.project_feature(feature=feature, image_path=image_path)

        out = _score_one_feature(out, score_cfg=score_cfg)
        if score_threshold > 0 and out.get("score", 0.0) < score_threshold:
            filtered += 1
            continue
        transformed.append(out)
        pbar.set_postfix(kept=len(transformed), filtered=filtered)

    return transformed
