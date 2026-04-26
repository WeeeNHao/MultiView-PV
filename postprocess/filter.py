from __future__ import annotations

import numpy as np
from typing import Any, Dict, List
from shapely.geometry import Polygon
from osgeo import ogr
from utils.common import Feature, FeatureList

def max_polygon_angle(poly: Polygon) -> float:
    if not poly.is_valid or poly.is_empty:
        return 0.0
    coords = list(poly.exterior.coords)
    if len(coords) < 4:
        return 0.0
    
    max_angle = 0.0
    for i in range(len(coords) - 1):
        p_prev = np.array(coords[i - 1])
        p_curr = np.array(coords[i])
        p_next = np.array(coords[(i + 1) % (len(coords) - 1)])
        
        v1 = p_prev - p_curr
        v2 = p_next - p_curr
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            continue
            
        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        if angle > max_angle:
            max_angle = angle
    return float(max_angle)


def _safe_aspect_ratio(poly: Polygon) -> float:
    if not poly.is_valid or poly.area <= 0:
        return 0.0
    min_rect = poly.minimum_rotated_rectangle
    coords = list(min_rect.exterior.coords)
    if len(coords) < 4:
        return 0.0
    
    edge_lengths = [np.linalg.norm(np.array(coords[i]) - np.array(coords[i+1])) for i in range(4)]
    edge_lengths.sort()
    w = edge_lengths[0]
    h = edge_lengths[2]
    if w <= 0 or h <= 0:
        return 0.0
    return float(max(h / w, w / h))


def _shape_rectangularity(poly: Polygon, ref: str = "min_rect") -> float:
    if not poly.is_valid or poly.area <= 0:
        return 0.0
    
    if ref == "min_rect":
        ref_area = poly.minimum_rotated_rectangle.area
    else:
        minx, miny, maxx, maxy = poly.bounds
        ref_area = max(maxx - minx, 0.0) * max(maxy - miny, 0.0)
        
    if ref_area <= 0:
        return 0.0
    ratio = poly.area / ref_area
    return float(np.exp(-abs(ratio - 1.0) / 0.1))


def _score_ratio(r: float, mu_r: float = 2.0, sigma_r: float = 0.2) -> float:
    return float(np.exp(-((r - mu_r) ** 2) / (2.0 * sigma_r ** 2)))


def _score_area(area: float, area_target: float, beta: float = 0.35) -> float:
    if area_target <= 0:
        return 0.0
    return float(np.exp(-abs(area - area_target) / (area_target * beta)))


def pv_consistency_score(
    poly: Polygon,
    area_target: float,
    mu_r: float = 2.0,
    sigma_r: float = 0.2,
    beta: float = 0.35,
    w_ratio: float = 1.0,
    w_size: float = 1.0,
    w_shape: float = 1.0,
    combine_mode: str = "weighted",
    gamma: float = 1.0,
    shape_ref: str = "min_rect",
) -> float:
    area = poly.area
    r = _safe_aspect_ratio(poly)
    s_ratio = _score_ratio(r, mu_r=mu_r, sigma_r=sigma_r)
    s_size = _score_area(area, area_target=area_target, beta=beta)
    s_shape = _shape_rectangularity(poly, ref=shape_ref)
    
    if combine_mode == "product":
        fused = (s_size * s_shape) ** gamma
    else:
        w_sum = max(w_ratio + w_size + w_shape, 1e-6)
        fused = (w_ratio * s_ratio + w_size * s_size + w_shape * s_shape) / w_sum
        
    return float(fused)


def _feature_to_polygon(feature: Feature) -> Polygon:
    from io_flow.shp_io import _segmentation_to_polygon
    geom = _segmentation_to_polygon(feature.get("segmentation"))
    if geom is None or geom.IsEmpty():
        bbox = feature.get("bbox", [])
        if len(bbox) == 4:
            from io_flow.shp_io import _bbox_to_polygon
            geom = _bbox_to_polygon(bbox)
            
    if geom is None or geom.IsEmpty():
        return Polygon()
        
    try:
        pts = []
        ring = geom.GetGeometryRef(0)
        if ring is not None:
            for i in range(ring.GetPointCount()):
                pts.append((ring.GetX(i), ring.GetY(i)))
            return Polygon(pts)
    except Exception:
        pass
    return Polygon()


def filter_features(
    features: FeatureList,
    cfg: Dict[str, Any],
) -> FeatureList:
    if not features:
        return []
        
    area_target = cfg.get("area_target", None)
    score_threshold = cfg.get("score_threshold", None)
    score_field = str(cfg.get("score_field", "con_weight"))
    
    mu_r = float(cfg.get("mu_r", 1.96))
    sigma_r = float(cfg.get("sigma_r", 0.2))
    beta = float(cfg.get("beta", 0.25))
    w_ratio = float(cfg.get("w_ratio", 5.0))
    w_size = float(cfg.get("w_size", 10.0))
    w_shape = float(cfg.get("w_shape", 5.0))
    combine_mode = str(cfg.get("combine_mode", "weighted"))
    gamma = float(cfg.get("gamma", 1.0))
    shape_ref = str(cfg.get("shape_ref", "min_rect"))
    
    if area_target is None:
        areas = []
        for feat in features:
            poly = _feature_to_polygon(feat)
            if poly.is_valid and poly.area > 0:
                areas.append(poly.area)
        if not areas:
            return features
        area_target = float(np.mean(areas))
    else:
        area_target = float(area_target)
        
    filtered: FeatureList = []
    for feat in features:
        poly = _feature_to_polygon(feat)
        if not poly.is_valid or poly.area <= 0:
            continue
            
        con_old = float(feat.get(score_field, 0.0))
        pv_score = pv_consistency_score(
            poly=poly,
            area_target=area_target,
            mu_r=mu_r,
            sigma_r=sigma_r,
            beta=beta,
            w_ratio=w_ratio,
            w_size=w_size,
            w_shape=w_shape,
            combine_mode=combine_mode,
            gamma=gamma,
            shape_ref=shape_ref,
        )
        
        final_score = float(pv_score * 0.8 + con_old * 0.2)
        
        if score_threshold is not None and final_score < float(score_threshold):
            continue
            
        out_feat = dict(feat)
        out_feat["con_old"] = con_old
        out_feat["pv_score"] = pv_score
        out_feat[score_field] = final_score
        out_feat["con_weight"] = final_score
        out_feat["score"] = final_score
        
        filtered.append(out_feat)
        
    return filtered
