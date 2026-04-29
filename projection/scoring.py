from __future__ import annotations

import math
from typing import Any, Dict

from osgeo import ogr
from shapely.wkb import loads as load_wkb
from shapely import Polygon
import numpy as np

def _safe_exp_decay(delta: float, beta: float) -> float:
    beta = max(beta, 1e-6)
    return float(math.exp(-abs(delta) / beta))


def _safe_gaussian(delta: float, sigma: float) -> float:
    sigma = max(sigma, 1e-6)
    return float(math.exp(-((delta * delta)  / (2.0 * sigma * sigma))))


def _get_rect_info(geom: ogr.Geometry, ref: str = "min_rect") -> tuple[float, float, float]:
    """Returns (width, height, rect_area). ref can be 'min_rect' or 'bbox'."""
    if ref == "bbox":
        env = geom.GetEnvelope()
        w = max(float(env[1] - env[0]), 0.0)
        h = max(float(env[3] - env[2]), 0.0)
        return w, h, w * h

    try:
        poly = Polygon(geom.GetGeometryRef(0).GetPoints())
        if not poly.is_valid or poly.area <= 0:
            return 0.0, 0.0, 0.0
        
        min_rect = poly.minimum_rotated_rectangle
        distance_edges = min_rect.exterior.coords[:-1]
        if len(distance_edges) != 4:
            return 0.0, 0.0, 0.0
        edge_lengths = [np.linalg.norm(np.array(distance_edges[i]) - np.array(distance_edges[(i + 1) % 4])) for i in range(4)]
        edge_lengths.sort()
        w = edge_lengths[0]
        h = edge_lengths[2]
        return w, h, min_rect.area
    except Exception:
        return 0.0, 0.0, 0.0


def _shape_rectangularity_score(area: float, rect_area: float, shape_beta: float) -> float:
    if rect_area <= 0:
        return 0.0
    ratio = area / rect_area
    return _safe_exp_decay(ratio - 1.0, shape_beta)


def _combine_scores(
    area_score: float,
    ratio_score: float,
    shape_score: float,
    score_cfg: Dict[str, Any],
) -> float:
    combine_mode = str(score_cfg.get("combine_mode", "weighted")).lower()
    if combine_mode == "product":
        gamma = float(score_cfg.get("gamma", 1.0))
        product_use_shape = bool(score_cfg.get("product_use_shape", True))
        fused = area_score * shape_score if product_use_shape else area_score
        fused = max(fused, 0.0)
        return float(fused**gamma)

    w_area = float(score_cfg.get("w_area", 1.0))
    w_ratio = float(score_cfg.get("w_ratio", 1.0))
    w_shape = float(score_cfg.get("w_shape", 1.0))
    wsum = max(w_area + w_ratio + w_shape, 1e-6)
    return (w_area * area_score + w_ratio * ratio_score + w_shape * shape_score) / wsum


def compute_pv_geometry_score(
    geom: ogr.Geometry,
    score_cfg: Dict[str, Any],
) -> Dict[str, float]:
    area = max(float(geom.Area()), 0.0)
    rect_mode = str(score_cfg.get("rect_mode", "min_rect")).lower()
    w, h, rect_area = _get_rect_info(geom, ref=rect_mode)
    ratio = max(w / h, h / w) if w > 0 and h > 0 else 0.0

    # -- Size Target --
    area_target = float(score_cfg.get("area_target", -1.0))
    if area_target <= 0:
        target_length = float(score_cfg.get("target_length", 2.2))
        target_width = float(score_cfg.get("target_width", 1.1))
        area_target = max(target_length * target_width, 1e-6)

    # -- Ratio Target --
    mu_r = float(score_cfg.get("mu_r", -1.0))
    if mu_r <= 0:
        target_length = float(score_cfg.get("target_length", 2.2))
        target_width = float(score_cfg.get("target_width", 1.1))
        mu_r = max(target_length / max(target_width, 1e-6), 1e-6)

    # -- Betas / Sigmas --
    # Handle both new (_beta) and old configs
    area_beta = float(score_cfg.get("area_beta", score_cfg.get("beta", 0.35)))
    sigma_r = float(score_cfg.get("sigma_r", score_cfg.get("ratio_beta", 0.2)))
    shape_beta = float(score_cfg.get("shape_beta", 0.1))

    # -- Individual Scores --
    area_score = _safe_exp_decay(area - area_target, area_target * area_beta)
    ratio_score = _safe_gaussian(ratio - mu_r, sigma_r)
    shape_score = _shape_rectangularity_score(area, rect_area, shape_beta=shape_beta)

    # -- Combined Score --
    con_pv = _combine_scores(area_score, ratio_score, shape_score, score_cfg)

    return {
        "area": area,
        "aspect_ratio": ratio,
        "area_score": area_score,
        "ratio_score": ratio_score,
        "shape_score": shape_score,
        "con_pv": float(con_pv),
    }
