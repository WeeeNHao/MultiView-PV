from __future__ import annotations

import math
from typing import Any, Dict

from osgeo import ogr


def _safe_exp_decay(delta: float, beta: float) -> float:
    beta = max(beta, 1e-6)
    return float(math.exp(-abs(delta) / beta))


def _safe_gaussian(delta: float, sigma: float) -> float:
    sigma = max(sigma, 1e-6)
    return float(math.exp(-((delta * delta) / (2.0 * sigma * sigma))))


def _aspect_ratio_from_envelope(geom: ogr.Geometry) -> float:
    env = geom.GetEnvelope()
    width = max(float(env[1] - env[0]), 0.0)
    height = max(float(env[3] - env[2]), 0.0)
    if width <= 0 or height <= 0:
        return 0.0
    return max(width / height, height / width)


def _shape_rectangularity_score(geom: ogr.Geometry, shape_beta: float) -> float:
    env = geom.GetEnvelope()
    width = max(float(env[1] - env[0]), 0.0)
    height = max(float(env[3] - env[2]), 0.0)
    rect_area = width * height
    if rect_area <= 0:
        return 0.0

    area = max(float(geom.Area()), 0.0)
    rectangularity = min(max(area / rect_area, 0.0), 1.0)
    return _safe_exp_decay(1.0 - rectangularity, shape_beta)


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
    ratio = _aspect_ratio_from_envelope(geom)

    mode = str(score_cfg.get("mode", "standard")).lower()
    if mode in {"legacy", "legacy_gaussian", "legacy_weighted"}:
        area_target = float(score_cfg.get("area_target", 2.42))
        if area_target <= 0:
            target_length = float(score_cfg.get("target_length", 2.2))
            target_width = float(score_cfg.get("target_width", 1.1))
            area_target = max(target_length * target_width, 1e-6)

        mu_r = float(score_cfg.get("mu_r", 2.03))
        sigma_r = float(score_cfg.get("sigma_r", 0.2))
        beta = float(score_cfg.get("beta", 0.25))
        shape_beta = float(score_cfg.get("shape_beta", 0.1))

        area_score = _safe_exp_decay(area - area_target, area_target * beta)
        ratio_score = _safe_gaussian(ratio - mu_r, sigma_r)
        shape_score = _shape_rectangularity_score(geom, shape_beta=shape_beta)
        con_pv = _combine_scores(area_score, ratio_score, shape_score, score_cfg)

        return {
            "area": area,
            "aspect_ratio": ratio,
            "area_score": area_score,
            "ratio_score": ratio_score,
            "shape_score": shape_score,
            "con_pv": float(con_pv),
        }

    target_length = float(score_cfg.get("target_length", 2.2))
    target_width = float(score_cfg.get("target_width", 1.1))
    target_area = max(target_length * target_width, 1e-6)
    target_ratio = max(target_length / max(target_width, 1e-6), 1e-6)

    area_beta = float(score_cfg.get("area_beta", 0.8)) * target_area
    ratio_beta = float(score_cfg.get("ratio_beta", 0.35)) * target_ratio
    shape_beta = float(score_cfg.get("shape_beta", 0.2))

    area_score = _safe_exp_decay(area - target_area, area_beta)
    ratio_score = _safe_exp_decay(ratio - target_ratio, ratio_beta)
    shape_score = _shape_rectangularity_score(geom, shape_beta=shape_beta)
    con_pv = _combine_scores(area_score, ratio_score, shape_score, score_cfg)
    return {
        "area": area,
        "aspect_ratio": ratio,
        "area_score": area_score,
        "ratio_score": ratio_score,
        "shape_score": shape_score,
        "con_pv": float(con_pv),
    }
