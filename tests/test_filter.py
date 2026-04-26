import pytest
import numpy as np
from shapely.geometry import Polygon

from postprocess.filter import (
    max_polygon_angle,
    _safe_aspect_ratio,
    _shape_rectangularity,
    pv_consistency_score,
    filter_features,
)


def test_max_polygon_angle():
    # A perfect square
    poly = Polygon([(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)])
    angle = max_polygon_angle(poly)
    assert np.isclose(angle, 90.0, atol=1e-1)

    # A triangle with a 135-degree angle
    poly2 = Polygon([(0, 0), (10, 0), (5, 5), (0, 0)])
    angle2 = max_polygon_angle(poly2)
    assert np.isclose(angle2, 135.0, atol=1e-1)


def test_safe_aspect_ratio():
    # 10x20 rectangle
    poly = Polygon([(0, 0), (0, 10), (20, 10), (20, 0), (0, 0)])
    r = _safe_aspect_ratio(poly)
    assert np.isclose(r, 2.0, atol=1e-1)


def test_shape_rectangularity():
    # A perfect rectangle
    poly = Polygon([(0, 0), (0, 10), (20, 10), (20, 0), (0, 0)])
    score = _shape_rectangularity(poly)
    assert np.isclose(score, 1.0, atol=1e-1)

    # A diamond/rotated square
    poly2 = Polygon([(10, 0), (20, 10), (10, 20), (0, 10), (10, 0)])
    score2 = _shape_rectangularity(poly2, ref="min_rect")
    assert np.isclose(score2, 1.0, atol=1e-1)  # Since minimum rotated rect is the square itself


def test_pv_consistency_score():
    poly = Polygon([(0, 0), (0, 10), (20, 10), (20, 0), (0, 0)])
    score = pv_consistency_score(
        poly=poly,
        area_target=200.0,
        mu_r=2.0,
        sigma_r=0.2,
        beta=0.25,
    )
    # Since area=200, ratio=2.0, rectangularity=1.0, everything is perfect
    assert np.isclose(score, 1.0, atol=1e-1)


def test_filter_features():
    # Create two features, one good, one bad
    feat_good = {
        "bbox": [0, 0, 20, 10],
        "segmentation": [[0, 0, 0, 10, 20, 10, 20, 0, 0, 0]],
        "score": 0.9,
        "con_weight": 0.9,
    }
    
    # Area is 10x10=100 (half of target), ratio=1.0 (target is 2.0)
    feat_bad = {
        "bbox": [0, 0, 10, 10],
        "segmentation": [[0, 0, 0, 10, 10, 10, 10, 0, 0, 0]],
        "score": 0.8,
        "con_weight": 0.8,
    }

    features = [feat_good, feat_bad]
    
    cfg = {
        "area_target": 200.0,
        "score_threshold": 0.5,
        "score_field": "con_weight",
    }
    
    # Mocking _feature_to_polygon temporarily by putting the poly directly or letting the actual logic run.
    # The actual logic in filter.py uses `_segmentation_to_polygon` which relies on `osgeo.ogr`.
    # Let's let it run and see if ogr works correctly in the test environment.
    filtered = filter_features(features, cfg)
    
    # We expect feat_good to have a high score, and feat_bad to have a lower score.
    # Or feat_bad might even be filtered out if its score drops below 0.5
    assert len(filtered) >= 1
    
    scores = [f["con_weight"] for f in filtered]
    # The first feature should have a higher score than the second (if it survived)
    if len(filtered) == 2:
        assert scores[0] > scores[1]
