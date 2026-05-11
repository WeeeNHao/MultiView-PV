import pytest
import numpy as np

from projection.collinearity import (
    build_rotation,
    photo_to_ground,
    ground_to_photo,
)
from projection.oblique_projector import ObliqueProjector


def test_oblique_projector_forces_affine_when_requested():
    projector = ObliqueProjector.__new__(ObliqueProjector)
    projector.method = "affine"
    projector.min_control_points = 999
    projector.enable_slope_correction = False
    projector._resolve_pose = lambda image_path: [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
    projector._build_affine_pairs = lambda bbox, pose: (
        [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
    )

    feature = {
        "segmentation": [[0.0, 0.0, 2.0, 0.0, 2.0, 1.0]],
        "bbox": [0.0, 0.0, 2.0, 1.0],
    }

    out = projector.project_feature(feature, "dummy.jpg")

    assert out["projection_method"] == "affine"
    assert out["segmentation"] == feature["segmentation"]


def test_oblique_projector_forces_collinearity_when_requested():
    projector = ObliqueProjector.__new__(ObliqueProjector)
    projector.method = "collinearity"
    projector.min_control_points = 999
    projector.enable_slope_correction = False
    projector._resolve_pose = lambda image_path: [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]

    def _unexpected_affine_pairs(bbox, pose):
        raise AssertionError("affine path must not be used when method=collinearity")

    projector._build_affine_pairs = _unexpected_affine_pairs
    projector._project_points_direct_collinearity = lambda points, pose: [(x + 1.0, y + 2.0) for x, y in points]

    feature = {
        "segmentation": [[0.0, 0.0, 2.0, 0.0, 2.0, 1.0]],
        "bbox": [0.0, 0.0, 2.0, 1.0],
    }

    out = projector.project_feature(feature, "dummy.jpg")

    assert out["projection_method"] == "collinearity"
    assert out["segmentation"] == [[1.0, 2.0, 3.0, 2.0, 3.0, 3.0]]


def test_oblique_projector_rejects_forced_affine_without_enough_control_points():
    projector = ObliqueProjector.__new__(ObliqueProjector)
    projector.method = "affine"
    projector.min_control_points = 999
    projector.enable_slope_correction = False
    projector._resolve_pose = lambda image_path: [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
    projector._build_affine_pairs = lambda bbox, pose: ([(0.0, 0.0), (1.0, 0.0)], [(0.0, 0.0), (1.0, 0.0)])

    feature = {
        "segmentation": [[0.0, 0.0, 2.0, 0.0, 2.0, 1.0]],
        "bbox": [0.0, 0.0, 2.0, 1.0],
    }

    with pytest.raises(ValueError, match="projection.oblique.method=affine requires at least 3 valid control points"):
        projector.project_feature(feature, "dummy.jpg")

def test_collinearity_roundtrip():
    # Camera parameters
    f = 3000.0
    xs, ys, zs = 1000.0, 2000.0, 500.0
    phi, omega, kappa = 0.05, -0.05, 0.1
    rot = build_rotation(phi, omega, kappa)

    # Pixel coordinates
    px, py = 500.0, -300.0
    z_ground = 50.0

    # photo -> ground
    gx, gy = photo_to_ground(px, py, f, z_ground, xs, ys, zs, rot)

    # ground -> photo
    px_back, py_back = ground_to_photo(f, gx, gy, z_ground, xs, ys, zs, rot)

    assert np.isclose(px, px_back, atol=1e-3)
    assert np.isclose(py, py_back, atol=1e-3)

def test_build_rotation_identity():
    rot = build_rotation(0, 0, 0)
    # Expected identity matrix elements flat:
    # a1=1, a2=0, a3=0, b1=0, b2=1, b3=0, c1=0, c2=0, c3=1
    expected = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    for r, e in zip(rot, expected):
        assert np.isclose(r, e, atol=1e-5)
