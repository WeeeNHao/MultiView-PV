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
    # compute_affine_transform solves via lstsq, so it recovers the identity
    # transform only to floating-point precision (residual ~1e-16). Comparing
    # exactly would fail on noise that carries no geometric meaning.
    assert out["segmentation"][0] == pytest.approx(feature["segmentation"][0])


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


def test_forced_affine_without_enough_control_points_is_tagged_failed():
    """Fewer than 3 pairs cannot determine an affine fit.

    The contract is tag-and-drop, not raise: the feature comes back marked
    ``affine_failed`` and *unchanged* -- still in image pixel coordinates -- and
    ``project_and_score_features`` is what removes it (see
    ``test_failed_projection_dropped.py``). Mapping it anyway with an
    underdetermined lstsq solution is what produced oversized affine footprints.

    This matters for the projection-method ablation: forcing
    ``projection.oblique.method=affine`` makes every such feature disappear, so
    that arm's recall loss is a real property of the method and must be counted,
    not mistaken for a pipeline bug.
    """
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

    out = projector.project_feature(feature, "dummy.jpg")

    assert out["projection_method"] == "affine_failed"
    assert out["segmentation"] == feature["segmentation"]

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
