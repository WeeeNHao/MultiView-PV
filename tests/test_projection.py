import pytest
import numpy as np

from refactor_v2.projection.collinearity import (
    build_rotation,
    photo_to_ground,
    ground_to_photo,
)

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
