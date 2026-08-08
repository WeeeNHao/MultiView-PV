"""Regression test: the slope_correction plane fit must be deterministic and
must land on the surface the feature actually sits on.

Two defects combine here, both visible on XinXie ``DJI_20251124151039_0355_V``:

1. ``_sample_plane_points`` takes the axis-aligned ground bounding box of the
   four projected bbox corners. At grazing incidence (~30 deg here, horizontal
   stretch 3.4x) that rectangle is far wider than the panel, so for some
   features it straddles the neighbouring panel row as well. Panel rows are a
   sawtooth -- each row tilts up across its width, then the next row restarts
   lower -- so the two rows are mutually exclusive under a single plane and form
   two competing RANSAC consensus sets. id 89 sampled 24383 cells spanning two
   rows; id 191, one row, sampled 13844.

2. ``RANSACRegressor`` was constructed without ``random_state``, so it drew from
   the global RNG. On an ambiguous two-row cloud every call returned a different
   plane: over 10 identical fits id 89's tilt ranged 2.24-6.92 deg and id 180's
   2.56-7.43 deg, while the unambiguous id 140 / id 191 clouds were stable to
   0.00 deg. That non-determinism is what made
   ``reproduction_max_vertex_diff_m`` 4-5 m for id 89/180 (the diagnostic's own
   fit and ``project_feature``'s internal fit disagreed) and 0.00 m for 140/191.

Consequence: id 89/180's footprint landed ~1.65 m off, GT IoU 0.09 instead of
~0.92, and which way it fell varied per run.

Fixtures are the real DSM sample clouds dumped by
analysis/DJI_20251124151039_0355_V_slope_debug/analyze.py.
"""
import math

import numpy as np
import pytest

from projection.oblique_projector import ObliqueProjector

DATA = __file__.rsplit("/", 1)[0] + "/data"

# The DSM point the feature's own centroid ray hits -- 0.28 m from GT ID 1750.
# (summary.json "centroid_world_dsm" for id 89.)
ID89_ANCHOR = (380309.20543169795, 3306116.3896045485, 37.872100830078125)

# Panels on this site sit at ~16-17.5 deg; id 191's clean single-row cloud
# fits 17.19 deg, and 6bfabe7 projected id 89 correctly at 16.9 deg.
PANEL_TILT_RANGE = (14.0, 20.0)


def _samples(fid):
    f = np.load(f"{DATA}/xinxie_0355_id{fid}_dsm_samples.npz")
    gx = f["dx"].astype(np.float64) + float(f["ox"])
    gy = f["dy"].astype(np.float64) + float(f["oy"])
    return gx, gy, f["z"].astype(np.float64)


def _make_projector():
    p = ObliqueProjector.__new__(ObliqueProjector)
    p.sc_ransac_threshold = 0.3
    p.sc_min_inliers = 5
    p.sc_ransac_max_trials = 100
    p.sc_max_tilt_deg = 60.0
    p.sc_ransac_random_state = 0
    p.sc_anchor_max_candidates = 4
    p.sc_anchor_radius_m = 2.0
    return p


def _tilt_deg(plane):
    a, b = plane[0], plane[1]
    return math.degrees(math.acos(1.0 / math.sqrt(1.0 + a * a + b * b)))


def _anchor_offset(plane, anchor):
    a, b, c = plane[:3]
    ax, ay, az = anchor
    return abs(a * ax + b * ay + c - az)


def test_fixture_reproduces_the_two_row_ambiguity():
    """Precondition: id 89's cloud spans two rows, id 191's spans one."""
    _, gy89, gz89 = _samples(89)
    _, _, gz191 = _samples(191)
    assert gz89.size > 1.6 * gz191.size, (gz89.size, gz191.size)
    # a gap with no samples separates the two rows
    hist, edges = np.histogram(gy89, bins=40)
    assert (hist == 0).sum() >= 5, hist


@pytest.mark.parametrize("fid", [89, 191])
def test_plane_fit_is_deterministic(fid):
    """Repeated fits of the same cloud must return the same plane."""
    p = _make_projector()
    gx, gy, gz = _samples(fid)
    planes = {tuple(round(v, 9) for v in p._fit_plane_ransac(gx, gy, gz)[:3])
              for _ in range(8)}
    assert len(planes) == 1, planes


def test_fit_lands_on_the_surface_under_the_feature():
    """id 89: the chosen plane must contain the feature's own DSM point and
    carry the panel's tilt, not a compromise across two rows."""
    p = _make_projector()
    gx, gy, gz = _samples(89)
    plane = p._fit_plane_ransac(gx, gy, gz, anchor=ID89_ANCHOR)
    assert plane is not None
    assert _anchor_offset(plane, ID89_ANCHOR) <= p.sc_ransac_threshold
    assert PANEL_TILT_RANGE[0] <= _tilt_deg(plane) <= PANEL_TILT_RANGE[1], _tilt_deg(plane)


def test_clean_single_row_fit_is_not_disturbed():
    """id 191 already resolves correctly; anchoring must not change it."""
    p = _make_projector()
    gx, gy, gz = _samples(191)
    base = p._fit_plane_ransac(gx, gy, gz)
    assert PANEL_TILT_RANGE[0] <= _tilt_deg(base) <= PANEL_TILT_RANGE[1]
    # an anchor on that same surface leaves the answer alone
    ax, ay = float(np.median(gx)), float(np.median(gy))
    az = base[0] * ax + base[1] * ay + base[2]
    anchored = p._fit_plane_ransac(gx, gy, gz, anchor=(ax, ay, az))
    assert abs(_tilt_deg(anchored) - _tilt_deg(base)) <= 1.0
    assert _anchor_offset(anchored, (ax, ay, az)) <= p.sc_ransac_threshold


def test_unreachable_anchor_falls_back_instead_of_dropping_the_feature():
    """If no candidate surface contains the anchor, still return a plane."""
    p = _make_projector()
    gx, gy, gz = _samples(89)
    ax, ay, az = ID89_ANCHOR
    plane = p._fit_plane_ransac(gx, gy, gz, anchor=(ax, ay, az + 50.0))
    assert plane is not None
