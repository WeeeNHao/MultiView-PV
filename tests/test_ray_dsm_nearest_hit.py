"""Regression test for the ray/DSM bistability fix (nearest-hit ray march).

Builds a tiny synthetic DSM with a 2 m height step and an oblique camera whose
ray crosses the step, so the ray has two valid intersections (near/high panel
edge at z=12, far/low ground at z=10). The old seed-dependent fixed point
returned either one depending on the seed; the new march must deterministically
return the nearer (higher) surface regardless of any cross-feature state.

See analysis/DJI_20251124151131_0403_V_slope_debug/id_1695/z_ref_bistability.md.
"""
import numpy as np

from projection.oblique_projector import ObliqueProjector
from projection.collinearity import build_rotation, photo_to_ground, geo_to_image_xy


# --- synthetic scene -------------------------------------------------------
# gt: world x = col, world y = 100 - row  (so the ray's positive y stays in range)
GT = (0.0, 1.0, 0.0, 100.0, 0.0, -1.0)
POSE = [50.0, 0.0, 40.0, 0.0, 0.6, 0.0]        # xs, ys, zs, phi, omega, kappa
PIXEL = (50.0, 10.0)                            # px=0 (x-centred), py=40


def _make_dsm():
    # rows >= 58  -> world y < 43 -> HIGH surface (12 m); else LOW surface (10 m)
    dsm = np.full((120, 120), 10.0, dtype=np.float64)
    dsm[58:, :] = 12.0
    return dsm


def _make_projector():
    p = ObliqueProjector.__new__(ObliqueProjector)
    p.cx, p.cy, p.focal = 50.0, 50.0, 100.0
    p._dsm_array = _make_dsm()
    p._dsm_ds = None
    p._dsm_ds_nodata = -9999.0
    p._dsm_h, p._dsm_w = p._dsm_array.shape
    p._dsm_geo = GT
    p.ray_dsm_tol = 0.05
    p.ray_dsm_init_window = 1
    p._dsm_global_median = float(np.median(p._dsm_array))
    p._last_dsm_z = None
    p.ray_dsm_march_up = 4.0
    p.ray_dsm_march_down = 4.0
    p.ray_dsm_march_step = 0.25
    p._dsm_geo_inv = None
    p.ray_dsm_z_margin = 2.0
    p.ray_dsm_max_steps = 4096
    p._dsm_z_min = float(np.min(p._dsm_array))
    p._dsm_z_max = float(np.max(p._dsm_array))
    return p


def _old_fixed_point(p, img_x, img_y, pose, seed, max_iter=12, tol=0.25):
    """The pre-fix seed-dependent iteration, kept here to prove the synthetic
    scene actually reproduces the bistability the fix targets."""
    xs, ys, zs, phi, omega, kappa = pose
    rot = build_rotation(phi, omega, kappa)
    px, py = img_x - p.cx, p.cy - img_y
    z = float(seed)
    for _ in range(max_iter):
        gx, gy = photo_to_ground(px, py, p.focal, z, xs, ys, zs, rot)
        cf, rf = geo_to_image_xy(p._dsm_geo, gx, gy)
        dz = p._read_dsm_value(int(round(cf)), int(round(rf)))
        if dz is None:
            return None
        if abs(dz - z) <= tol:
            return round(float(dz), 3)
        z = float(dz)
    return None


def test_scene_is_bistable_under_old_iteration():
    """Precondition: the old fixed point returns different surfaces for
    different seeds -- i.e. this scene really exercises the bug."""
    p = _make_projector()
    low = _old_fixed_point(p, *PIXEL, POSE, seed=10.0)
    high = _old_fixed_point(p, *PIXEL, POSE, seed=13.0)
    assert low == 10.0
    assert high == 12.0
    assert low != high            # bistable -> order-dependent under old code


def test_nearest_hit_is_seed_independent_and_picks_the_near_surface():
    p = _make_projector()
    results = []
    for seed in [5.0, 9.0, 10.0, 11.0, 12.0, 13.0, 20.0, None]:
        p._last_dsm_z = seed        # must have no effect anymore
        hit = p._ray_dsm_intersection(*PIXEL, POSE)
        assert hit is not None
        results.append((round(hit[0], 2), round(hit[1], 2), round(hit[2], 2)))

    # all seeds agree (determinism) ...
    assert len(set(results)) == 1, results
    # ... and the agreed answer is the NEAR/high surface (z=12), not z=10.
    assert abs(results[0][2] - 12.0) <= 0.1, results[0]
    # returned point is a real intersection: DSM height there == returned z
    gx, gy, z = results[0]
    col, row = geo_to_image_xy(GT, gx, gy)
    assert abs(p._read_dsm_value(int(round(col)), int(round(row))) - z) <= 0.1


def test_single_surface_ray_unchanged():
    """A ray over a flat single-surface region must still resolve to that
    surface (the fix does not perturb the well-behaved majority)."""
    p = _make_projector()
    p._dsm_array[:] = 10.0          # flat, no step
    p._dsm_global_median = 10.0
    p._dsm_z_min = p._dsm_z_max = 10.0
    hit = p._ray_dsm_intersection(*PIXEL, POSE)
    assert hit is not None
    assert abs(hit[2] - 10.0) <= 0.1
