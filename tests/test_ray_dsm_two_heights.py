"""Regression test for sites with more than one mounting height.

004-CangFang carries PV on two surfaces ~9 m apart (ground array ~92 m,
warehouse roof ~102 m) with terrain dropping to ~79 m around them. The first
nearest-hit implementation anchored its search band on a single global scalar
(``_dsm_global_median``, in practice the 9x9 window at the DSM's *centre*
pixel) and only looked +-``ray_dsm_march_up/down`` around it, so it could only
reach one height regime: rays aimed at the roof came back on a phantom surface
20+ m below.

The band must instead be derived from the DSM's actual elevation range, which
carries no assumption about how many surfaces the site has.

See analysis/cangfang_two_heights/FINDINGS.md.
"""
import numpy as np

from projection.oblique_projector import ObliqueProjector
from projection.collinearity import build_rotation, photo_to_ground, geo_to_image_xy


# --- synthetic two-height scene -------------------------------------------
# world x = col, world y = 100 - row   (same convention as the single-step test)
GT = (0.0, 1.0, 0.0, 100.0, 0.0, -1.0)
POSE = [50.0, 0.0, 133.0, 0.0, 0.6, 0.0]     # xs, ys, zs, phi, omega, kappa
PIXEL = (50.0, 10.0)                          # px=0 (x-centred), py=40

LOW = 79.5          # surrounding terrain, and the far crossing of the ray
ROOF = 102.0        # the surface the camera actually sees first
ROOF_ROWS = slice(45, 58)


def _make_dsm():
    dsm = np.full((120, 120), LOW, dtype=np.float64)
    dsm[ROOF_ROWS, :] = ROOF
    return dsm


def _make_projector():
    p = ObliqueProjector.__new__(ObliqueProjector)
    p.cx, p.cy, p.focal = 50.0, 50.0, 100.0
    p._dsm_array = _make_dsm()
    p._dsm_ds = None
    p._dsm_ds_nodata = -9999.0
    p._dsm_h, p._dsm_w = p._dsm_array.shape
    p._dsm_geo = GT
    p._dsm_geo_inv = None
    p.ray_dsm_tol = 0.05
    p.ray_dsm_init_window = 1
    p._dsm_global_median = float(p._dsm_array[60, 60])   # centre pixel: the LOW surface
    p._last_dsm_z = None
    p.ray_dsm_march_up = 4.0        # deprecated, must no longer bound the search
    p.ray_dsm_march_down = 4.0
    p.ray_dsm_march_step = 0.25
    p.ray_dsm_z_margin = 2.0
    p.ray_dsm_max_steps = 4096
    p._dsm_z_min = float(np.min(p._dsm_array))
    p._dsm_z_max = float(np.max(p._dsm_array))
    return p


def _banded_march(p, img_x, img_y, pose):
    """The previous implementation: search band anchored on a single global
    scalar and only ``march_up``/``march_down`` wide. Kept here to prove the
    synthetic scene actually reproduces the CangFang failure."""
    xs, ys, zs, phi, omega, kappa = pose
    rot = build_rotation(phi, omega, kappa)
    px, py = img_x - p.cx, p.cy - img_y

    gx0, gy0 = photo_to_ground(px, py, p.focal, p._dsm_global_median, xs, ys, zs, rot)
    c0, r0 = geo_to_image_xy(p._dsm_geo, gx0, gy0)
    z_center = p._dsm_window_median(int(round(c0)), int(round(r0)), p.ray_dsm_init_window)
    if z_center is None:
        return None

    z_floor = z_center - p.ray_dsm_march_down
    z_start = min(float(zs), z_center + p.ray_dsm_march_up)
    for _ in range(64):
        if z_start >= float(zs):
            break
        _, _, dz = p._ray_surface_probe(px, py, z_start, xs, ys, zs, rot)
        if dz is None or z_start - dz > 0.0:
            break
        z_start = min(float(zs), z_start + p.ray_dsm_march_up)

    prev_z = prev_diff = None
    z = z_start
    while z >= z_floor:
        _, _, dz = p._ray_surface_probe(px, py, z, xs, ys, zs, rot)
        if dz is None:
            prev_z = prev_diff = None
        else:
            diff = z - dz
            if diff == 0.0:
                return round(float(dz), 2)
            if prev_diff is not None and prev_diff > 0.0 and diff < 0.0:
                return round(float(dz), 2)
            prev_z, prev_diff = z, diff
        z -= p.ray_dsm_march_step
    return None


def test_scene_has_two_reachable_surfaces():
    """Precondition: this ray really does cross both surfaces, and the near
    one (the roof) is the physically correct answer."""
    p = _make_projector()
    xs, ys, zs, phi, omega, kappa = POSE
    rot = build_rotation(phi, omega, kappa)
    px, py = PIXEL[0] - p.cx, p.cy - PIXEL[1]

    seen = set()
    for z in np.arange(zs, LOW - 1.0, -0.5):
        _, _, dz = p._ray_surface_probe(px, py, float(z), xs, ys, zs, rot)
        if dz is not None:
            seen.add(round(dz, 1))
    assert ROOF in seen and LOW in seen, seen
    # the DSM centre pixel -- what the old code anchored on -- is the LOW one
    assert p._dsm_global_median == LOW


def test_old_banded_march_misses_the_near_surface():
    """The +-4 m band anchored on the global scalar returns the far/low
    surface, ~22 m below the surface the camera actually sees."""
    got = _banded_march(_make_projector(), *PIXEL, POSE)
    assert got == LOW
    assert abs(got - ROOF) > 20.0


def test_nearest_hit_reaches_the_upper_surface():
    p = _make_projector()
    hit = p._ray_dsm_intersection(*PIXEL, POSE)
    assert hit is not None
    assert abs(hit[2] - ROOF) <= 0.1, hit

    # the returned point is a real intersection
    gx, gy, z = hit
    col, row = geo_to_image_xy(GT, gx, gy)
    assert abs(p._read_dsm_value(int(round(col)), int(round(row))) - z) <= 0.1


def test_both_height_regimes_are_reachable():
    """A pixel whose ray only ever sees the low terrain must still resolve to
    it -- the fix must not simply bias everything upward."""
    p = _make_projector()
    p._dsm_array[ROOF_ROWS, :] = LOW          # remove the roof entirely
    p._dsm_z_min = p._dsm_z_max = LOW
    hit = p._ray_dsm_intersection(*PIXEL, POSE)
    assert hit is not None
    assert abs(hit[2] - LOW) <= 0.1, hit


def test_march_is_independent_of_cross_feature_state():
    p = _make_projector()
    results = []
    for seed in [None, 79.0, 92.0, 102.0, 133.0]:
        p._last_dsm_z = seed
        hit = p._ray_dsm_intersection(*PIXEL, POSE)
        assert hit is not None
        results.append(tuple(round(v, 3) for v in hit))
    assert len(set(results)) == 1, results
