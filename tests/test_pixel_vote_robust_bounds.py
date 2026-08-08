"""A lone degenerate projection must not set the pixel-vote grid extent.

At XinXie one projected feature out of 150 610 landed 295 km from the site while
every other feature was within 71 m. Plain min/max bounds grew the grid to
62 x 541 km, MAX_GRID_CELLS then widened the cell from 0.20 m to 9.2 m -- larger
than a 1.10 x 2.20 m module -- and the whole array fused into two connected
components. Nothing logged the change, so the M2/M3 baselines were reported as
if they had run at 0.20 m.
"""

import numpy as np

from postprocess.pixel_fusion import fuse_pixel_vote, robust_bounds


def _square(cx: float, cy: float, half: float = 0.5) -> dict:
    ring = [cx - half, cy - half, cx + half, cy - half,
            cx + half, cy + half, cx - half, cy + half]
    return {
        "segmentation": [ring],
        "bbox": [cx - half, cy - half, cx + half, cy + half],
        "con_weight": 1.0,
    }


def _site(n: int = 200, spacing: float = 2.0) -> list:
    """A small regular array, the shape a real site has."""
    out = []
    side = int(np.sqrt(n))
    for i in range(side):
        for j in range(side):
            out.append(_square(i * spacing, j * spacing))
    return out


def test_single_outlier_does_not_set_the_extent():
    feats = _site()
    clean_bounds, clean_kept, clean_dropped = robust_bounds(feats)
    assert clean_dropped == 0
    assert len(clean_kept) == len(feats)

    feats_with_outlier = feats + [_square(295_000.0, 295_000.0)]
    bounds, kept, dropped = robust_bounds(feats_with_outlier)

    assert dropped == 1, "the 295 km feature must be rejected"
    assert len(kept) == len(feats)
    # Extent must stay the site's, not 295 km across.
    assert bounds is not None
    width = bounds[2] - bounds[0]
    assert width < 100.0, f"extent blew up to {width} m"
    assert bounds == clean_bounds


def test_outlier_no_longer_destroys_the_vote_grid():
    """End to end: the fused result must still resolve individual squares."""
    # Two overlapping burns per module so min_votes=2 is satisfiable.
    feats = _site() + _site()
    cfg = {"cell_size": 0.2, "min_votes": 2}

    before = fuse_pixel_vote(list(feats), cfg)
    after = fuse_pixel_vote(list(feats) + [_square(295_000.0, 295_000.0)], cfg)

    assert len(before) > 100, "sanity: the clean input should resolve instances"
    # Without robust bounds this collapsed to a handful of giant components.
    assert len(after) == len(before)


def test_disabling_restores_min_max_behaviour():
    """Reproducibility escape hatch for anything that needs the old extent."""
    feats = _site() + [_square(295_000.0, 295_000.0)]
    bounds, kept, dropped = robust_bounds(feats, iqr_k=0.0)
    assert dropped == 0
    assert len(kept) == len(feats)
    assert bounds is not None and (bounds[2] - bounds[0]) > 200_000.0


def test_tiny_inputs_are_left_alone():
    """Below the sample size where a percentile is meaningful, do not trim."""
    feats = [_square(0.0, 0.0), _square(2.0, 0.0), _square(500.0, 500.0)]
    bounds, kept, dropped = robust_bounds(feats)
    assert dropped == 0
    assert len(kept) == 3
