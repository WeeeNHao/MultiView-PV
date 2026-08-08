"""Regression test: a feature whose projection failed must never reach output.

``ObliqueProjector.project_feature`` signals failure by tagging
``projection_method`` and returning the feature *unchanged* -- meaning its
segmentation is still in image pixel coordinates. ``project_and_score_features``
is the only place that drops those, and it compared against the single literal
``"affine_failed"`` while the oblique projector also emits
``"slope_correction_failed"`` from four separate sites.

Nothing caught it because the module-geometry prior masks the leak: a pixel
polygon spans thousands of units, so its area score collapses, ``con_pv`` goes
to zero, and ``score_threshold`` removed it one branch further down. The mask
disappears exactly in the table-4 ablations that switch the prior off -- BeiOu
``abl_noprior/iter_0`` shipped 51 of 5489 features at pixel coordinates, which
stretched the layer extent from (0, 243) to (370455, 3313053) and made the
shapefile impossible to view in QGIS.
"""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import projection.projector as projector  # noqa: E402


def _feature(x0, y0, x1, y1):
    return {
        "segmentation": [[x0, y0, x1, y0, x1, y1, x0, y1, x0, y0]],
        "bbox": [x0, y0, x1, y1],
        "con": 0.99,
    }


def _run(monkeypatch_target, outcomes, score_threshold=0.0):
    """Project two features, with ``outcomes`` giving each one's method tag."""
    calls = {"n": 0}

    class _FakeProjector:
        def project_feature(self, feature, image_path):
            out = dict(feature)
            out["projection_method"] = outcomes[calls["n"]]
            calls["n"] += 1
            return out

    monkeypatch_target(_FakeProjector())

    cfg = {
        "mode": "oblique",
        "project_coordinates": True,
        "oblique": {},
        # No prior weighting, so nothing but the failure check can filter:
        # this is the table-4 "No module-geometry prior" configuration.
        "score": {"w_sem": 1.0, "w_pv": 0.0, "score_threshold": score_threshold},
    }
    feats = [_feature(0, 243, 4000, 3000), _feature(370400, 3312900, 370402, 3312901)]
    return projector.project_and_score_features(
        features=feats,
        geo_meta=SimpleNamespace(geotransform=None),
        projection_cfg=cfg,
        image_path="dummy.jpg",
    )


def test_slope_correction_failure_is_dropped(monkeypatch):
    def _set(fake):
        monkeypatch.setattr(projector, "_get_oblique_projector", lambda cfg: fake)

    out = _run(_set, ["slope_correction_failed", "slope_correction"])

    assert len(out) == 1, "the failed projection leaked into the output"
    assert out[0]["projection_method"] == "slope_correction"
    # The survivor is the one in map coordinates, not the pixel-space polygon.
    assert out[0]["bbox"][0] > 100000


def test_affine_failure_is_dropped(monkeypatch):
    def _set(fake):
        monkeypatch.setattr(projector, "_get_oblique_projector", lambda cfg: fake)

    out = _run(_set, ["affine_failed", "affine"])

    assert len(out) == 1
    assert out[0]["projection_method"] == "affine"


def test_successful_projections_all_survive(monkeypatch):
    def _set(fake):
        monkeypatch.setattr(projector, "_get_oblique_projector", lambda cfg: fake)

    out = _run(_set, ["slope_correction", "slope_correction"])

    assert len(out) == 2
