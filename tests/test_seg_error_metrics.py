"""Tests for the three metrics the draft tables need but the evaluator lacked:
centroid RMSE (tables 1/5) and over-/under-segmentation (tables 4/6).

Over- and under-segmentation cannot be read off the one-to-one match: a GT panel
split into two predictions shows up there as 1 TP + 1 FP, indistinguishable from
a plain false positive next to a correct detection. They are counted by
containment instead, which is also why an IoU-based rule would not work -- two
predictions splitting one GT can never both reach IoU 0.5 with it.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import evaluate_feature_lists  # noqa: E402


def _rect(x0, y0, x1, y1, score=1.0):
    """A feature dict shaped like what ``read_features_from_shapefile`` yields.

    Rings are flat ``[x0, y0, x1, y1, ...]`` sequences, which is what
    ``postprocess.nms._flat_to_pairs`` parses; a list of ``[x, y]`` pairs is
    silently rejected and the feature falls back to bbox-only overlap.
    """
    return {
        "segmentation": [[x0, y0, x1, y0, x1, y1, x0, y1, x0, y0]],
        "bbox": [x0, y0, x1, y1],
        "con_weight": score,
    }


def _ev(preds, gts):
    return evaluate_feature_lists(
        preds=preds, gts=gts, primary_iou=0.5, score_field="con_weight"
    )


def test_perfect_prediction_has_no_error():
    gts = [_rect(0, 0, 2, 1), _rect(0, 3, 2, 4)]
    r = _ev([_rect(0, 0, 2, 1), _rect(0, 3, 2, 4)], gts)

    assert r.tp == 2 and r.fp == 0 and r.fn == 0
    assert r.centroid_rmse == 0.0
    assert r.over_seg_count == 0 and r.under_seg_count == 0


def test_split_gt_counts_as_over_segmentation():
    # One 2x1 GT panel, predicted as two 1x1 halves. Each half lies entirely
    # inside the GT, so the GT is over-segmented once.
    gts = [_rect(0, 0, 2, 1)]
    preds = [_rect(0, 0, 1, 1), _rect(1, 0, 2, 1)]
    r = _ev(preds, gts)

    assert r.over_seg_count == 1
    assert r.over_seg_rate == 1.0  # 1 of 1 GT
    assert r.under_seg_count == 0
    # The one-to-one match reports 1 TP + 1 FP here -- identical to a correct
    # detection sitting next to an unrelated false positive. That ambiguity is
    # exactly what the containment rule exists to resolve.
    assert (r.tp, r.fp) == (1, 1)


def test_merged_gts_count_as_under_segmentation():
    # Two adjacent GT panels swallowed by a single prediction spanning both.
    gts = [_rect(0, 0, 1, 1), _rect(1, 0, 2, 1)]
    preds = [_rect(0, 0, 2, 1)]
    r = _ev(preds, gts)

    assert r.under_seg_count == 1
    assert r.under_seg_rate == 1.0  # 1 of 1 prediction
    assert r.over_seg_count == 0


def test_a_plain_false_positive_is_neither():
    gts = [_rect(0, 0, 2, 1)]
    preds = [_rect(0, 0, 2, 1), _rect(10, 10, 12, 11)]
    r = _ev(preds, gts)

    assert r.fp == 1
    assert r.over_seg_count == 0 and r.under_seg_count == 0


def test_centroid_rmse_is_the_quadratic_mean_of_offsets():
    gts = [_rect(0, 0, 2, 1), _rect(0, 10, 2, 11)]
    # First prediction shifted 0.1 m in x, second 0.3 m -- both still IoU > 0.5.
    preds = [_rect(0.1, 0, 2.1, 1), _rect(0.3, 10, 2.3, 11)]
    r = _ev(preds, gts)

    assert r.tp == 2
    expected = math.sqrt((0.1 ** 2 + 0.3 ** 2) / 2)
    assert abs(r.centroid_rmse - expected) < 1e-9
    assert abs(r.centroid_mae - 0.2) < 1e-9
    # RMSE must exceed MAE whenever the offsets differ -- that asymmetry is the
    # reason the table asks for RMSE rather than a mean.
    assert r.centroid_rmse > r.centroid_mae


def test_empty_inputs_do_not_divide_by_zero():
    assert _ev([], [_rect(0, 0, 1, 1)]).over_seg_rate == 0.0
    assert _ev([_rect(0, 0, 1, 1)], []).under_seg_rate == 0.0
    r = _ev([], [])
    assert r.centroid_rmse == 0.0 and r.over_seg_rate == 0.0
