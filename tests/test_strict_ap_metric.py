"""The strict high-IoU AP added because mAP@[.50:.95] is saturated here.

On BeiOu/XinXie object F1 is identical from IoU 0.50 to 0.90, so the COCO sweep
cannot separate configurations and the benefit of iterative refinement -- which
is boundary accuracy, not detection -- is invisible. AP@{.90,.95,.975} is the
headline metric instead. These tests pin down two things that are easy to break:
the extra threshold really is swept, and adding it did not redefine mAP.
"""

from evaluation.metrics import (
    COCO_IOU_THRESHOLDS,
    EVAL_IOU_THRESHOLDS,
    HIGH_IOU_THRESHOLDS,
    evaluate_feature_lists,
)


def _rect(x0: float, y0: float, x1: float, y1: float, score: float = 1.0) -> dict:
    """Axis-aligned rectangle. `segmentation` rings are flat [x0,y0,x1,y1,...]."""
    return {
        "segmentation": [[x0, y0, x1, y0, x1, y1, x0, y1]],
        "bbox": [x0, y0, x1, y1],
        "con_weight": score,
    }


def _evaluate_at_iou(iou: float):
    """One GT square against one prediction narrowed to hit exactly `iou`."""
    gt = [_rect(0.0, 0.0, 10.0, 10.0)]
    pred = [_rect(0.0, 0.0, 10.0 * iou, 10.0)]
    return evaluate_feature_lists(preds=pred, gts=gt, score_field="con_weight")


def test_strict_thresholds_are_actually_swept():
    assert 0.975 in EVAL_IOU_THRESHOLDS
    assert set(COCO_IOU_THRESHOLDS) <= set(EVAL_IOU_THRESHOLDS)
    assert set(HIGH_IOU_THRESHOLDS) <= set(EVAL_IOU_THRESHOLDS)

    r = _evaluate_at_iou(1.0)
    swept = [round(t.iou_threshold, 3) for t in r.per_threshold]
    assert swept == sorted(swept), "per-threshold report must stay ordered"
    assert 0.975 in swept


def test_map_definition_unchanged_by_the_extra_threshold():
    """mAP must remain the mean over exactly the COCO ten.

    A prediction at true IoU 0.90 clears 9 of the 10 COCO thresholds, so
    mAP == 0.9 regardless of how many strict thresholds the sweep also visits.
    Averaging over the swept set instead would give 9/11.
    """
    r = _evaluate_at_iou(0.90)
    assert r.mAP == 0.9


def test_strict_ap_resolves_what_map_cannot():
    loose = _evaluate_at_iou(0.90)
    tight = _evaluate_at_iou(0.98)

    # mAP saturates: both are perfect or near-perfect detections.
    assert tight.mAP == 1.0
    assert loose.mAP == 0.9

    # The strict AP separates them by a wide margin.
    assert tight.ap_high == 1.0
    assert loose.ap_high < 0.4
    assert tight.ap_high - loose.ap_high > 0.6


def test_strict_ap_is_the_mean_of_its_three_components():
    for iou in (0.88, 0.93, 0.96, 0.99):
        r = _evaluate_at_iou(iou)
        expected = (r.ap90 + r.ap95 + r.ap975) / 3.0
        assert abs(r.ap_high - expected) < 1e-9, iou


def test_components_are_monotone_in_threshold():
    """At a fixed prediction quality, a stricter threshold cannot score higher."""
    r = _evaluate_at_iou(0.96)
    assert r.ap90 >= r.ap95 >= r.ap975
