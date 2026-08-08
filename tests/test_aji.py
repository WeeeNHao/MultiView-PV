"""Aggregated Jaccard Index must actually penalise over- and under-segmentation.

The point of adding AJI is that F1/AP treat an instance as a single countable
unit, so they cannot express *how* wrong a fragmented result is -- 5866
half-module fragments and a clean miss both just score zero. AJI is area
weighted and charges every unclaimed prediction's full area to the denominator,
so it degrades smoothly. These tests pin that behaviour down.
"""

from evaluation.metrics import evaluate_feature_lists


def _rect(x0: float, y0: float, x1: float, y1: float, score: float = 1.0) -> dict:
    return {
        "segmentation": [[x0, y0, x1, y0, x1, y1, x0, y1]],
        "bbox": [x0, y0, x1, y1],
        "con_weight": score,
    }


def _aji(preds, gts):
    return evaluate_feature_lists(preds=preds, gts=gts, score_field="con_weight").aji


def test_perfect_prediction_scores_one():
    gts = [_rect(0, 0, 10, 10), _rect(20, 0, 30, 10)]
    assert abs(_aji(list(gts), gts) - 1.0) < 1e-9


def test_over_segmentation_is_penalised():
    """One GT split into two halves: the unclaimed half is charged in full."""
    gts = [_rect(0, 0, 10, 10)]
    whole = _aji([_rect(0, 0, 10, 10)], gts)
    split = _aji([_rect(0, 0, 5, 10), _rect(5, 0, 10, 10)], gts)
    assert whole == 1.0
    # best match covers half; the other half is unused and enters the denominator
    # AJI = 50 / (100 + 50) = 1/3
    assert abs(split - 1.0 / 3.0) < 1e-6
    assert split < whole


def test_more_fragments_score_worse():
    gts = [_rect(0, 0, 12, 10)]
    two = _aji([_rect(0, 0, 6, 10), _rect(6, 0, 12, 10)], gts)
    three = _aji([_rect(0, 0, 4, 10), _rect(4, 0, 8, 10), _rect(8, 0, 12, 10)], gts)
    assert three < two, "AJI must keep falling as fragmentation worsens"


def test_under_segmentation_is_penalised():
    """One prediction swallowing two GTs leaves the second GT unmatched."""
    gts = [_rect(0, 0, 10, 10), _rect(10, 0, 20, 10)]
    merged = _aji([_rect(0, 0, 20, 10)], gts)
    exact = _aji([_rect(0, 0, 10, 10), _rect(10, 0, 20, 10)], gts)
    assert exact == 1.0
    # the blob claims GT0 (union 200), GT1 is unmatched (+100) -> 100/300
    assert abs(merged - 1.0 / 3.0) < 1e-6
    assert merged < exact


def test_false_positives_are_charged_their_area():
    gts = [_rect(0, 0, 10, 10)]
    clean = _aji([_rect(0, 0, 10, 10)], gts)
    with_fp = _aji([_rect(0, 0, 10, 10), _rect(50, 50, 60, 60)], gts)
    assert clean == 1.0
    assert abs(with_fp - 100.0 / 200.0) < 1e-6


def test_no_prediction_scores_zero():
    assert _aji([], [_rect(0, 0, 10, 10)]) == 0.0
