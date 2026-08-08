"""NMM merges overlapping fragments but is blind to disjoint ones.

The second property is the reason NMM is in the study: M1's half-modules sit
side by side with pairwise IoU 0.000, so no overlap-based operator can rejoin
them. These tests pin both halves of that behaviour.
"""

from postprocess.nmm import nmm_features


def _rect(x0, y0, x1, y1, score=1.0):
    return {"segmentation": [[x0, y0, x1, y0, x1, y1, x0, y1]],
            "bbox": [x0, y0, x1, y1], "con_weight": score}


def test_contained_fragment_is_absorbed():
    """A half inside a whole scores IoS 1.0 -- exactly what NMM is for."""
    out = nmm_features([_rect(0, 0, 10, 10, 0.9), _rect(0, 0, 5, 10, 0.5)])
    assert len(out) == 1
    assert abs(out[0]["geom"].Area() - 100.0) < 1e-6


def test_disjoint_halves_are_NOT_merged():
    """The M1 failure: two adjacent halves, zero overlap, NMM cannot see them."""
    out = nmm_features([_rect(0, 0, 5, 10, 0.9), _rect(5, 0, 10, 10, 0.8)])
    assert len(out) == 2, "overlap-based merging must leave disjoint fragments alone"


def test_partially_overlapping_fragments_merge_into_union():
    out = nmm_features([_rect(0, 0, 6, 10, 0.9), _rect(4, 0, 10, 10, 0.8)],
                       match_threshold=0.2)
    assert len(out) == 1
    assert abs(out[0]["geom"].Area() - 100.0) < 1e-6


def test_far_apart_detections_are_untouched():
    out = nmm_features([_rect(0, 0, 10, 10), _rect(100, 100, 110, 110)])
    assert len(out) == 2


def test_empty_input():
    assert nmm_features([]) == []
