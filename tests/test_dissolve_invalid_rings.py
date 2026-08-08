"""Regression test: dissolving a set containing a self-intersecting ring.

``Buffer(0)`` is the standard repair for a bow-tie ring, but it does not return
a polygon -- it splits the ring and returns a MultiPolygon (or, for messier
input, a GeometryCollection). Adding that to a ``wkbMultiPolygon`` collection
raises ``OGR Error: Unsupported geometry type``.

The failure was not local: ``area_overlap_metrics`` is computed per variant, so
one bad polygon aborted the *whole* variant. In the 2026-07-26 evaluation of the
``-exp2`` tree that lost 9 of 69 result sets, among them CangFang ``m1`` at
t=1/t=2 -- rows that tables 1 and 2 need.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from osgeo import ogr  # noqa: E402

ogr.DontUseExceptions()

from evaluation.geom import area_overlap_metrics, dissolve_features  # noqa: E402


def _feat(ring, bbox):
    return {"segmentation": [ring], "bbox": bbox, "con_weight": 1.0}


# A bow-tie: the ring crosses itself at (1, 1), enclosing two unit-area
# triangles of opposite orientation.
BOWTIE = _feat([0, 0, 2, 2, 2, 0, 0, 2, 0, 0], [0, 0, 2, 2])
SQUARE = _feat([10, 10, 12, 10, 12, 12, 10, 12, 10, 10], [10, 10, 12, 12])


def test_dissolve_survives_a_self_intersecting_ring():
    g = dissolve_features([BOWTIE])
    assert g is not None and not g.IsEmpty()
    # Buffer(0) keeps both triangles: 2 x (1/2 base x height) = 1.0
    assert abs(g.Area() - 1.0) < 1e-9


def test_one_bad_ring_does_not_lose_the_valid_features():
    g = dissolve_features([BOWTIE, SQUARE])
    assert g is not None
    assert abs(g.Area() - (1.0 + 4.0)) < 1e-9


def test_area_metrics_computable_with_invalid_geometry_present():
    pred = dissolve_features([BOWTIE, SQUARE])
    gt = dissolve_features([SQUARE])
    m = area_overlap_metrics(pred, gt)

    assert abs(m["area_recall"] - 1.0) < 1e-9      # the square is fully covered
    assert abs(m["area_precision"] - 4.0 / 5.0) < 1e-9  # bow-tie area is spurious
