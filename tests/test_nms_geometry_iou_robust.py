"""Regression test: a GEOS overlay failure must not abort the pipeline.

``postprocess/nms.py:_geometry_iou`` holds the pipeline's only
``Geometry.Intersection`` call. GEOS throws ``TopologyException`` on some
self-intersecting rings, and the call was unguarded, so a single bad pair killed
the process: XinXie's ``abl_noprior`` run died after 14 h in
``fuse_multiview_features`` on one pair out of 59 869 fused features.

The features get there because 4.7% of that run's projected polygons are invalid
-- grazing ray/plane intersections push vertices to Z = -40 km and flip the ring
order. The module-geometry prior normally rejects them on area, so only the
no-prior ablation exercises this path, which is why the main suite never hit it.

The guard mirrors ``evaluation/geom.pair_overlap``: retry through ``Buffer(0)``,
then degrade to bbox IoU. It must never propagate.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from osgeo import ogr  # noqa: E402

ogr.DontUseExceptions()

import postprocess.nms as nms  # noqa: E402


def _rect(x0, y0, x1, y1):
    return {
        "segmentation": [[x0, y0, x1, y0, x1, y1, x0, y1, x0, y0]],
        "bbox": [x0, y0, x1, y1],
        "con_weight": 1.0,
    }


class _Exploding:
    """Stands in for a geometry GEOS cannot overlay."""

    def __init__(self, area, buffer_works):
        self._area = area
        self._buffer_works = buffer_works

    def Intersection(self, other):
        raise RuntimeError("TopologyException: found non-noded intersection")

    def Buffer(self, _d):
        if not self._buffer_works:
            raise RuntimeError("TopologyException: side location conflict")
        return _Repaired(self._area)

    def Area(self):
        return self._area

    def IsEmpty(self):
        return False


class _Repaired:
    def __init__(self, area):
        self._area = area

    def Intersection(self, other):
        return _Repaired(self._area / 2.0)

    def Area(self):
        return self._area

    def IsEmpty(self):
        return False


def test_buffer0_recovers_a_topology_exception(monkeypatch):
    f1, f2 = _rect(0, 0, 2, 1), _rect(0, 0, 2, 1)
    monkeypatch.setattr(nms, "_feature_to_geometry",
                        lambda f: _Exploding(2.0, buffer_works=True))

    iou = nms._geometry_iou(f1, f2)

    # Repaired pair: inter 1.0, union 2+2-1 = 3.
    assert abs(iou - 1.0 / 3.0) < 1e-9


def test_falls_back_to_bbox_iou_when_repair_also_fails(monkeypatch):
    # Identical boxes -> bbox IoU is exactly 1.0, so the fallback is visible.
    f1, f2 = _rect(0, 0, 2, 1), _rect(0, 0, 2, 1)
    monkeypatch.setattr(nms, "_feature_to_geometry",
                        lambda f: _Exploding(2.0, buffer_works=False))

    assert nms._geometry_iou(f1, f2) == 1.0


def test_nms_completes_despite_an_unusable_geometry(monkeypatch):
    """The end-to-end point: one bad pair must not stop NMS."""
    feats = [_rect(0, 0, 2, 1), _rect(0.1, 0, 2.1, 1), _rect(50, 50, 52, 51)]
    for i, f in enumerate(feats):
        f["con_weight"] = 1.0 - i * 0.1

    monkeypatch.setattr(nms, "_feature_to_geometry",
                        lambda f: _Exploding(2.0, buffer_works=False))

    kept = nms.nms_features(feats, score_field="con_weight", iou_threshold=0.5,
                            use_geometry_iou=True, backend="naive")
    assert len(kept) >= 1


def test_valid_geometry_is_untouched():
    """The guard must not change the answer for ordinary input."""
    a = _rect(0, 0, 2, 1)          # area 2
    b = _rect(1, 0, 3, 1)          # area 2, overlap 1
    assert abs(nms._geometry_iou(a, b) - 1.0 / 3.0) < 1e-9
