"""Regression test for ``eval_0518_batch.discover_predictions``.

The experiment tree carries two opposite nestings. The main line and the
table-7 direction sets put the iteration on the outside
(``iter_0/full/final.shp``), while every baseline and the headline method put it
on the inside (``ours/iter_0/final.shp``, ``m1/iter_0/final.shp``, ...). Only the
first was recognised, so a completed run of SD / SP / SM1-SM3 evaluated to zero
rows and tables 1/2/3/5/6/8 could not be filled at all -- with no error, because
the discovery simply returned fewer tuples.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.eval_0518_batch import discover_predictions  # noqa: E402


def _touch_shp(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("")


def test_discovers_both_nestings(tmp_path):
    station = str(tmp_path / "004-CangFang-exp2")

    _touch_shp(os.path.join(station, "iter_0", "full", "final.shp"))
    _touch_shp(os.path.join(station, "iter_1", "full", "final.shp"))
    _touch_shp(os.path.join(station, "iter_0", "dirs", "d1_nadir", "final.shp"))
    for method in ("dom", "ours", "m1", "m2", "m3"):
        for t in (0, 1, 2):
            _touch_shp(os.path.join(station, method, f"iter_{t}", "final.shp"))

    # Bookkeeping dirs and a half-finished direction set must not show up.
    os.makedirs(os.path.join(station, "_logs"), exist_ok=True)
    os.makedirs(os.path.join(station, "iter_0", "dirs", "d0_tdom"), exist_ok=True)

    found = discover_predictions(station)
    got = {(i, v) for i, v, _ in found}

    assert (0, "full") in got and (1, "full") in got
    assert (0, "d1_nadir") in got
    assert (0, "d0_tdom") not in got
    for method in ("dom", "ours", "m1", "m2", "m3"):
        for t in (0, 1, 2):
            assert (t, method) in got, f"missed {method}/iter_{t}"

    assert len(found) == len(got), "duplicate (iter, variant) rows"
    assert all(os.path.isfile(p) for _, _, p in found)


def test_ignores_method_dirs_without_iterations(tmp_path):
    station = str(tmp_path / "001-BeiOu-exp2")
    _touch_shp(os.path.join(station, "iter_0", "full", "final.shp"))
    # The shared inference/projection cache is a directory of per-image shp --
    # it has no iter_<i>/final.shp and must not be mistaken for a method.
    _touch_shp(os.path.join(station, "iter_0", "shared", "proj", "DJI_0001.shp"))
    os.makedirs(os.path.join(station, "_preflight"), exist_ok=True)

    assert discover_predictions(station) == [
        (0, "full", os.path.join(station, "iter_0", "full", "final.shp"))
    ]
