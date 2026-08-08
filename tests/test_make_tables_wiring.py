"""Which variant, and which iteration, each table row reads.

The DOM-free restructure repoints almost every row: Ours becomes the
perspective-only ``full`` run rather than the dual-source ``ours`` run, the
baselines lose M1 and M3, table 4 moves from t=1 to t=2, and two tables are
replaced outright. Every one of those is a silent error if it goes wrong -- the
table still renders, the numbers still look plausible, and nothing says the
"Ours" row is quoting the pipeline that was just abandoned.

So these tests give each ``(variant, iteration)`` a distinct value and check
that it lands in the row that should be quoting it. They are about wiring, not
arithmetic: ``Results.macro`` is exercised against real data by
``scripts/check_tables.py``.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import make_tables  # noqa: E402
from scripts.make_tables import (  # noqa: E402
    MISSING,
    Results,
    table1,
    table3,
    table4,
    table5,
    table6,
    table7,
    table8,
)

_FIELDS = ("obj_F1", "obj_mIoU", "AJI", "AP95", "area_IoU", "area_Dice",
           "area_Prec", "area_Rec", "centroid_RMSE", "over_seg_rate",
           "under_seg_rate", "n_pred", "TP", "FP", "FN")


def _results(values):
    """``{(variant, iter): value}`` -> a Results where every field is ``value``."""
    acc = {}
    for (variant, t), v in values.items():
        for station in make_tables.ALL_STATIONS:
            row = {f: f"{v:.6f}" for f in _FIELDS}
            row["station"], row["iter"], row["variant"] = station, str(t), variant
            acc[(station, t, variant)] = row
    return Results(acc, {})


def _cells(md):
    """Rendered markdown -> ``{first column: {header: cell}}``."""
    lines = [ln for ln in md.splitlines() if ln.startswith("|")]
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    out = {}
    for line in lines[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        out[cells[0]] = dict(zip(header, cells))
    return out


RQ = "RQ (=F1) ↑"


def test_table1_ours_row_reads_the_perspective_only_run():
    """The whole point of the restructure: Ours is `full`, never `ours`."""
    r = _results({("dom", 2): 0.10, ("m2", 2): 0.20, ("m2_tuned", 2): 0.30,
                  ("full", 2): 0.40, ("ours", 2): 0.97})

    cells = _cells(table1(r, 2))

    assert cells["**Ours**"][RQ] == "0.4000"


def test_table1_drops_the_tdom_bound_baselines():
    r = _results({("dom", 2): 0.10, ("m2", 2): 0.20, ("full", 2): 0.40,
                  ("m1", 2): 0.91, ("m3", 2): 0.92, ("m3_tuned", 2): 0.93})

    out = table1(r, 2)

    assert "LateFusion" not in out
    assert "0.9100" not in out and "0.9200" not in out and "0.9300" not in out


def test_table1_dom_baseline_reads_the_prior_bearing_tdom_run():
    """`dom` (our pipeline on TDOM alone), not `m1` (no geometry prior)."""
    r = _results({("dom", 2): 0.10, ("m1", 2): 0.91, ("m2", 2): 0.20,
                  ("full", 2): 0.40})

    cells = _cells(table1(r, 2))
    labels = [k for k in cells if "TDOM" in k]

    assert len(labels) == 1
    assert cells[labels[0]][RQ] == "0.1000"


def test_table3_arms_read_their_own_variants():
    r = _results({("full", 0): 0.11, ("fb_selfimg", 2): 0.22,
                  ("fb_srcview", 2): 0.33, ("full", 2): 0.44})

    cells = _cells(table3(r, 2))
    values = [c[RQ] for c in cells.values()]

    assert values == ["0.1100", "0.2200", "0.3300", "0.4400"]


def test_table3_no_feedback_row_stays_at_t0_while_the_rest_report_t2():
    """t=0 is a configuration (no feedback), not a reporting level."""
    r = _results({("full", 0): 0.11, ("full", 1): 0.99, ("full", 2): 0.44,
                  ("fb_selfimg", 2): 0.22, ("fb_srcview", 2): 0.33})

    out = table3(r, 2)

    assert "0.9900" not in out


def test_table4_reports_the_last_iteration_not_t1():
    r = _results({("abl_noprior", 1): 0.91, ("abl_noprior", 2): 0.12,
                  ("full", 1): 0.92, ("full", 2): 0.44})

    cells = _cells(table4(r, 2))

    assert cells["No module-geometry prior"]["RQ (=F1) ↑"] == "0.1200"
    assert cells["**Full prior (all three)**"]["RQ (=F1) ↑"] == "0.4400"


def test_table5_compares_the_three_projection_methods():
    r = _results({("proj_collin", 2): 0.11, ("proj_affine", 2): 0.22,
                  ("full", 2): 0.44})

    cells = _cells(table5(r, 2))
    values = [c[RQ] for c in cells.values()]

    assert values == ["0.1100", "0.2200", "0.4400"]


def test_table5_reports_centroid_rmse():
    """Projection accuracy is what this table is for; RMSE is the discriminator."""
    r = _results({("proj_collin", 2): 0.11, ("proj_affine", 2): 0.22,
                  ("full", 2): 0.44})

    assert "Centroid RMSE (m) ↓" in table5(r, 2)


def test_table6_direction_labels_no_longer_claim_tdom():
    """The dirs/ runs never merged a TDOM; only the first row is a DOM run."""
    r = _results({("dom", 2): 0.10, ("d1_nadir", 2): 0.21, ("d2_o1", 2): 0.22,
                  ("d3_o2", 2): 0.23, ("d4_o3", 2): 0.24, ("d5_o4", 2): 0.25})

    cells = _cells(table6(r, 2))
    labels = list(cells)

    assert labels[0].startswith("TDOM only")
    assert not any("TDOM" in lb for lb in labels[1:])


def test_table7_convergence_tracks_the_perspective_only_run():
    r = _results({("full", 0): 0.11, ("full", 1): 0.22, ("full", 2): 0.33,
                  ("ours", 0): 0.91, ("ours", 1): 0.92, ("ours", 2): 0.93})

    out = table7(r, 2)
    cells = _cells(out)

    assert [c[RQ] for c in cells.values()] == ["0.1100", "0.2200", "0.3300"]


def _costs(values):
    """``{(variant, iter): wall_minutes}`` -> a cost table over all stations."""
    cost = {}
    for (variant, t), v in values.items():
        for station in make_tables.ALL_STATIONS:
            cost[(station, t, variant)] = {"wall_minutes": str(v),
                                           "sam3_calls": "1"}
    return cost


def test_runtime_is_missing_for_a_variant_that_has_not_run():
    """The shared t=0 cache cost must not masquerade as an unrun variant's runtime.

    cumulative_cost adds the ``shared`` row once, because every t=0 variant
    reads the same inference cache. When the variant itself contributed nothing
    that shared figure is all that remains -- and printing it puts a plausible
    runtime beside a row of ``--``, which reads as "it ran, it just measured
    nothing" instead of "it has not been run".
    """
    r = Results({}, _costs({("shared", 0): 68.8}))

    assert r.cumulative_cost("fb_selfimg", 2, "wall_minutes") == MISSING


def test_runtime_still_counts_the_shared_cache_for_a_variant_that_did_run():
    r = Results({}, _costs({("shared", 0): 10.0, ("full", 1): 5.0}))

    assert r.cumulative_cost("full", 1, "wall_minutes") == "15.0"


def test_runtime_is_missing_when_the_reported_iteration_was_never_reached():
    """Cost is the total that produced *this row*. No result, no total.

    `fb_tdom_only` is the live case: it ran to t=1 and stopped, so summing
    t=0..2 still found its t=1 cost and printed a plausible 108.8 min beside a
    row whose every metric was `--`. Reading that row, the configuration looks
    like it ran and simply scored nothing.
    """
    r = Results({}, _costs({("shared", 0): 10.0, ("fb_tdom_only", 1): 5.0}))

    assert r.cumulative_cost("fb_tdom_only", 2, "wall_minutes") == MISSING
    assert r.cumulative_cost("fb_tdom_only", 1, "wall_minutes") == "15.0"


def test_table8_orders_the_tdom_roles_from_sole_input_to_unused():
    r = _results({("dom", 2): 0.11, ("ours", 2): 0.33, ("full", 2): 0.44})

    cells = _cells(table8(r, 2))

    assert [c[RQ] for c in cells.values()] == ["0.1100", "0.3300", "0.4400"]


def test_table8_excludes_the_tdom_into_mv_feedback_row():
    """`fb_tdom_only` was the last TDOM-into-multi-view hybrid in the run plan.

    Isolating the two branches means nothing left to run mixes them; the three
    remaining rows carry the argument and are all already computed.
    """
    r = _results({("dom", 2): 0.11, ("fb_tdom_only", 2): 0.22,
                  ("ours", 2): 0.33, ("full", 2): 0.44})

    assert "0.2200" not in table8(r, 2)
