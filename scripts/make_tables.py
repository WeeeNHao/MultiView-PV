#!/usr/bin/env python
"""Render the draft's tables 1-8 as Markdown from the two result CSVs.

Inputs (join key ``station, iter, variant``):
    <eval-root>/all_stations_summary.csv   accuracy   (eval_0518_batch.py)
    <eval-root>/run_stats.csv              cost       (collect_run_stats.py)

Variant -> table row mapping (see the DOM-free restructure design doc):

    full          Ours: perspective-only object-space   tables 1,2,3,4,5,6,7,8
                  instance fusion
    dom           TDOM fed through this pipeline,       tables 1,2 baseline;
                  prior and iteration intact            table 6 anchor; table 8
    m2            pixel-vote fusion baseline            tables 1,2
    fb_selfimg    image-space self re-prompt            table 3 arm 2
    fb_srcview    object-space, no cross-view           table 3 arm 3
                  broadcast
    proj_collin   collinearity projection               table 5
    proj_affine   affine projection                     table 5
    d1..d5        cumulative camera-direction sets      table 6
    abl_*         module-geometry prior ablation        table 4
    fb_tdom_only  TDOM-only re-prompting                table 8
    ours          the abandoned dual-source pipeline    table 8 only

`ours` and `m1`/`m3` are deliberately absent from the main tables: TDOM was
dropped from the method. At t=2 the perspective-only run beats the dual-source
one (F1 0.9999 vs 0.9969, AJI 0.9752 vs 0.9702) because merging the TDOM adds
32 false positives on BeiOu. `ours` survives in table 8 as evidence for that
decision; `m1`/`m3` stay in the per-station dump only.

Protocol: tables 1-6 and 8 report the last iteration (ITER_MAX); table 7 is the
convergence analysis and reports every level. t=1 is an execution step, never a
reporting level -- an unfinished configuration renders as `--` rather than
borrowing its t=1 number.

A cell whose run does not exist on disk renders as ``--``; the table is emitted
anyway, with a note listing what is missing, so partial progress is visible.

Usage:
    python scripts/make_tables.py > experiments_results.md
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from typing import Dict, List, Optional, Sequence, Tuple

ALL_STATIONS = ["001-BeiOu", "003-XinXie", "004-CangFang"]
# Which stations the macro averages cover; narrowed by --stations so a finished
# station can be tabulated on its own while the others are still running.
STATIONS = list(ALL_STATIONS)
MISSING = "--"


Key = Tuple[str, int, str]


def _station_key(name: str) -> str:
    """``001-BeiOu-exp2`` -> ``001-BeiOu``.

    Matched against ALL_STATIONS, not the (possibly narrowed) STATIONS, so that
    --stations filters what gets *reported* without changing how rows are keyed.
    """
    for s in ALL_STATIONS:
        if name.startswith(s):
            return s
    return name


def load_csv(path: str) -> Dict[Key, dict]:
    if not os.path.isfile(path):
        print(f"[tables] !! missing {path}", file=sys.stderr)
        return {}
    out: Dict[Key, dict] = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            out[(_station_key(row["station"]), int(row["iter"]), row["variant"])] = row
    return out


class Results:
    def __init__(self, eval_rows: Dict[Key, dict], stat_rows: Dict[Key, dict],
                 fallback_rows: Optional[Dict[Key, dict]] = None):
        # PQ/SQ/RQ are exact functions of columns the evaluator has always
        # written, so derive them when absent rather than forcing a re-eval:
        #   SQ = obj_mIoU,  RQ == obj_F1 (TP/(TP+.5FP+.5FN) is the F1 identity),
        #   PQ = SQ * RQ.
        for row in eval_rows.values():
            if not row.get("SQ"):
                row["SQ"] = row.get("obj_mIoU", "")
            if not row.get("RQ"):
                row["RQ"] = row.get("obj_F1", "")
            if not row.get("PQ"):
                try:
                    row["PQ"] = f"{float(row['SQ']) * float(row['RQ']):.6f}"
                except (TypeError, ValueError):
                    row["PQ"] = ""
        self.acc = eval_rows
        self.cost = stat_rows
        self.cost_fallback = fallback_rows or {}
        self.missing: List[str] = []

    def get(self, station: str, iter_idx: int, variant: str) -> Optional[dict]:
        row = self.acc.get((station, iter_idx, variant))
        if row is None:
            tag = f"{station} t={iter_idx} {variant}"
            if tag not in self.missing:
                self.missing.append(tag)
        return row

    def metric(self, station: str, iter_idx: int, variant: str, field: str,
               fmt: str = "{:.4f}") -> str:
        row = self.get(station, iter_idx, variant)
        if row is None or row.get(field) in (None, ""):
            return MISSING
        try:
            return fmt.format(float(row[field]))
        except (TypeError, ValueError):
            return str(row[field])

    def has(self, variant: str) -> bool:
        """Whether any station/iteration produced this variant."""
        return any(k[2] == variant for k in self.acc)

    def macro(self, iter_idx: int, variant: str, field: str,
              fmt: str = "{:.4f}") -> str:
        vals = self._station_values(iter_idx, variant, field)
        if not vals:
            return MISSING
        return fmt.format(sum(vals) / len(vals))

    def _station_values(self, iter_idx: int, variant: str,
                        field: str) -> List[float]:
        vals = []
        for st in STATIONS:
            # Route through get() rather than self.acc directly: it is what
            # records the absence. Reading self.acc here made every macro table
            # (1/3/4/5/6/8) render `--` silently, so a table that was entirely
            # unrun carried no note saying so.
            row = self.get(st, iter_idx, variant)
            if row is None or row.get(field) in (None, ""):
                continue
            try:
                vals.append(float(row[field]))
            except (TypeError, ValueError):
                pass
        return vals

    def cost_of(self, station: str, iter_idx: int, variant: str,
                field: str) -> Optional[float]:
        val = self._cost_from(self.cost, station, iter_idx, variant, field)
        if not val:  # absent, or zero because the work was paid in another run
            fallback = self._cost_from(self.cost_fallback, station, iter_idx,
                                       variant, field)
            if fallback:
                return fallback
        return val

    @staticmethod
    def _cost_from(table: Dict[Key, dict], station: str, iter_idx: int,
                   variant: str, field: str) -> Optional[float]:
        row = table.get((station, iter_idx, variant))
        if row is None:
            return None
        try:
            return float(row[field])
        except (TypeError, ValueError, KeyError):
            return None

    def cumulative_cost(self, variant: str, up_to_iter: int, field: str,
                        include_shared: bool = True) -> str:
        """Sum a cost field over t=0..up_to_iter, macro-averaged over stations.

        The t=0 inference pass lives on its own ``shared`` row because every t=0
        variant reads the same cache; it is added once here.
        """
        totals = []
        for st in STATIONS:
            shared = 0.0
            if include_shared:
                v = self.cost_of(st, 0, "shared", field)
                if v is not None:
                    shared = v

            total = 0.0
            for t in range(0, up_to_iter + 1):
                v = self.cost_of(st, t, variant, field)
                if v is not None:
                    total += v

            # Gate on the *reported* iteration, not on any iteration. This cost
            # is the total that produced this row, so if the run never reached
            # that level there is no total to report. Two ways it went wrong
            # before: a variant with no rows at all still printed the shared
            # cache cost, and one that stopped at t=1 still printed its t=1 cost
            # in a t=2 row. Both put a plausible runtime beside a row of `--`,
            # which reads as "it ran and measured nothing".
            if self.cost_of(st, up_to_iter, variant, field) is not None:
                totals.append(total + shared)
        if not totals:
            return MISSING
        avg = sum(totals) / len(totals)
        return f"{avg:.0f}" if field == "sam3_calls" else f"{avg:.1f}"


def md_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


# Accuracy columns shared by most tables: (csv field, header, format).
#
# AP75 used to sit here and was dropped: on these stations it reads 1.0000 for
# every method that works at all, because object F1 is flat from IoU 0.50 to
# 0.90. The strict thresholds are what actually separate configurations, and are
# where iterative refinement shows up at all -- refinement improves boundaries,
# not detection, so a loose threshold hides its entire contribution.
#
# Reported as three separate columns rather than their mean: the three move by
# very different amounts (AP90 is already near-saturated while AP975 is not), so
# averaging them buries exactly the resolution they were added to provide.
# `AP_high` is still in the CSV for anyone who wants the single-number summary.
INSTANCE = [
    # RQ rather than a separate "Obj. F1" column: RQ = TP/(TP+.5FP+.5FN) is the
    # F1 identity, so listing both would print the same number twice. Naming it
    # RQ keeps it consistent with PQ = SQ x RQ, which is also in the table.
    ("RQ", "RQ (=F1) ↑", "{:.4f}"),
    ("SQ", "SQ ↑", "{:.4f}"),
    ("PQ", "PQ ↑", "{:.4f}"),
    ("AJI", "AJI ↑", "{:.4f}"),
    ("AP95", "AP95 ↑", "{:.4f}"),
]

SEMANTIC = [
    ("area_IoU", "Area IoU ↑", "{:.4f}"),
    ("area_Dice", "Area Dice ↑", "{:.4f}"),
    ("area_Prec", "Area Prec. ↑", "{:.4f}"),
    ("area_Rec", "Area Rec. ↑", "{:.4f}"),
]

# Four instance-level and four semantic-level columns. AP90 and AP75 were
# dropped: both read ~1.0000 for every method that works at all, because object
# F1 is flat from IoU 0.50 to 0.90 here. PQ and SQ earn their place by splitting
# the two failure modes apart -- M1 has SQ 0.84 with RQ 0.03, i.e. correct masks
# shattered into fragments, which no single detection score expresses.
CORE = INSTANCE + SEMANTIC


def baseline_rows(r: Results, base: str, label: str) -> List[Tuple[str, str]]:
    """The pixel-vote baseline, as default and (when run) tuned rows.

    `min_votes=2` is an untuned default an order of magnitude below the useful
    threshold -- it scores obj_F1 = 0.0000 on all three stations, which reads as
    a strawman. Reporting the per-station tuned configuration beside it is what
    makes the comparison defensible. Before any tuned run exists the table is
    unchanged, so this is safe to leave in place.
    """
    tuned = f"{base}_tuned"
    if not r.has(tuned):
        return [(label, base)]
    return [(f"{label} (default)", base), (f"{label} (tuned)", tuned)]


def _main_methods(r: Results) -> List[Tuple[str, str]]:
    """The rows tables 1 and 2 share, defined once so they cannot drift apart.

    The TDOM baseline is `dom` -- this pipeline fed a single orthophoto, with the
    geometry prior and iteration intact -- not `m1`, which runs SAM 3 on the TDOM
    with no prior and gets *worse* as it iterates (0.111 -> 0.080 -> 0.056). `dom`
    is the honest ceiling of the orthophoto route and is what motivates needing
    multi-view at all; `m1` at 0.056 reads as a strawman.

    Ours is `full`, the perspective-only run. `ours` is the abandoned dual-source
    pipeline: at t=2 it scores *lower* (F1 0.9969 vs 0.9999, AJI 0.9702 vs
    0.9752) because merging the TDOM adds 32 false positives on BeiOu.
    """
    methods = [("SAM3-TDOM-Iter (w/ prior)", "dom")]
    methods += baseline_rows(r, "m2", "SAM3-MV-PixelVote-Iter")
    methods += [("**Ours**", "full")]
    return methods


def table1(r: Results, last: int) -> str:
    methods = _main_methods(r)
    header = ["Method"] + [h for _, h, _ in CORE] + [
        "Centroid RMSE (m) ↓", "#SAM3 calls ↓", "Runtime (min) ↓"]
    rows = []
    for label, variant in methods:
        row = [label]
        row += [r.macro(last, variant, f, fmt) for f, _, fmt in CORE]
        row.append(r.macro(last, variant, "centroid_RMSE", "{:.3f}"))
        row.append(r.cumulative_cost(variant, last, "sam3_calls"))
        row.append(r.cumulative_cost(variant, last, "wall_minutes"))
        rows.append(row)
    return md_table(header, rows)


def table2(r: Results, last: int) -> str:
    methods = _main_methods(r)
    # Derived from STATIONS so a narrowed --stations does not leave headers
    # pointing at columns that are no longer there.
    header = ["Method"] + [f"{s.split('-', 1)[-1]} Obj. F1 ↑" for s in STATIONS] + \
        ["Macro Avg. ↑", "Worst-site ↑", "Across-site Std. ↓"]
    rows = []
    for label, variant in methods:
        vals = r._station_values(last, variant, "obj_F1")
        row = [label] + [r.metric(st, last, variant, "obj_F1") for st in STATIONS]
        if vals:
            row.append(f"{sum(vals)/len(vals):.4f}")
            row.append(f"{min(vals):.4f}")
            row.append(f"{statistics.pstdev(vals):.4f}" if len(vals) > 1 else MISSING)
        else:
            row += [MISSING] * 3
        rows.append(row)
    return md_table(header, rows)


def table3(r: Results, last: int) -> str:
    """Where the feedback is routed. One mechanism is added per row.

        1 -> 2  iterative re-prompting itself
        2 -> 3  object-space fusion and geometry-prior filtering
        3 -> 4  broadcast to every view whose footprint covers the instance

    The geometry prior is only definable in object space, so arm 2 necessarily
    lacks it; that is a property of the design, not a confounded comparison.

    t=0 is a *configuration* here -- no feedback at all -- not a reporting
    level. Every other row reports `last`.
    """
    configs = [
        ("No feedback (t=0)", 0, "full"),
        ("Image-space self re-prompt", last, "fb_selfimg"),
        ("Object-space, source view only", last, "fb_srcview"),
        ("**Object-space, all covering views (Ours)**", last, "full"),
    ]
    header = ["Configuration"] + [h for _, h, _ in CORE] + ["Runtime (min) ↓"]
    rows = []
    for label, t, variant in configs:
        row = [label] + [r.macro(t, variant, f, fmt) for f, _, fmt in CORE]
        row.append(r.cumulative_cost(variant, t, "wall_minutes"))
        rows.append(row)
    return md_table(header, rows)


def table4(r: Results, last: int) -> str:
    # Cumulative, not leave-one-out. Leave-one-out is uninformative here because
    # the sub-scores are redundant for the dominant failure mode: a half-module
    # fragment has 50% of the nominal area AND aspect ratio 1.0 instead of 2.0,
    # so w_area and w_ratio each catch it alone and dropping either one barely
    # moves the score. Adding them one at a time shows where the gain actually
    # comes from.
    #
    # "area + rectangularity" needs no run of its own: with three sub-scores,
    # keeping area and rectangularity *is* w_ratio=0, i.e. abl_no_ratio.
    #
    # The full-prior row is `full`, not `ours`. The ablations run with
    # dom_merge disabled (see configs/*/oblique_views.yaml), so they are
    # perspective-only; `ours` is dual-source, and using it here would confound
    # the geometry prior with the TDOM branch.
    # (a) cumulative, both orderings, and (b) leave-one-out. Reported together
    # because either alone is misleading: RQ saturates the moment area joins, so
    # a rectangularity-first curve makes rectangularity look inert, while
    # leave-one-out understates area for the same reason. Read the strict
    # columns (AP95/AJI/over-seg), not RQ, for what each term actually buys.
    configs = [
        ("No module-geometry prior", "abl_noprior"),
        ("— cumulative, rectangularity first —", None),
        ("+ rectangularity", "abl_only_shape"),
        ("+ area + rectangularity", "abl_no_ratio"),
        ("— cumulative, area first —", None),
        ("+ area", "abl_only_area"),
        ("+ area + rectangularity ", "abl_no_ratio"),
        ("— leave-one-out from the full prior —", None),
        ("w/o area", "abl_no_area"),
        ("w/o rectangularity", "abl_no_shape"),
        ("w/o aspect ratio", "abl_no_ratio"),
        ("**Full prior (all three)**", "full"),
    ]
    header = ["Configuration", "RQ (=F1) ↑", "PQ ↑", "AJI ↑", "AP95 ↑",
              "Obj. mIoU ↑", "Over-seg. ↓", "Under-seg. ↓"]
    rows = []
    for label, variant in configs:
        if variant is None:          # section divider
            rows.append([f"*{label}*"] + [""] * (len(header) - 1))
            continue
        rows.append([
            label,
            r.macro(last, variant, "RQ"),
            r.macro(last, variant, "PQ"),
            r.macro(last, variant, "AJI"),
            r.macro(last, variant, "AP95"),
            r.macro(last, variant, "obj_mIoU"),
            r.macro(last, variant, "over_seg_rate"),
            r.macro(last, variant, "under_seg_rate"),
        ])
    return md_table(header, rows)


def table5(r: Results, last: int) -> str:
    """Projection method. Centroid RMSE is the column that separates them.

    All three are already implemented and selected by one config key,
    `projection.oblique.method`. `auto` is excluded: it tries affine and falls
    back to collinearity, so as an ablation row it is a mixture of the other two.

    Caveat for the affine row: forcing the method makes every feature with fewer
    than three control points undeterminable, so it is tagged `affine_failed`
    and dropped downstream (see tests/test_projection.py). Part of that row's
    recall loss is therefore features never reaching the output rather than
    being mislocated -- report the dropped fraction beside this table, or the
    row looks inexplicably bad.
    """
    configs = [
        ("Collinearity (direct)", "proj_collin"),
        ("Affine (control-point)", "proj_affine"),
        ("**Slope correction (ours)**", "full"),
    ]
    header = ["Projection method"] + [h for _, h, _ in CORE] + [
        "Centroid RMSE (m) ↓", "Runtime (min) ↓"]
    rows = []
    for label, variant in configs:
        row = [label] + [r.macro(last, variant, f, fmt) for f, _, fmt in CORE]
        row.append(r.macro(last, variant, "centroid_RMSE", "{:.3f}"))
        row.append(r.cumulative_cost(variant, last, "wall_minutes"))
        rows.append(row)
    return md_table(header, rows)


def table6(r: Results, last: int) -> str:
    """Camera directions, added cumulatively.

    The labels used to read "TDOM + Nadir + ...", which was wrong: the dirs/
    runs never merged a TDOM. `d5_o4` and `full` are identical at t=0 on every
    station (BeiOu: 1772 predictions, Area IoU 0.9711, F1 1.0000), so these are
    perspective view sets only.

    The first row is kept as the zero-perspective-view anchor, but it is the DOM
    branch -- a different pipeline, not a point on this curve.
    """
    sets = [
        ("TDOM only (no perspective views)", "dom", 0),
        ("Nadir only", "d1_nadir", 1),
        ("Nadir + O1", "d2_o1", 2),
        ("Nadir + O1 + O2", "d3_o2", 3),
        ("Nadir + O1 + O2 + O3", "d4_o3", 4),
        ("Nadir + O1 + O2 + O3 + O4", "d5_o4", 5),
    ]
    header = ["Raw-view set", "# Oblique directions"] + [h for _, h, _ in CORE]
    rows = []
    for label, variant, n_dir in sets:
        n_obl = max(n_dir - 1, 0)
        row = [label, str(n_obl)]
        row += [r.macro(last, variant, f, fmt) for f, _, fmt in CORE]
        rows.append(row)
    return md_table(header, rows)


def table7(r: Results, last: int) -> str:
    """Iteration convergence -- the one table that still reports every level."""
    header = ["Iteration t", "#Pred.", "TP", "FP", "FN"] + \
        [h for _, h, _ in CORE] + ["Cumulative #SAM3 calls", "Cumulative runtime (min)"]
    rows = []
    for t in range(0, last + 1):
        row = [str(t)]
        for field in ("n_pred", "TP", "FP", "FN"):
            row.append(r.macro(t, "full", field, "{:.0f}"))
        row += [r.macro(t, "full", f, fmt) for f, _, fmt in CORE]
        row.append(r.cumulative_cost("full", t, "sam3_calls"))
        row.append(r.cumulative_cost("full", t, "wall_minutes"))
        rows.append(row)
    return md_table(header, rows)


def table8(r: Results, last: int) -> str:
    """Why the orthophoto route is a dead end, ordered by where TDOM enters.

    Reuses runs that already exist, including the abandoned dual-source `ours`.
    Keeping them answers the obvious reviewer question -- you had an orthophoto,
    why not use it -- with measurements rather than assertion.
    """
    configs = [
        ("TDOM as the sole input", "dom"),
        ("TDOM as a feedback source only", "fb_tdom_only"),
        ("TDOM as input + feedback (dual-source)", "ours"),
        ("**TDOM unused (Ours)**", "full"),
    ]
    header = ["Role of TDOM"] + [h for _, h, _ in CORE] + ["Runtime (min) ↓"]
    rows = []
    for label, variant in configs:
        row = [label] + [r.macro(last, variant, f, fmt) for f, _, fmt in CORE]
        row.append(r.cumulative_cost(variant, last, "wall_minutes"))
        rows.append(row)
    return md_table(header, rows)


def per_station_dump(r: Results) -> str:
    out = []
    fields = [("n_pred", "#Pred"), ("area_IoU", "Area IoU"),
              ("obj_Prec", "Prec"), ("obj_Rec", "Rec"), ("obj_F1", "F1"),
              ("obj_mIoU", "mIoU"), ("AP95", "AP95"), ("PQ", "PQ"), ("AJI", "AJI"),
              ("centroid_RMSE", "cRMSE"), ("over_seg", "Over"),
              ("under_seg", "Under")]
    for st in STATIONS:
        keys = sorted((k for k in r.acc if k[0] == st), key=lambda k: (k[2], k[1]))
        if not keys:
            continue
        out.append(f"\n### {st}\n")
        rows = []
        for k in keys:
            row = r.acc[k]
            cells = [k[2], str(k[1])]
            for f, _ in fields:
                v = row.get(f, "")
                try:
                    cells.append(f"{float(v):.4f}" if "." in str(v) else str(v))
                except (TypeError, ValueError):
                    cells.append(str(v))
            rows.append(cells)
        out.append(md_table(["Variant", "t"] + [h for _, h in fields], rows))
    return "\n".join(out)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-root", default="/data/dataset/PV/ZS_PV/eval_exp2")
    p.add_argument("--extra-stats", default=None,
                   help="Second run_stats.csv consulted only where the primary "
                        "one reports no cost. Needed because -exp2 symlinks its "
                        "t=0 inference cache back to -exp: the SAM3 calls for "
                        "that pass were paid in the earlier run and are recorded "
                        "there, so without this the t=0 row reads as free.")
    # No per-table iteration knob. Every table except the convergence one
    # reports this level; t=1 is an execution step, never a reporting level, and
    # must not stand in for a t=2 run that has not finished.
    p.add_argument("--iter-max", type=int, default=2)
    p.add_argument("--stations", default=None,
                   help="Comma-separated subset to report (default: all three). "
                        "Use to tabulate a finished station while the others "
                        "are still running; macro columns then cover only these.")
    args = p.parse_args(argv)

    if args.stations:
        global STATIONS
        wanted = [s.strip() for s in args.stations.split(",") if s.strip()]
        unknown = [s for s in wanted if s not in ALL_STATIONS]
        if unknown:
            print(f"[tables] unknown station(s): {unknown}; expected any of "
                  f"{ALL_STATIONS}", file=sys.stderr)
            return 1
        STATIONS = wanted

    r = Results(
        load_csv(os.path.join(args.eval_root, "all_stations_summary.csv")),
        load_csv(os.path.join(args.eval_root, "run_stats.csv")),
        load_csv(args.extra_stats) if args.extra_stats else None,
    )
    if not r.acc:
        print("[tables] no accuracy rows loaded", file=sys.stderr)
        return 1

    last = args.iter_max
    # "三站宏平均" is a lie whenever --stations narrows the report, and these
    # headings get pasted straight into the paper.
    scope = ("三站宏平均" if len(STATIONS) == len(ALL_STATIONS)
             else (f"{STATIONS[0].split('-', 1)[1]} 单站" if len(STATIONS) == 1
                   else f"{len(STATIONS)} 站宏平均"))
    print(f"""# 实验结果（自动生成）

> 来源：`{args.eval_root}/all_stations_summary.csv` + `run_stats.csv`
> 生成命令：`python scripts/make_tables.py`
> 实例级指标：`RQ`(=F1，检出) / `SQ`(匹配掩膜贴合度) / `PQ`(=SQ×RQ) /
> `AJI`(面积加权，惩罚过分割与欠分割) / `AP95`(IoU 0.95 下的 AP)，匹配阈值 0.5。
> 语义级指标：整体掩膜的 `IoU` / `Dice` / `Precision` / `Recall`。
> 不报 AP50/AP75/AP90：本数据上 F1 在 IoU 0.50–0.90 之间完全饱和，三者同值。
> 表 1–6、表 8 报 t={last}；表 7 是迭代收敛分析，报 t=0..{last}。
> t=1 是执行的必经步骤，但不作为任何表格的渲染层：未跑完的配置一律显示 `{MISSING}`，
> 不用 t=1 的数字占位。
> `{MISSING}` = 该配置尚未跑出结果。

## 表 1：主实验总体结果（{scope}）

{table1(r, last)}

> M2 与 Ours 是"相同输入源、不同融合范式"的端到端对照：融合单位、几何先验、
> 逐模块局部投影三者一起变，**不是单变量消融**。单变量证据见表 3/4/5。

## 表 2：主实验逐电站结果

{table2(r, last)}

## 表 3：迭代反馈路由消融（t={last}）

{table3(r, last)}

## 表 4：模块几何先验累积消融（t={last}）

{table4(r, last)}

## 表 5：投影方式消融（t={last}）

{table5(r, last)}

## 表 6：镜头方向增量分析（t={last}）

{table6(r, last)}

## 表 7：迭代收敛分析（Ours，{scope}）

{table7(r, last)}

## 表 8：分析——正射路径为何是死路（t={last}）

{table8(r, last)}

## 附：逐站逐变体明细
{per_station_dump(r)}
""")

    if r.missing:
        print("\n> 缺失的运行（渲染为 `--`）：\n>")
        for m in r.missing:
            print(f"> - {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
