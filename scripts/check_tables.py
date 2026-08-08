#!/usr/bin/env python
"""Verify every number in the generated tables straight from the source CSVs.

Re-derives each cell of ``experiments_results.md`` from
``all_stations_summary.csv`` / ``run_stats.csv`` and compares it to what was
rendered, character for character.

This shares no code with ``make_tables.py`` on purpose. Importing its
aggregation would make the check circular -- it would agree with whatever
make_tables does, including the thing the check exists to catch. The row-to-
variant map below is therefore an independent restatement of the table spec,
not a copy: if make_tables points a row at a different variant, or reports a
different iteration, the two declarations disagree and this fails.

What it cannot catch: an error present in both declarations. The row map is
short and the design doc is the third copy, so that is a deliberate trade.

Usage:
    python scripts/check_tables.py [experiments_results.md]
    python scripts/check_tables.py --eval-root ... --iter-max 2
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from typing import Dict, List, Optional, Sequence, Tuple

STATIONS = ["001-BeiOu", "003-XinXie", "004-CangFang"]
MISSING = "--"

# header -> (csv column, format). Restated here rather than imported.
COLUMNS: Dict[str, Tuple[str, str]] = {
    "RQ (=F1) ↑": ("RQ", "{:.4f}"),
    "SQ ↑": ("SQ", "{:.4f}"),
    "PQ ↑": ("PQ", "{:.4f}"),
    "AJI ↑": ("AJI", "{:.4f}"),
    "AP95 ↑": ("AP95", "{:.4f}"),
    "Area IoU ↑": ("area_IoU", "{:.4f}"),
    "Area Dice ↑": ("area_Dice", "{:.4f}"),
    "Area Prec. ↑": ("area_Prec", "{:.4f}"),
    "Area Rec. ↑": ("area_Rec", "{:.4f}"),
    "Centroid RMSE (m) ↓": ("centroid_RMSE", "{:.3f}"),
    "Obj. mIoU ↑": ("obj_mIoU", "{:.4f}"),
    "Over-seg. ↓": ("over_seg_rate", "{:.4f}"),
    "Under-seg. ↓": ("under_seg_rate", "{:.4f}"),
    "#Pred.": ("n_pred", "{:.0f}"),
    "TP": ("TP", "{:.0f}"),
    "FP": ("FP", "{:.0f}"),
    "FN": ("FN", "{:.0f}"),
}

# header -> run_stats column. These are cumulative over t=0..row iteration.
COST_COLUMNS: Dict[str, str] = {
    "Runtime (min) ↓": "wall_minutes",
    "#SAM3 calls ↓": "sam3_calls",
    "Cumulative runtime (min)": "wall_minutes",
    "Cumulative #SAM3 calls": "sam3_calls",
}


def table_spec(last: int) -> List[Tuple[str, List[Optional[Tuple[str, int]]]]]:
    """(heading prefix, one (variant, iter) per body row; None marks a divider)."""
    return [
        ("表 1", [("dom", last), ("m2", last), ("m2_tuned", last), ("full", last)]),
        ("表 3", [("full", 0), ("fb_selfimg", last),
                  ("fb_srcview", last), ("full", last)]),
        ("表 4", [("abl_noprior", last), None,
                  ("abl_only_shape", last), ("abl_no_ratio", last), None,
                  ("abl_only_area", last), ("abl_no_ratio", last), None,
                  ("abl_no_area", last), ("abl_no_shape", last),
                  ("abl_no_ratio", last), ("full", last)]),
        ("表 5", [("proj_collin", last), ("proj_affine", last), ("full", last)]),
        ("表 6", [("dom", last), ("d1_nadir", last), ("d2_o1", last),
                  ("d3_o2", last), ("d4_o3", last), ("d5_o4", last)]),
        ("表 7", [("full", t) for t in range(last + 1)]),
        ("表 8", [("dom", last), ("fb_tdom_only", last),
                  ("ours", last), ("full", last)]),
    ]


def load(path: str) -> Dict[Tuple[str, int, str], dict]:
    out: Dict[Tuple[str, int, str], dict] = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            station = next((s for s in STATIONS if row["station"].startswith(s)),
                           row["station"])
            out[(station, int(row["iter"]), row["variant"])] = row
    return out


def macro(acc, variant: str, iter_idx: int, field: str, fmt: str) -> str:
    vals = []
    for st in STATIONS:
        row = acc.get((st, iter_idx, variant))
        if row is None or not row.get(field):
            continue
        vals.append(float(row[field]))
    if not vals:
        return MISSING
    return fmt.format(sum(vals) / len(vals))


def cumulative(cost, variant: str, up_to: int, field: str) -> str:
    """Variant rows over t=0..up_to, plus the shared t=0 cache once.

    Reported only when the variant has a row at ``up_to`` itself: this is the
    cost that produced *this row*, so a run that never reached that iteration
    has no total to report. The shared cache cost alone never qualifies -- it is
    common to every variant and says nothing about whether this one ran.
    """
    totals = []
    for st in STATIONS:
        shared = 0.0
        row = cost.get((st, 0, "shared"))
        if row and row.get(field):
            shared = float(row[field])

        total = 0.0
        for t in range(up_to + 1):
            row = cost.get((st, t, variant))
            if row and row.get(field):
                total += float(row[field])
        reached = cost.get((st, up_to, variant))
        if reached and reached.get(field):
            totals.append(total + shared)
    if not totals:
        return MISSING
    avg = sum(totals) / len(totals)
    return f"{avg:.0f}" if field == "sam3_calls" else f"{avg:.1f}"


def parse_tables(md: str) -> Dict[str, Tuple[List[str], List[List[str]]]]:
    """heading -> (header cells, body rows)."""
    out: Dict[str, Tuple[List[str], List[List[str]]]] = {}
    heading = None
    header: List[str] = []
    rows: List[List[str]] = []
    for line in md.splitlines():
        if line.startswith("## "):
            if heading and header:
                out[heading] = (header, rows)
            heading, header, rows = line[3:].strip(), [], []
        elif line.startswith("|") and heading:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not header:
                header = cells
            elif set("".join(cells)) <= set("-: "):
                continue
            else:
                rows.append(cells)
    if heading and header:
        out[heading] = (header, rows)
    return out


def check(md_path: str, eval_root: str, last: int) -> int:
    acc = load(os.path.join(eval_root, "all_stations_summary.csv"))
    cost = load(os.path.join(eval_root, "run_stats.csv"))
    with open(md_path, encoding="utf-8") as f:
        tables = parse_tables(f.read())

    problems: List[str] = []
    checked = 0

    for prefix, spec in table_spec(last):
        heading = next((h for h in tables if h.startswith(prefix)), None)
        if heading is None:
            problems.append(f"{prefix}: not present in {md_path}")
            continue
        header, rows = tables[heading]
        if len(rows) != len(spec):
            problems.append(
                f"{prefix}: {len(rows)} body rows, spec declares {len(spec)}")
            continue

        for cells, want in zip(rows, spec):
            if want is None:            # divider
                continue
            variant, iter_idx = want
            label = cells[0]
            for col, got in zip(header[1:], cells[1:]):
                if col in COLUMNS:
                    field, fmt = COLUMNS[col]
                    expect = macro(acc, variant, iter_idx, field, fmt)
                elif col in COST_COLUMNS:
                    expect = cumulative(cost, variant, iter_idx,
                                        COST_COLUMNS[col])
                else:
                    continue
                checked += 1
                if got != expect:
                    problems.append(
                        f"{prefix} | {label} | {col}: rendered {got!r}, "
                        f"recomputed {expect!r} from {variant} t={iter_idx}")

    # Table 2 is per-station rather than macro, so it gets its own pass.
    heading = next((h for h in tables if h.startswith("表 2")), None)
    if heading is None:
        problems.append("表 2: not present")
    else:
        header, rows = tables[heading]
        variants = ["dom", "m2", "m2_tuned", "full"]
        if len(rows) != len(variants):
            problems.append(f"表 2: {len(rows)} rows, spec declares {len(variants)}")
        else:
            for cells, variant in zip(rows, variants):
                vals = []
                for st in STATIONS:
                    row = acc.get((st, last, variant))
                    vals.append(float(row["obj_F1"]) if row and row.get("obj_F1")
                                else None)
                present = [v for v in vals if v is not None]
                expect = [f"{v:.4f}" if v is not None else MISSING for v in vals]
                if present:
                    expect.append(f"{sum(present)/len(present):.4f}")
                    expect.append(f"{min(present):.4f}")
                    expect.append(f"{statistics.pstdev(present):.4f}"
                                  if len(present) > 1 else MISSING)
                else:
                    expect += [MISSING] * 3
                checked += len(expect)
                if cells[1:] != expect:
                    problems.append(
                        f"表 2 | {cells[0]}: rendered {cells[1:]}, "
                        f"recomputed {expect} from {variant} t={last}")

    if problems:
        print(f"[check] {len(problems)} problem(s) across {checked} cells:",
              file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print(f"[check] {checked} cells verified against {eval_root}; all match.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("markdown", nargs="?", default="experiments_results.md")
    p.add_argument("--eval-root", default="/data/dataset/PV/ZS_PV/eval_exp2")
    p.add_argument("--iter-max", type=int, default=2)
    p.add_argument("--stations", default=None,
                   help="Comma-separated subset, matching the --stations the "
                        "markdown was generated with. The macro columns cover "
                        "only these, so checking a narrowed file against all "
                        "three would disagree on every cell.")
    args = p.parse_args(argv)

    if args.stations:
        global STATIONS
        STATIONS = [s.strip() for s in args.stations.split(",") if s.strip()]

    return check(args.markdown, args.eval_root, args.iter_max)


if __name__ == "__main__":
    raise SystemExit(main())
