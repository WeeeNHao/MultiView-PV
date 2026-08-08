#!/usr/bin/env python
"""Aggregate per-run cost from the pipeline's own logs into one CSV.

Tables 1, 3, 5, 7 and 8 of the draft report ``#SAM3 calls`` and ``Runtime``
alongside the accuracy metrics, and neither comes out of the evaluator -- they
live in ``<result>/logs/pipeline_summary_*.json``, one file per rank, written by
``PipelineRunLogger``.

Two things make this more than a sum:

  * ``infer_image.count`` is per rank. The number of SAM3 image calls for a
    round is the sum across ranks; the wall clock is the *slowest* rank, not the
    sum, because ranks run concurrently.
  * A round can be more than one pipeline invocation. ``stage_SP`` (Ours) runs
    the TDOM branch and then the perspective branch, which log into
    ``ours/iter_<t>/dom/logs`` and ``ours/iter_<t>/logs`` respectively. Those
    are sequential, so their wall clocks add. Invocations sharing one log
    directory are separated by clustering the summaries into groups of
    ``world_size`` ordered by finish time.

Variant naming matches ``eval_0518_batch.discover_predictions`` so the two CSVs
join on ``(station, iter, variant)``.

Usage:
    python scripts/collect_run_stats.py --pattern '*-exp2' \
        --out /data/dataset/PV/ZS_PV/eval_exp2/run_stats.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

_ITER_RE = re.compile(r"^iter_(\d+)$")

FIELDS = [
    "station",
    "iter",
    "variant",
    "sam3_calls",
    "wall_minutes",
    "invocations",
    "world_size",
    "infer_ms_total",
    "project_ms_total",
    "postprocess_ms_total",
    "logs_dir",
]

# Stage keys that make up the postprocess tail, for the runtime breakdown.
_POSTPROCESS_STAGES = (
    "collect_per_image_outputs",
    "per_image_nms",
    "multiview_fusion",
    "dom_merge",
    "write_final_output",
    "export_bbox_prompts",
)


def _classify(station_dir: str, logs_dir: str) -> Optional[Tuple[int, str]]:
    """Map a ``logs/`` directory to ``(iter, variant)``, or None if it is not a
    result directory we report on."""
    rel = os.path.relpath(os.path.dirname(logs_dir), station_dir)
    parts = rel.split(os.sep)

    # iter_<t>/full, iter_<t>/dirs/<name>, iter_<t>/shared
    m = _ITER_RE.match(parts[0]) if parts else None
    if m:
        if len(parts) == 2 and parts[1] == "full":
            return int(m.group(1)), "full"
        if len(parts) == 3 and parts[1] == "dirs":
            return int(m.group(1)), parts[2]
        # The t=0 inference pass every t=0 variant reads from. Reported on its
        # own row rather than folded into full/dirs/ours, which would count the
        # same pass three times; a cumulative total adds it exactly once.
        if len(parts) == 2 and parts[1] == "shared":
            return int(m.group(1)), "shared"
        return None

    # <method>/iter_<t>  and Ours' inner TDOM branch <method>/iter_<t>/dom
    if len(parts) >= 2:
        m = _ITER_RE.match(parts[1])
        if m and len(parts) == 2:
            return int(m.group(1)), parts[0]
        if m and len(parts) == 3 and parts[2] == "dom":
            # Same round as the perspective branch; costs are added to it.
            return int(m.group(1)), parts[0]
    return None


def _load_summaries(logs_dir: str) -> List[dict]:
    out = []
    for p in sorted(glob.glob(os.path.join(logs_dir, "pipeline_summary_*.json"))):
        try:
            with open(p, encoding="utf-8") as f:
                out.append(json.load(f))
        except Exception as exc:  # noqa: BLE001
            print(f"[stats] !! unreadable {p}: {exc}", file=sys.stderr)
    return out


def _stage_total(summary: dict, key: str) -> float:
    return float(summary.get("stages", {}).get(key, {}).get("total_ms", 0.0))


def _stage_count(summary: dict, key: str) -> int:
    return int(summary.get("stages", {}).get(key, {}).get("count", 0))


def _aggregate(summaries: List[dict]) -> dict:
    """Fold one directory's rank summaries into a single cost record."""
    if not summaries:
        return {}

    world = max(int(s.get("world_size", 1) or 1) for s in summaries)
    # Sort by finish time so sequential invocations land in separate chunks.
    ordered = sorted(summaries, key=lambda s: str(s.get("ts", "")))
    chunks = [ordered[i:i + world] for i in range(0, len(ordered), world)]

    wall = 0.0
    for chunk in chunks:
        # Ranks of one invocation run concurrently -> the slowest one is the
        # elapsed time; separate invocations are sequential -> they add.
        wall += max(float(c.get("total_elapsed_mins", 0.0) or 0.0) for c in chunk)

    return {
        "sam3_calls": sum(_stage_count(s, "infer_image") for s in summaries),
        "wall_minutes": wall,
        "invocations": len(chunks),
        "world_size": world,
        "infer_ms_total": sum(_stage_total(s, "infer_image") for s in summaries),
        "project_ms_total": sum(_stage_total(s, "project_raw_shp") for s in summaries),
        "postprocess_ms_total": sum(
            _stage_total(s, k) for s in summaries for k in _POSTPROCESS_STAGES
        ),
    }


def collect_station(station_dir: str) -> List[dict]:
    station = os.path.basename(station_dir.rstrip("/"))
    # (iter, variant) -> accumulated record. Ours' TDOM sub-branch folds into
    # the same key as its perspective branch, so the round's cost is complete.
    acc: Dict[Tuple[int, str], dict] = {}

    for dirpath, dirnames, _ in os.walk(station_dir):
        if os.path.basename(dirpath) != "logs":
            continue
        dirnames[:] = []  # a logs/ dir has no result dirs beneath it
        key = _classify(station_dir, dirpath)
        if key is None:
            continue
        rec = _aggregate(_load_summaries(dirpath))
        if not rec:
            continue

        cur = acc.setdefault(key, {
            "station": station,
            "iter": key[0],
            "variant": key[1],
            "sam3_calls": 0,
            "wall_minutes": 0.0,
            "invocations": 0,
            "world_size": 0,
            "infer_ms_total": 0.0,
            "project_ms_total": 0.0,
            "postprocess_ms_total": 0.0,
            "logs_dir": [],
        })
        cur["sam3_calls"] += rec["sam3_calls"]
        cur["wall_minutes"] += rec["wall_minutes"]
        cur["invocations"] += rec["invocations"]
        cur["world_size"] = max(cur["world_size"], rec["world_size"])
        cur["infer_ms_total"] += rec["infer_ms_total"]
        cur["project_ms_total"] += rec["project_ms_total"]
        cur["postprocess_ms_total"] += rec["postprocess_ms_total"]
        cur["logs_dir"].append(os.path.relpath(dirpath, station_dir))

    rows = []
    for key in sorted(acc):
        r = dict(acc[key])
        r["wall_minutes"] = f"{r['wall_minutes']:.2f}"
        r["infer_ms_total"] = f"{r['infer_ms_total']:.0f}"
        r["project_ms_total"] = f"{r['project_ms_total']:.0f}"
        r["postprocess_ms_total"] = f"{r['postprocess_ms_total']:.0f}"
        r["logs_dir"] = ";".join(sorted(r["logs_dir"]))
        rows.append(r)
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zs-root", default="/data/dataset/PV/ZS_PV")
    p.add_argument("--pattern", default="*-exp2")
    p.add_argument("--out", default=None,
                   help="Output CSV (default: <zs-root>/eval_exp2/run_stats.csv).")
    args = p.parse_args(argv)

    out_path = args.out or os.path.join(args.zs_root, "eval_exp2", "run_stats.csv")

    stations = sorted(
        d for d in glob.glob(os.path.join(args.zs_root, args.pattern))
        if os.path.isdir(d)
    )
    if not stations:
        print(f"[stats] no stations matching {args.pattern}", file=sys.stderr)
        return 1

    rows: List[dict] = []
    for station_dir in stations:
        srows = collect_station(station_dir)
        print(f"[stats] {os.path.basename(station_dir)}: {len(srows)} runs", flush=True)
        rows.extend(srows)

    if not rows:
        print("[stats] nothing collected", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"[stats] -> {out_path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
