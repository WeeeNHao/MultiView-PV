#!/usr/bin/env python
"""Batch-evaluate the 0518 power-station results against ground truth.

For every station folder matching ``*0518*`` under the ZS_PV root, this walks
each iteration (``iter_<i>``) and result variant (the full direction set, a
table-7 camera-direction subset, or a legacy view_num sweep entry), evaluates
the predicted polygons against the station ground truth, and writes CSV reports.

Two complementary metric families are reported per (iter, variant):

  * Area-level ("semantic mask", matches QGIS "Area Results"): the whole
    prediction footprint and the whole GT footprint are each dissolved into a
    single mask, then compared -> area_IoU / area_Dice / area_Prec / area_Rec.
  * Object-level + COCO: per-instance greedy matching at IoU >= 0.5 gives
    obj_Prec / obj_Rec / obj_F1 / obj_mIoU and TP/FP/FN, plus the COCO sweep
    mAP@[.50:.95] / AP50 / AP75, the strict AP@{.90,.95,.975} that is the
    headline number here (object F1 is flat from IoU 0.50 to 0.90 on these
    stations, so only the high thresholds separate methods), and per-instance
    area error (MSE / MAE).

Outputs (one folder per station, requirement #1):

    <out-root>/
        all_stations_summary.csv          # every (station, iter, variant) row
        <station>/
            summary.csv                   # this station's (iter, variant) rows
            per_threshold.csv             # full IoU sweep per (iter, variant)

GT path convention: a station folder ``001-BeiOu-0518`` maps to ground truth
``/data/dataset/PV/001-BeiOu/gt/pv.shp`` (strip the ``-0518`` suffix).

Usage:
    python scripts/eval_0518_batch.py
    python scripts/eval_0518_batch.py --zs-root /data/dataset/PV/ZS_PV \
        --out-root /data/dataset/PV/ZS_PV/eval_0518 --primary-iou 0.5
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import traceback
from typing import Dict, List, Optional, Tuple

# Quiet down the tqdm bars used by the shapefile reader.
os.environ.setdefault("TQDM_DISABLE", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from osgeo import ogr  # noqa: E402

ogr.DontUseExceptions()

from evaluation.geom import area_overlap_metrics, dissolve_features  # noqa: E402
from evaluation.metrics import EvalResult, evaluate_feature_lists  # noqa: E402
from io_flow.shp_io import read_features_from_shapefile  # noqa: E402

SCORE_FIELD = "con_weight"

# Columns of the headline summary CSV, in order.
SUMMARY_FIELDS = [
    "station",
    "iter",
    "variant",
    "n_pred",
    "n_gt",
    # --- Area-level (QGIS "Area Results"): dissolved-mask overlap ---
    "area_IoU",
    "area_Dice",
    "area_Prec",
    "area_Rec",
    # --- Object-level @ IoU>=0.5 (QGIS "Object Results") ---
    "obj_Prec",
    "obj_Rec",
    "obj_F1",
    "obj_mIoU",
    "TP",
    "FP",
    "FN",
    # --- COCO detection sweep ---
    "mAP",
    "AP50",
    "AP75",
    # --- Strict high-IoU AP: the only range that still separates methods here ---
    "AP_high",
    "AP90",
    "AP95",
    "AP975",
    # --- Panoptic Quality: SQ (mask fit) x RQ (detection) = PQ ---
    "SQ",
    "RQ",
    "PQ",
    "AJI",
    # --- Per-instance area error ---
    "area_MSE",
    "area_MAE",
    # --- Positional error over matched pairs (map units, m) ---
    "centroid_RMSE",
    "centroid_MAE",
    # --- Instance-splitting errors (counts and rates) ---
    "over_seg",
    "under_seg",
    "over_seg_rate",
    "under_seg_rate",
    "pred_shp",
]

PER_THRESHOLD_FIELDS = [
    "station",
    "iter",
    "variant",
    "iou_threshold",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
    "ap",
    "mean_matched_iou",
]

_ITER_RE = re.compile(r"^iter_(\d+)$")
_VIEW_RE = re.compile(r"^view_(\d+)$")

# Station-root entries that are never a method root (bookkeeping, not results).
_NON_METHOD_DIRS = {"_logs", "_preflight", "eval", "logs"}


def gt_path_for_station(station_dir: str) -> str:
    """Map ``.../001-BeiOu-exp`` -> ``/data/dataset/PV/001-BeiOu/gt/pv.shp``."""
    name = os.path.basename(station_dir.rstrip("/"))
    # Trailing digits let re-runs live side by side (``-exp``, ``-exp2``, ...)
    # while still resolving to the one ground truth for that station.
    base = re.sub(r"-(0518|exp\d*)$", "", name)
    return os.path.join("/data/dataset/PV", base, "gt", "pv.shp")


def discover_predictions(station_dir: str) -> List[Tuple[int, str, str]]:
    """Return sorted (iter, variant, pred_shp) tuples that exist on disk.

    Four layouts are recognised:
      iter_<i>/full/final.shp        -> variant "full"     (main line)
      iter_<i>/dirs/<name>/final.shp -> variant "<name>"   (table 7 view sets)
      iter_<i>/views/view_<v>/view_<v>.shp -> variant "view_<v>"
      <method>/iter_<i>/final.shp    -> variant "<method>"

    The third is the retired view_num sweep, kept so historical result trees
    still evaluate. The fourth is the *inverted* layout the baseline and
    headline stages write (``dom`` from SD, ``ours`` from SP, ``m1``/``m2``/
    ``m3`` from SM1-SM3): method on the outside, iteration on the inside. Method
    roots are discovered by shape rather than by name, so a new baseline stage
    becomes evaluable without touching this script.
    """
    found: List[Tuple[int, str, str]] = []
    for entry in os.listdir(station_dir):
        if _ITER_RE.match(entry) or entry in _NON_METHOD_DIRS:
            continue
        method_dir = os.path.join(station_dir, entry)
        if not os.path.isdir(method_dir):
            continue
        for sub in os.listdir(method_dir):
            m_sub = _ITER_RE.match(sub)
            if not m_sub:
                continue
            shp = os.path.join(method_dir, sub, "final.shp")
            if os.path.isfile(shp):
                found.append((int(m_sub.group(1)), entry, shp))

    for entry in os.listdir(station_dir):
        m_iter = _ITER_RE.match(entry)
        if not m_iter:
            continue
        iter_idx = int(m_iter.group(1))
        iter_dir = os.path.join(station_dir, entry)

        shp = os.path.join(iter_dir, "full", "final.shp")
        if os.path.isfile(shp):
            found.append((iter_idx, "full", shp))

        dirs_dir = os.path.join(iter_dir, "dirs")
        if os.path.isdir(dirs_dir):
            for name in os.listdir(dirs_dir):
                shp = os.path.join(dirs_dir, name, "final.shp")
                if os.path.isfile(shp):
                    found.append((iter_idx, name, shp))

        views_dir = os.path.join(iter_dir, "views")
        if os.path.isdir(views_dir):
            for vname in os.listdir(views_dir):
                if not _VIEW_RE.match(vname):
                    continue
                shp = os.path.join(views_dir, vname, f"{vname}.shp")
                if os.path.isfile(shp):
                    found.append((iter_idx, vname, shp))

    found.sort(key=lambda t: (t[0], t[1]))
    return found


def summary_row(station: str, iter_idx: int, variant: str, shp: str,
                r: EvalResult, area: Dict[str, float]) -> dict:
    return {
        "station": station,
        "iter": iter_idx,
        "variant": variant,
        "n_pred": r.n_pred,
        "n_gt": r.n_gt,
        "area_IoU": f"{area['area_iou']:.6f}",
        "area_Dice": f"{area['area_dice']:.6f}",
        "area_Prec": f"{area['area_precision']:.6f}",
        "area_Rec": f"{area['area_recall']:.6f}",
        "obj_Prec": f"{r.precision:.6f}",
        "obj_Rec": f"{r.recall:.6f}",
        "obj_F1": f"{r.f1:.6f}",
        "obj_mIoU": f"{r.mean_matched_iou:.6f}",
        "TP": r.tp,
        "FP": r.fp,
        "FN": r.fn,
        "mAP": f"{r.mAP:.6f}",
        "AP50": f"{r.ap50:.6f}",
        "AP75": f"{r.ap75:.6f}",
        "AP_high": f"{r.ap_high:.6f}",
        "AP90": f"{r.ap90:.6f}",
        "AP95": f"{r.ap95:.6f}",
        "AP975": f"{r.ap975:.6f}",
        "SQ": f"{r.sq:.6f}",
        "RQ": f"{r.rq:.6f}",
        "PQ": f"{r.pq:.6f}",
        "AJI": f"{r.aji:.6f}",
        "area_MSE": f"{r.area_mse_per_instance:.6f}",
        "area_MAE": f"{r.area_mae_per_instance:.6f}",
        "centroid_RMSE": f"{r.centroid_rmse:.6f}",
        "centroid_MAE": f"{r.centroid_mae:.6f}",
        "over_seg": r.over_seg_count,
        "under_seg": r.under_seg_count,
        "over_seg_rate": f"{r.over_seg_rate:.6f}",
        "under_seg_rate": f"{r.under_seg_rate:.6f}",
        "pred_shp": shp,
    }


def per_threshold_rows(station: str, iter_idx: int, variant: str,
                       r: EvalResult) -> List[dict]:
    rows = []
    for tm in r.per_threshold:
        rows.append({
            "station": station,
            "iter": iter_idx,
            "variant": variant,
            "iou_threshold": f"{tm.iou_threshold:.2f}",
            "tp": tm.tp,
            "fp": tm.fp,
            "fn": tm.fn,
            "precision": f"{tm.precision:.6f}",
            "recall": f"{tm.recall:.6f}",
            "f1": f"{tm.f1:.6f}",
            "ap": f"{tm.average_precision:.6f}",
            "mean_matched_iou": f"{tm.mean_matched_iou:.6f}",
        })
    return rows


def write_csv(path: str, fields: List[str], rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zs-root", default="/data/dataset/PV/ZS_PV",
                   help="Root containing the *0518* station folders.")
    p.add_argument("--pattern", default="*-exp",
                   help="Glob for station folder names (default: *-exp).")
    p.add_argument("--out-root", default=None,
                   help="Output root (default: <zs-root>/eval_0518).")
    p.add_argument("--primary-iou", type=float, default=0.5,
                   help="IoU threshold for object-level F1/IoU matching.")
    p.add_argument("--score-field", default=SCORE_FIELD,
                   help="Confidence field used to rank predictions.")
    args = p.parse_args(argv)

    out_root = args.out_root or os.path.join(args.zs_root, "eval_exp")

    import glob
    stations = sorted(
        d for d in glob.glob(os.path.join(args.zs_root, args.pattern))
        if os.path.isdir(d)
    )
    if not stations:
        print(f"[eval] no stations matching {args.pattern} under {args.zs_root}",
              file=sys.stderr)
        return 1

    print(f"[eval] {len(stations)} station(s): "
          + ", ".join(os.path.basename(s) for s in stations), flush=True)

    all_summary: List[dict] = []

    for station_dir in stations:
        station = os.path.basename(station_dir)
        gt = gt_path_for_station(station_dir)
        if not os.path.isfile(gt):
            print(f"[eval] !! GT not found for {station}: {gt} -- skipping",
                  file=sys.stderr, flush=True)
            continue

        preds = discover_predictions(station_dir)
        print(f"\n[eval] === {station} === ({len(preds)} pred sets) gt={gt}",
              flush=True)

        # Read + dissolve GT once per station (reused across all iter/view).
        gts = read_features_from_shapefile(gt, score_field=args.score_field)
        gt_union = dissolve_features(gts)
        print(f"[eval]   gt features={len(gts)}", flush=True)

        st_summary: List[dict] = []
        st_threshold: List[dict] = []

        for iter_idx, variant, shp in preds:
            tag = f"iter={iter_idx} variant={variant}"
            try:
                preds_f = read_features_from_shapefile(shp, score_field=args.score_field)
                r = evaluate_feature_lists(
                    preds=preds_f,
                    gts=gts,
                    primary_iou=args.primary_iou,
                    score_field=args.score_field,
                )
                pred_union = dissolve_features(preds_f)
                area = area_overlap_metrics(pred_union, gt_union)
            except Exception as exc:  # noqa: BLE001
                print(f"[eval]   {tag}  FAILED: {exc}", file=sys.stderr, flush=True)
                traceback.print_exc()
                continue

            print(f"[eval]   {tag:<22} n_pred={r.n_pred:<5} n_gt={r.n_gt:<5} "
                  f"areaIoU={area['area_iou']:.4f} objF1={r.f1:.4f} "
                  f"mAP={r.mAP:.4f} APhi={r.ap_high:.4f} cRMSE={r.centroid_rmse:.3f} "
                  f"over/under={r.over_seg_count}/{r.under_seg_count}", flush=True)

            srow = summary_row(station, iter_idx, variant, shp, r, area)
            st_summary.append(srow)
            all_summary.append(srow)
            st_threshold.extend(per_threshold_rows(station, iter_idx, variant, r))

        if st_summary:
            st_dir = os.path.join(out_root, station)
            write_csv(os.path.join(st_dir, "summary.csv"),
                      SUMMARY_FIELDS, st_summary)
            write_csv(os.path.join(st_dir, "per_threshold.csv"),
                      PER_THRESHOLD_FIELDS, st_threshold)
            print(f"[eval]   -> {st_dir}/summary.csv ({len(st_summary)} rows)",
                  flush=True)

    if all_summary:
        combined = os.path.join(out_root, "all_stations_summary.csv")
        write_csv(combined, SUMMARY_FIELDS, all_summary)
        print(f"\n[eval] combined summary: {combined} ({len(all_summary)} rows)",
              flush=True)
    else:
        print("\n[eval] no results produced", file=sys.stderr, flush=True)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
