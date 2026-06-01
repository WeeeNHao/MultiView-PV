from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import (
    COCO_IOU_THRESHOLDS,
    EvalResult,
    evaluate_shapefiles,
)


def _parse_thresholds(spec: Optional[str]) -> List[float]:
    if not spec:
        return list(COCO_IOU_THRESHOLDS)
    out: List[float] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok:
            out.append(round(float(tok), 4))
    return out or list(COCO_IOU_THRESHOLDS)


def _print_report(result: EvalResult) -> None:
    print("=" * 64)
    print("Vector evaluation (prediction vs ground truth)")
    print("=" * 64)
    print(f"  #predictions : {result.n_pred}")
    print(f"  #ground-truth: {result.n_gt}")
    print("-" * 64)
    print(f"  mAP@[.50:.95]      : {result.mAP:.4f}")
    print(f"  AP50               : {result.ap50:.4f}")
    print(f"  AP75               : {result.ap75:.4f}")
    print(f"  mean matched IoU   : {result.mean_matched_iou:.4f}")
    print(f"  micro IoU          : {result.micro_iou:.4f}")
    print("-" * 64)
    print(f"  total area (pred)  : {result.total_area_pred:.4f}")
    print(f"  total area (gt)    : {result.total_area_gt:.4f}")
    print(f"  total area error   : {result.total_area_error:+.4f}")
    print(f"  total area sq.err  : {result.total_area_squared_error:.4f}")
    print(f"  per-instance MSE   : {result.area_mse_per_instance:.4f}")
    print(f"  per-instance MAE   : {result.area_mae_per_instance:.4f}")
    print("-" * 64)
    print("  Per-IoU-threshold:")
    print(
        f"    {'IoU':>5} {'TP':>6} {'FP':>6} {'FN':>6} "
        f"{'Prec':>7} {'Recall':>7} {'F1':>7} {'AP':>7}"
    )
    for tm in result.per_threshold:
        print(
            f"    {tm.iou_threshold:>5.2f} {tm.tp:>6} {tm.fp:>6} {tm.fn:>6} "
            f"{tm.precision:>7.4f} {tm.recall:>7.4f} {tm.f1:>7.4f} "
            f"{tm.average_precision:>7.4f}"
        )
    print("=" * 64)


def _write_json(result: EvalResult, path: str) -> None:
    payload = {
        "summary": result.summary_dict(),
        "per_threshold": [
            {
                "iou_threshold": tm.iou_threshold,
                "tp": tm.tp,
                "fp": tm.fp,
                "fn": tm.fn,
                "precision": tm.precision,
                "recall": tm.recall,
                "f1": tm.f1,
                "average_precision": tm.average_precision,
                "mean_matched_iou": tm.mean_matched_iou,
            }
            for tm in result.per_threshold
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[eval] wrote JSON report: {path}")


def _write_csv(result: EvalResult, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["iou_threshold", "tp", "fp", "fn", "precision", "recall", "f1", "ap", "mean_matched_iou"]
        )
        for tm in result.per_threshold:
            w.writerow(
                [
                    tm.iou_threshold,
                    tm.tp,
                    tm.fp,
                    tm.fn,
                    f"{tm.precision:.6f}",
                    f"{tm.recall:.6f}",
                    f"{tm.f1:.6f}",
                    f"{tm.average_precision:.6f}",
                    f"{tm.mean_matched_iou:.6f}",
                ]
            )
    print(f"[eval] wrote CSV report: {path}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate predicted PV polygons against ground-truth polygons "
        "(IoU / F1 / area-MSE / mAP)."
    )
    parser.add_argument("--pred", required=True, help="Prediction shapefile (.shp)")
    parser.add_argument("--gt", required=True, help="Ground-truth shapefile (.shp)")
    parser.add_argument(
        "--score-field",
        default="con_weight",
        help="Confidence field used to rank predictions (default: con_weight)",
    )
    parser.add_argument(
        "--primary-iou",
        type=float,
        default=0.5,
        help="IoU threshold for headline F1 / IoU / area matching (default: 0.5)",
    )
    parser.add_argument(
        "--iou-thresholds",
        default=None,
        help="Comma-separated IoU sweep for mAP (default: 0.50,0.55,...,0.95)",
    )
    parser.add_argument(
        "--label",
        type=int,
        default=None,
        help="Only evaluate features with this class label (default: all)",
    )
    parser.add_argument("--json", default=None, help="Optional path to write JSON report")
    parser.add_argument("--csv", default=None, help="Optional path to write CSV report")
    args = parser.parse_args(argv)

    for tag, path in (("pred", args.pred), ("gt", args.gt)):
        if not os.path.exists(path):
            parser.error(f"--{tag} shapefile not found: {path}")

    thresholds = _parse_thresholds(args.iou_thresholds)

    result = evaluate_shapefiles(
        pred_shp=args.pred,
        gt_shp=args.gt,
        iou_thresholds=thresholds,
        primary_iou=args.primary_iou,
        score_field=args.score_field,
        label=args.label,
    )

    _print_report(result)

    if args.json:
        _write_json(result, args.json)
    if args.csv:
        _write_csv(result, args.csv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())