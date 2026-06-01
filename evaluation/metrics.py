from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from utils.common import Feature, FeatureList

from .geom import build_iou_matrix, geometry_area
from .matcher import greedy_match

# COCO IoU sweep: 0.50, 0.55, ..., 0.95
COCO_IOU_THRESHOLDS: Tuple[float, ...] = tuple(round(0.50 + 0.05 * i, 2) for i in range(10))


@dataclass
class ThresholdMetrics:
    """Detection metrics at a single IoU threshold."""

    iou_threshold: float
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    average_precision: float
    mean_matched_iou: float  # mean IoU over the TP matches (0 if none)


@dataclass
class EvalResult:
    """Aggregate evaluation between a prediction set and a ground-truth set."""

    n_pred: int
    n_gt: int

    # mAP = mean of per-threshold AP over the IoU sweep (COCO style)
    mAP: float
    ap50: float
    ap75: float

    # Segmentation overlap (computed at the primary IoU threshold matching)
    mean_matched_iou: float      # macro IoU: average IoU over matched pairs
    micro_iou: float             # sum(inter) / sum(union) over matched pairs

    # Area error (MSE), in squared map units (e.g. m^4 if areas are m^2)
    area_mse_per_instance: float  # MSE over matched (pred_area - gt_area)
    area_mae_per_instance: float  # MAE over matched (pred_area - gt_area)
    total_area_pred: float
    total_area_gt: float
    total_area_error: float       # pred_total - gt_total (signed)
    total_area_squared_error: float  # (pred_total - gt_total) ** 2

    per_threshold: List[ThresholdMetrics] = field(default_factory=list)

    def summary_dict(self) -> Dict[str, float]:
        return {
            "n_pred": self.n_pred,
            "n_gt": self.n_gt,
            "mAP@[.50:.95]": self.mAP,
            "AP50": self.ap50,
            "AP75": self.ap75,
            "mean_matched_IoU": self.mean_matched_iou,
            "micro_IoU": self.micro_iou,
            "area_MSE_per_instance": self.area_mse_per_instance,
            "area_MAE_per_instance": self.area_mae_per_instance,
            "total_area_pred": self.total_area_pred,
            "total_area_gt": self.total_area_gt,
            "total_area_error": self.total_area_error,
            "total_area_squared_error": self.total_area_squared_error,
        }


def _average_precision(
    iou_matrix: Sequence[Sequence[float]],
    pred_scores: Sequence[float],
    n_gt: int,
    iou_threshold: float,
) -> Tuple[float, int, int, int, float]:
    """COCO-style AP at one IoU threshold via the precision/recall curve.

    Returns (ap, tp, fp, fn, mean_matched_iou).
    """
    n_pred = len(iou_matrix)
    if n_gt == 0:
        # No ground truth: AP is 1.0 if there are also no predictions, else 0.
        return (1.0 if n_pred == 0 else 0.0, 0, n_pred, 0, 0.0)
    if n_pred == 0:
        return (0.0, 0, 0, n_gt, 0.0)

    order = sorted(range(n_pred), key=lambda i: float(pred_scores[i]), reverse=True)
    gt_taken = [False] * n_gt

    tp_flags: List[int] = []
    matched_ious: List[float] = []

    for pi in order:
        best_iou = iou_threshold
        best_gt = -1
        row = iou_matrix[pi]
        for gj in range(n_gt):
            if gt_taken[gj]:
                continue
            if row[gj] >= best_iou:
                best_iou = row[gj]
                best_gt = gj
        if best_gt >= 0:
            gt_taken[best_gt] = True
            tp_flags.append(1)
            matched_ious.append(float(row[best_gt]))
        else:
            tp_flags.append(0)

    # Running precision / recall along the score-sorted predictions.
    cum_tp = 0
    cum_fp = 0
    precisions: List[float] = []
    recalls: List[float] = []
    for flag in tp_flags:
        if flag:
            cum_tp += 1
        else:
            cum_fp += 1
        precisions.append(cum_tp / (cum_tp + cum_fp))
        recalls.append(cum_tp / n_gt)

    ap = _ap_from_pr_curve(recalls, precisions)

    tp = cum_tp
    fp = cum_fp
    fn = n_gt - tp
    mean_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0
    return (ap, tp, fp, fn, mean_iou)


def _ap_from_pr_curve(recalls: Sequence[float], precisions: Sequence[float]) -> float:
    """101-point interpolated AP (COCO convention)."""
    if not recalls:
        return 0.0

    # Make precision monotonically non-increasing from the right.
    prec = list(precisions)
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])

    recall_thresholds = [r / 100.0 for r in range(101)]
    ap = 0.0
    for rt in recall_thresholds:
        p = 0.0
        for r, pr in zip(recalls, prec):
            if r >= rt:
                p = pr
                break
        ap += p
    return ap / len(recall_thresholds)


def evaluate_feature_lists(
    preds: FeatureList,
    gts: FeatureList,
    iou_thresholds: Sequence[float] = COCO_IOU_THRESHOLDS,
    primary_iou: float = 0.5,
    score_field: str = "score",
) -> EvalResult:
    """Evaluate predicted features against ground-truth features.

    Args:
        preds: predicted features (each a dict as produced by
            ``read_features_from_shapefile``).
        gts: ground-truth features.
        iou_thresholds: IoU sweep used for mAP and the per-threshold report.
        primary_iou: threshold used for the headline F1 / IoU / area metrics.
        score_field: feature key holding the confidence used to rank predictions.
    """
    n_pred = len(preds)
    n_gt = len(gts)

    pred_scores = [float(p.get(score_field, p.get("con_weight", 0.0))) for p in preds]

    iou_matrix = build_iou_matrix(preds, gts)

    # --- mAP over the IoU sweep ---
    per_threshold: List[ThresholdMetrics] = []
    ap_values: List[float] = []
    ap50 = 0.0
    ap75 = 0.0
    for thr in iou_thresholds:
        ap, tp, fp, fn, mean_iou = _average_precision(iou_matrix, pred_scores, n_gt, thr)
        ap_values.append(ap)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_threshold.append(
            ThresholdMetrics(
                iou_threshold=thr,
                tp=tp,
                fp=fp,
                fn=fn,
                precision=precision,
                recall=recall,
                f1=f1,
                average_precision=ap,
                mean_matched_iou=mean_iou,
            )
        )
        if abs(thr - 0.50) < 1e-6:
            ap50 = ap
        if abs(thr - 0.75) < 1e-6:
            ap75 = ap

    mAP = sum(ap_values) / len(ap_values) if ap_values else 0.0

    # --- Primary-threshold matching for IoU / area metrics ---
    matches, _unmatched_pred, _unmatched_gt = greedy_match(
        iou_matrix, primary_iou, pred_scores
    )

    matched_ious = [m[2] for m in matches]
    mean_matched_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0

    # Micro IoU over matched pairs: sum(intersection) / sum(union).
    inter_sum = 0.0
    union_sum = 0.0
    sq_errors: List[float] = []
    abs_errors: List[float] = []
    for pi, gj, iou in matches:
        pa = geometry_area(preds[pi])
        ga = geometry_area(gts[gj])
        # Recover intersection/union from IoU and areas:
        #   IoU = I / (pa + ga - I)  =>  I = IoU * (pa + ga) / (1 + IoU)
        if iou > 0:
            inter = iou * (pa + ga) / (1.0 + iou)
        else:
            inter = 0.0
        union = pa + ga - inter
        inter_sum += inter
        union_sum += union
        err = pa - ga
        sq_errors.append(err * err)
        abs_errors.append(abs(err))

    micro_iou = (inter_sum / union_sum) if union_sum > 0 else 0.0
    area_mse = sum(sq_errors) / len(sq_errors) if sq_errors else 0.0
    area_mae = sum(abs_errors) / len(abs_errors) if abs_errors else 0.0

    total_area_pred = sum(geometry_area(p) for p in preds)
    total_area_gt = sum(geometry_area(g) for g in gts)
    total_area_error = total_area_pred - total_area_gt

    return EvalResult(
        n_pred=n_pred,
        n_gt=n_gt,
        mAP=mAP,
        ap50=ap50,
        ap75=ap75,
        mean_matched_iou=mean_matched_iou,
        micro_iou=micro_iou,
        area_mse_per_instance=area_mse,
        area_mae_per_instance=area_mae,
        total_area_pred=total_area_pred,
        total_area_gt=total_area_gt,
        total_area_error=total_area_error,
        total_area_squared_error=total_area_error * total_area_error,
        per_threshold=per_threshold,
    )


def evaluate_shapefiles(
    pred_shp: str,
    gt_shp: str,
    iou_thresholds: Sequence[float] = COCO_IOU_THRESHOLDS,
    primary_iou: float = 0.5,
    score_field: str = "con_weight",
    label: Optional[int] = None,
) -> EvalResult:
    """Read two shapefiles and evaluate the prediction against ground truth.

    Args:
        pred_shp: path to the prediction shapefile.
        gt_shp: path to the ground-truth shapefile.
        score_field: confidence field used to rank predictions.
        label: if set, only features with this ``label`` are evaluated
            (ground truth and prediction filtered identically).
    """
    from io_flow.shp_io import read_features_from_shapefile

    preds = read_features_from_shapefile(pred_shp, score_field=score_field)
    gts = read_features_from_shapefile(gt_shp, score_field=score_field)

    if label is not None:
        preds = [f for f in preds if int(f.get("label", 0)) == label]
        gts = [f for f in gts if int(f.get("label", 0)) == label]

    return evaluate_feature_lists(
        preds=preds,
        gts=gts,
        iou_thresholds=iou_thresholds,
        primary_iou=primary_iou,
        score_field="con_weight",
    )
