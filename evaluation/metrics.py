from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from utils.common import FeatureList

from .geom import build_overlap_matrices, feature_centroid, geometry_area
from .matcher import greedy_match

# COCO IoU sweep: 0.50, 0.55, ..., 0.95
COCO_IOU_THRESHOLDS: Tuple[float, ...] = tuple(round(0.50 + 0.05 * i, 2) for i in range(10))

# High-IoU thresholds averaged into the strict AP. On these stations object F1 is
# flat from IoU 0.50 all the way to 0.90 -- every method that finds a panel finds
# it well enough to clear 0.90 -- so the COCO sweep cannot separate them and the
# whole benefit of iterative refinement (which is boundary accuracy, not
# detection) is invisible. Only 0.95 and above still discriminate.
HIGH_IOU_THRESHOLDS: Tuple[float, ...] = (0.90, 0.95, 0.975)

# Thresholds actually evaluated: the COCO sweep plus whatever the strict AP needs
# that it does not already cover (only 0.975 today).
EVAL_IOU_THRESHOLDS: Tuple[float, ...] = tuple(
    sorted(set(COCO_IOU_THRESHOLDS) | set(HIGH_IOU_THRESHOLDS))
)

# Containment ratio above which a prediction counts as "lying inside" a GT
# instance (and vice versa) for the over/under-segmentation counts.
DEFAULT_SEG_CONTAINMENT: float = 0.5


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

    # mAP = mean of per-threshold AP over the COCO sweep (0.50:0.05:0.95)
    mAP: float
    ap50: float
    ap75: float
    # Strict AP: mean of per-threshold AP over HIGH_IOU_THRESHOLDS. This is the
    # headline number for these stations -- see that constant for why.
    ap_high: float
    ap90: float
    ap95: float
    ap975: float

    # Panoptic Quality (Kirillov et al. 2019), at the primary IoU threshold.
    #   SQ = mean IoU over matched pairs      (how well matched masks fit)
    #   RQ = TP/(TP+0.5FP+0.5FN), identical to F1  (how many were found at all)
    #   PQ = SQ * RQ
    # Reported because SQ and RQ separate the two failure modes this study cares
    # about: M1 scores SQ 0.84 (its masks are fine) with RQ 0.03 (it shatters
    # every module), which a single detection number cannot express.
    sq: float
    rq: float
    pq: float

    # Aggregated Jaccard Index: area-weighted, penalises over- and
    # under-segmentation directly (see _aggregated_jaccard_index).
    aji: float

    # Segmentation overlap (computed at the primary IoU threshold matching)
    mean_matched_iou: float      # macro IoU: average IoU over matched pairs
    micro_iou: float             # sum(inter) / sum(union) over matched pairs
    precision: float
    recall: float
    f1: float
    tp: int                      # object TP/FP/FN at the primary IoU threshold
    fp: int
    fn: int

    # Area error (MSE), in squared map units (e.g. m^4 if areas are m^2)
    area_mse_per_instance: float  # MSE over matched (pred_area - gt_area)
    area_mae_per_instance: float  # MAE over matched (pred_area - gt_area)

    # Positional error over matched pairs, in map units (m for a projected CRS)
    centroid_rmse: float
    centroid_mae: float

    # Instance-splitting errors, independent of the one-to-one matching:
    #   over  = one GT instance carved into several predictions
    #   under = several GT instances swallowed by one prediction
    over_seg_count: int      # GT instances containing >=2 predictions
    under_seg_count: int     # predictions containing >=2 GT instances
    over_seg_rate: float     # over_seg_count / n_gt
    under_seg_rate: float    # under_seg_count / n_pred
    # total_area_pred: float
    # total_area_gt: float
    # total_area_error: float       # pred_total - gt_total (signed)
    # total_area_squared_error: float  # (pred_total - gt_total) ** 2

    per_threshold: List[ThresholdMetrics] = field(default_factory=list)

    def summary_dict(self) -> Dict[str, float]:
        return {
            "n_pred": self.n_pred,
            "n_gt": self.n_gt,
            "mAP@[.50:.95]": self.mAP,
            "AP50": self.ap50,
            "AP75": self.ap75,
            "AP@{.90,.95,.975}": self.ap_high,
            "AP90": self.ap90,
            "AP95": self.ap95,
            "AP975": self.ap975,
            "SQ": self.sq,
            "RQ": self.rq,
            "PQ": self.pq,
            "AJI": self.aji,
            "mean_matched_IoU": self.mean_matched_iou,
            "micro_IoU": self.micro_iou,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "area_MSE_per_instance": self.area_mse_per_instance,
            "area_MAE_per_instance": self.area_mae_per_instance,
            "centroid_RMSE": self.centroid_rmse,
            "centroid_MAE": self.centroid_mae,
            "over_seg_rate": self.over_seg_rate,
            "under_seg_rate": self.under_seg_rate,
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


def _aggregated_jaccard_index(
    iou_matrix: Sequence[Sequence[float]],
    inter_matrix: Sequence[Sequence[float]],
    pred_areas: Sequence[float],
    gt_areas: Sequence[float],
) -> float:
    """Aggregated Jaccard Index (Kumar et al. 2017).

        AJI = sum_i |G_i & P_j*(i)|  /  ( sum_i |G_i | P_j*(i)| + sum_{k unused} |P_k| )

    Each GT takes the highest-IoU prediction still available; that prediction is
    then consumed. Predictions never claimed by any GT add their **whole area**
    to the denominator, and an unmatched GT adds its own area, so the score is
    pushed down by over-segmentation (extra fragments), under-segmentation
    (one prediction swallowing several GTs, leaving the others unmatched) and
    plain false positives alike.

    That is the property F1 and AP lack: they count an instance once, so 5866
    half-module fragments and 1772 clean modules can both score "wrong" without
    the metric expressing *how* wrong. AJI is area-weighted, so it degrades
    smoothly instead of collapsing to zero.
    """
    n_pred = len(pred_areas)
    n_gt = len(gt_areas)
    if n_gt == 0:
        return 0.0

    used = [False] * n_pred
    inter_sum = 0.0
    union_sum = 0.0

    for j in range(n_gt):
        best_i = -1
        best_iou = 0.0
        for i in range(n_pred):
            if used[i]:
                continue
            v = iou_matrix[i][j]
            if v > best_iou:
                best_iou = v
                best_i = i
        if best_i >= 0:
            used[best_i] = True
            inter = float(inter_matrix[best_i][j])
            inter_sum += inter
            union_sum += float(pred_areas[best_i]) + float(gt_areas[j]) - inter
        else:
            # No prediction overlaps this GT at all: it contributes only union.
            union_sum += float(gt_areas[j])

    union_sum += sum(a for i, a in enumerate(pred_areas) if not used[i])
    return inter_sum / union_sum if union_sum > 0 else 0.0


def _segmentation_errors(
    inter_matrix: Sequence[Sequence[float]],
    pred_areas: Sequence[float],
    gt_areas: Sequence[float],
    containment: float,
) -> Tuple[int, int]:
    """Count over- and under-segmented instances.

    Deliberately *not* derived from the one-to-one match: that match keeps at
    most one prediction per GT, so a GT split in two looks like 1 TP + 1 FP and
    the split itself is invisible. Association here is by containment instead --
    a prediction belongs to a GT when at least ``containment`` of the
    prediction's own area falls inside it, and symmetrically for the GT. IoU
    cannot express this: two predictions splitting one GT can never both reach
    IoU 0.5 with it, so an IoU-based rule would report no over-segmentation by
    construction.

    Returns ``(over_seg_count, under_seg_count)``:
      over  -- GT instances that contain >= 2 predictions (one panel split up)
      under -- predictions that contain >= 2 GT instances (panels merged)
    """
    n_pred = len(pred_areas)
    n_gt = len(gt_areas)
    if n_pred == 0 or n_gt == 0:
        return 0, 0

    preds_inside_gt = [0] * n_gt
    gts_inside_pred = [0] * n_pred
    for i in range(n_pred):
        row = inter_matrix[i]
        pa = pred_areas[i]
        for j in range(n_gt):
            inter = row[j]
            if inter <= 0.0:
                continue
            if pa > 0.0 and inter / pa >= containment:
                preds_inside_gt[j] += 1
            ga = gt_areas[j]
            if ga > 0.0 and inter / ga >= containment:
                gts_inside_pred[i] += 1

    over = sum(1 for c in preds_inside_gt if c >= 2)
    under = sum(1 for c in gts_inside_pred if c >= 2)
    return over, under


def evaluate_feature_lists(
    preds: FeatureList,
    gts: FeatureList,
    iou_thresholds: Sequence[float] = EVAL_IOU_THRESHOLDS,
    primary_iou: float = 0.5,
    score_field: str = "score",
    seg_containment: float = DEFAULT_SEG_CONTAINMENT,
) -> EvalResult:
    """Evaluate predicted features against ground-truth features.

    Args:
        preds: predicted features (each a dict as produced by
            ``read_features_from_shapefile``).
        gts: ground-truth features.
        iou_thresholds: IoU sweep used for mAP and the per-threshold report.
        primary_iou: threshold used for the headline F1 / IoU / area metrics.
        score_field: feature key holding the confidence used to rank predictions.
        seg_containment: area fraction above which one instance counts as lying
            inside another, for the over/under-segmentation counts.
    """
    n_pred = len(preds)
    n_gt = len(gts)

    pred_scores = [float(p.get(score_field, p.get("con_weight", 0.0))) for p in preds]

    iou_matrix, inter_matrix = build_overlap_matrices(preds, gts)

    # --- mAP over the IoU sweep ---
    per_threshold: List[ThresholdMetrics] = []
    # Keyed by threshold so mAP stays the mean over exactly the COCO ten even
    # though the sweep now evaluates extra strict thresholds too. Averaging over
    # whatever happened to be swept would silently redefine mAP@[.50:.95].
    ap_by_thr: Dict[float, float] = {}
    ap50 = 0.0
    ap75 = 0.0
    for thr in iou_thresholds:
        ap, tp, fp, fn, mean_iou = _average_precision(iou_matrix, pred_scores, n_gt, thr)
        ap_by_thr[round(float(thr), 6)] = ap
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

    def _ap_at(thr: float) -> float:
        return ap_by_thr.get(round(float(thr), 6), 0.0)

    def _mean_ap_over(thrs: Sequence[float]) -> float:
        vals = [ap_by_thr[k] for k in (round(float(t), 6) for t in thrs) if k in ap_by_thr]
        return sum(vals) / len(vals) if vals else 0.0

    mAP = _mean_ap_over(COCO_IOU_THRESHOLDS)
    ap_high = _mean_ap_over(HIGH_IOU_THRESHOLDS)
    ap90, ap95, ap975 = _ap_at(0.90), _ap_at(0.95), _ap_at(0.975)

    # --- Primary-threshold matching for IoU / area metrics ---
    matches, unmatched_pred, unmatched_gt = greedy_match(
        iou_matrix, primary_iou, pred_scores
    )

    tp_primary = len(matches)
    fp_primary = len(unmatched_pred)
    fn_primary = len(unmatched_gt)
    precision_primary = tp_primary / (tp_primary + fp_primary) if (tp_primary + fp_primary) > 0 else 0.0
    recall_primary = tp_primary / (tp_primary + fn_primary) if (tp_primary + fn_primary) > 0 else 0.0
    f1_primary = (
        (2 * precision_primary * recall_primary / (precision_primary + recall_primary))
        if (precision_primary + recall_primary) > 0
        else 0.0
    )

    matched_ious = [m[2] for m in matches]
    mean_matched_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0

    # Micro IoU over matched pairs: sum(intersection) / sum(union).
    inter_sum = 0.0
    union_sum = 0.0
    sq_errors: List[float] = []
    abs_errors: List[float] = []
    centroid_sq: List[float] = []
    centroid_abs: List[float] = []
    for pi, gj, iou in matches:
        pa = geometry_area(preds[pi])
        ga = geometry_area(gts[gj])
        pc = feature_centroid(preds[pi])
        gc = feature_centroid(gts[gj])
        if pc is not None and gc is not None:
            d = math.hypot(pc[0] - gc[0], pc[1] - gc[1])
            centroid_sq.append(d * d)
            centroid_abs.append(d)
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
    centroid_rmse = math.sqrt(sum(centroid_sq) / len(centroid_sq)) if centroid_sq else 0.0
    centroid_mae = sum(centroid_abs) / len(centroid_abs) if centroid_abs else 0.0

    pred_areas_all = [geometry_area(p) for p in preds]
    gt_areas_all = [geometry_area(g) for g in gts]
    aji = _aggregated_jaccard_index(iou_matrix, inter_matrix,
                                    pred_areas_all, gt_areas_all)

    over_seg, under_seg = _segmentation_errors(
        inter_matrix,
        pred_areas_all,
        gt_areas_all,
        seg_containment,
    )

    return EvalResult(
        n_pred=n_pred,
        n_gt=n_gt,
        mAP=mAP,
        ap50=ap50,
        ap75=ap75,
        ap_high=ap_high,
        ap90=ap90,
        ap95=ap95,
        ap975=ap975,
        sq=mean_matched_iou,
        rq=f1_primary,
        pq=mean_matched_iou * f1_primary,
        aji=aji,
        mean_matched_iou=mean_matched_iou,
        micro_iou=micro_iou,
        precision=precision_primary,
        recall=recall_primary,
        f1=f1_primary,
        tp=tp_primary,
        fp=fp_primary,
        fn=fn_primary,
        area_mse_per_instance=area_mse,
        area_mae_per_instance=area_mae,
        centroid_rmse=centroid_rmse,
        centroid_mae=centroid_mae,
        over_seg_count=over_seg,
        under_seg_count=under_seg,
        over_seg_rate=(over_seg / n_gt) if n_gt > 0 else 0.0,
        under_seg_rate=(under_seg / n_pred) if n_pred > 0 else 0.0,
        per_threshold=per_threshold,
    )


def evaluate_shapefiles(
    pred_shp: str,
    gt_shp: str,
    iou_thresholds: Sequence[float] = EVAL_IOU_THRESHOLDS,
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
