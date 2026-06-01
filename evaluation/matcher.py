from __future__ import annotations

from typing import List, Sequence, Tuple


def greedy_match(
    iou_matrix: Sequence[Sequence[float]],
    iou_threshold: float,
    pred_scores: Sequence[float],
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """Greedy one-to-one matching by descending prediction score.

    This is the COCO matching rule: predictions are considered from highest to
    lowest confidence; each prediction takes the still-unmatched ground truth
    with the highest IoU above ``iou_threshold``.

    Returns:
        matches      : list of (pred_idx, gt_idx, iou)
        unmatched_pred : prediction indices with no match (false positives)
        unmatched_gt   : ground-truth indices with no match (false negatives)
    """
    n_pred = len(iou_matrix)
    n_gt = len(iou_matrix[0]) if n_pred else 0

    order = sorted(range(n_pred), key=lambda i: float(pred_scores[i]), reverse=True)

    gt_taken = [False] * n_gt
    pred_matched = [False] * n_pred
    matches: List[Tuple[int, int, float]] = []

    for pi in order:
        best_iou = iou_threshold
        best_gt = -1
        row = iou_matrix[pi]
        for gj in range(n_gt):
            if gt_taken[gj]:
                continue
            iou = row[gj]
            if iou >= best_iou:
                best_iou = iou
                best_gt = gj
        if best_gt >= 0:
            gt_taken[best_gt] = True
            pred_matched[pi] = True
            matches.append((pi, best_gt, float(iou_matrix[pi][best_gt])))

    unmatched_pred = [i for i in range(n_pred) if not pred_matched[i]]
    unmatched_gt = [j for j in range(n_gt) if not gt_taken[j]]
    return matches, unmatched_pred, unmatched_gt
