"""Vector-result evaluation package.

Computes instance-level detection/segmentation metrics between two sets of
polygon features (typically a prediction shapefile and a ground-truth
shapefile) produced by the MultiView-PV pipeline:

    - IoU      : per-match IoU plus aggregate (micro/macro) IoU
    - F1       : precision / recall / F1 at one or more IoU thresholds
    - MSE      : area error (predicted total / per-instance area vs ground truth)
    - mAP      : COCO-style mean average precision over IoU thresholds

All geometry math is done with OGR, matching ``postprocess.nms`` conventions.
"""

from .metrics import (
    EvalResult,
    evaluate_feature_lists,
    evaluate_shapefiles,
)

__all__ = [
    "EvalResult",
    "evaluate_feature_lists",
    "evaluate_shapefiles",
]
