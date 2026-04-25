from __future__ import annotations

from typing import List

import cv2
import numpy as np


def mask_to_polygon(
    mask: np.ndarray,
    threshold: float = 0.5,
) -> List[List[float]]:
    if mask.ndim > 2:
        mask = mask.squeeze()

    mask_uint8 = (mask > threshold).astype(np.uint8) * 255
    outs = cv2.findContours(mask_uint8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    contours = outs[-2]
    hierarchy = outs[-1]
    if hierarchy is None or len(contours) == 0:
        return []

    poly = contours[0].flatten().tolist()
    if len(poly) < 6:
        return []
    return [poly]
