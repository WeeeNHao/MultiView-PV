from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.common import Feature, FeatureList, GeoMeta
from inference.models.base import DetectionModelAdapter
from inference.window_dataset import SlidingWindowDataset, window_collate


def mask_to_polygon(mask: np.ndarray, threshold: float = 0.5) -> List[List[float]]:
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


def _ring_area(flat_xy: Sequence[float]) -> float:
    if len(flat_xy) < 6 or len(flat_xy) % 2 != 0:
        return 0.0
    pts = [(float(flat_xy[i]), float(flat_xy[i + 1])) for i in range(0, len(flat_xy), 2)]
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    area = 0.0
    for i in range(len(pts) - 1):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def _shift_segmentation(segmentation: List[List[float]], offset: Tuple[int, int]) -> List[List[float]]:
    ox, oy = offset
    shifted: List[List[float]] = []
    for ring in segmentation:
        if len(ring) < 6:
            continue
        out: List[float] = []
        for i in range(0, len(ring), 2):
            out.append(float(ring[i]) + ox)
            out.append(float(ring[i + 1]) + oy)
        shifted.append(out)
    return shifted


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.asarray(x)


def _convert_prediction_to_features(
    prediction: Dict[str, Any],
    offset: Tuple[int, int],
    src: str,
    early_filter_cfg: Dict[str, Any],
) -> FeatureList:
    if not prediction or "boxes" not in prediction:
        return []

    boxes = _to_numpy(prediction["boxes"])
    masks = _to_numpy(prediction["masks"])
    scores = _to_numpy(prediction["scores"])
    labels = _to_numpy(prediction["labels"])

    if masks.ndim == 4:
        masks = masks[:, 0]

    min_score = float(early_filter_cfg.get("min_score", 0.0))
    min_area_px = float(early_filter_cfg.get("min_area_px", 0.0))

    features: FeatureList = []
    ox, oy = offset
    for i in range(len(boxes)):
        score = float(scores[i])
        if score < min_score:
            continue

        segmentation = mask_to_polygon(masks[i])
        if not segmentation:
            continue

        shifted_seg = _shift_segmentation(segmentation, offset)
        if not shifted_seg:
            continue

        area_px = max(_ring_area(shifted_seg[0]), 0.0)
        if area_px < min_area_px:
            continue

        box = boxes[i].tolist()
        bbox = [float(box[0]) + ox, float(box[1]) + oy, float(box[2]) + ox, float(box[3]) + oy]

        feature: Feature = {
            "bbox": bbox,
            "segmentation": shifted_seg,
            "label": int(labels[i]),
            "src": src,
            "con_sem": score,
            "con_pv": 0.0,
            "con_weight": score,
            "score": score,
            "pixel_area": area_px,
        }
        features.append(feature)

    return features


class InferenceRunner:
    def __init__(
        self,
        model: DetectionModelAdapter,
        inference_cfg: Dict[str, Any],
    ) -> None:
        self.model = model
        self.inference_cfg = inference_cfg
        self.text_prompt = str(inference_cfg.get("text_prompt", "a PV Panel"))

        dataloader_cfg = inference_cfg.get("dataloader", {})
        self.batch_size = int(dataloader_cfg.get("batch_size", 4))
        self.num_workers = int(dataloader_cfg.get("num_workers", 0))
        self.pin_memory = bool(dataloader_cfg.get("pin_memory", False))
        self.prefetch_factor = int(dataloader_cfg.get("prefetch_factor", 2))
        self.persistent_workers = bool(dataloader_cfg.get("persistent_workers", False))

        self.prompt_cfg = inference_cfg.get("prompt", {})
        self.slicing_cfg = inference_cfg.get("slicing", {})
        self.early_filter_cfg = inference_cfg.get("early_filter", {})
        self.prompt_enabled = bool(self.prompt_cfg.get("enabled", False))
        self.strict_window_prompt = bool(self.prompt_cfg.get("strict_window_prompt", False))

    def run_image(
        self,
        image_path: str,
        progress_position: int = 1,
    ) -> Tuple[FeatureList, GeoMeta]:
        dataset = SlidingWindowDataset(
            image_path=image_path,
            slicing_cfg=self.slicing_cfg,
            prompt_cfg=self.prompt_cfg,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            collate_fn=window_collate,
        )

        all_features: FeatureList = []
        for batch in tqdm(
            dataloader,
            desc="Inferring sliding windows",
            leave=False,
            position=progress_position,
        ):
            images = batch["images"]
            offsets = batch["offsets"]
            prompts = batch["geometry_prompts"]
            labels = batch["geometry_labels"]
            srcs = batch["srcs"]

            if self.prompt_enabled and self.strict_window_prompt:
                keep_indices = [i for i, prompt in enumerate(prompts) if prompt]
                if not keep_indices:
                    continue
                images = [images[i] for i in keep_indices]
                offsets = [offsets[i] for i in keep_indices]
                prompts = [prompts[i] for i in keep_indices]
                labels = [labels[i] for i in keep_indices]
                srcs = [srcs[i] for i in keep_indices]

            results = self.model.predict_batch(
                images=images,
                text_prompt=self.text_prompt,
                geometry_prompts=prompts,
                geometry_labels=labels,
            )
            total = min(len(offsets), len(srcs))
            for i in range(total):
                pred = results[i] if i < len(results) else {}
                window_features = _convert_prediction_to_features(
                    prediction=pred,
                    offset=offsets[i],
                    src=srcs[i],
                    early_filter_cfg=self.early_filter_cfg,
                )
                all_features.extend(window_features)

        return all_features, dataset.geo_meta


if __name__ == "__main__":
    pass