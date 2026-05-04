from __future__ import annotations

from typing import Any, Dict, List, Optional

from PIL import Image

from inference.models.base import DetectionModelAdapter
from inference.models.sam3_segmenter import SAM3Segmenter


class SAM3Adapter(DetectionModelAdapter):
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.segmenter = SAM3Segmenter(
            bpe_path=cfg.get("bpe_path"),
            checkpoint_path=cfg.get("checkpoint_path"),
            detection_threshold=float(cfg.get("detection_threshold", 0.5)),
            device=cfg.get("device", "cuda"),
        )

    def predict_batch(
        self,
        images: List[Image.Image],
        text_prompt: str,
        geometry_prompts: Optional[List[Optional[List[List[float]]]]] = None,
        geometry_labels: Optional[List[Optional[List[bool]]]] = None,
    ) -> List[Dict[str, Any]]:
        if not images:
            return []

        has_prompt = bool(geometry_prompts) and any(p for p in geometry_prompts)
        if not has_prompt:
            return self.segmenter.segment(images, text_prompt=text_prompt)
        return self.segmenter.segment(
            images,
            text_prompt=text_prompt,
            geometry_prompt=geometry_prompts,
            geometry_labels=geometry_labels,
        )
