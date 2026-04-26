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

        outputs: List[Dict[str, Any]] = []
        for idx, img in enumerate(images):
            prompts = None
            labels = None
            if geometry_prompts and idx < len(geometry_prompts):
                prompts = geometry_prompts[idx]
            if geometry_labels and idx < len(geometry_labels):
                labels = geometry_labels[idx]
            if prompts and not labels:
                labels = [True] * len(prompts)
            one = self.segmenter.segment(
                [img],
                text_prompt=text_prompt,
                geometry_prompt=prompts,
                geometry_labels=labels,
            )
            if one:
                outputs.append(one[0])
            else:
                outputs.append({})
        return outputs
