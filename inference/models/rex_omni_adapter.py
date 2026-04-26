from __future__ import annotations

from typing import Any, Dict, List, Optional

from PIL import Image

from inference.models.base import DetectionModelAdapter


class RexOmniAdapter(DetectionModelAdapter):
    """Placeholder adapter for future Rex-Omni integration."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg

    def predict_batch(
        self,
        images: List[Image.Image],
        text_prompt: str,
        geometry_prompts: Optional[List[Optional[List[List[float]]]]] = None,
        geometry_labels: Optional[List[Optional[List[bool]]]] = None,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError(
            "Rex-Omni adapter is not implemented yet. "
            "Please implement model loading/inference in inference/models/rex_omni_adapter.py"
        )
