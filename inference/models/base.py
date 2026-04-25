from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from PIL import Image


class DetectionModelAdapter(ABC):
    @abstractmethod
    def predict_batch(
        self,
        images: List[Image.Image],
        text_prompt: str,
        geometry_prompts: Optional[List[Optional[List[List[float]]]]] = None,
        geometry_labels: Optional[List[Optional[List[bool]]]] = None,
    ) -> List[Dict[str, Any]]:
        """Return per-image detection dict with keys: boxes, masks, scores, labels."""

    def close(self) -> None:
        """Optional resource release hook."""
        return
