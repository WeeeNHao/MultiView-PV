from __future__ import annotations

from typing import Any, Dict

from refactor_v2.inference.models.base import DetectionModelAdapter
from refactor_v2.inference.models.rex_omni_adapter import RexOmniAdapter
from refactor_v2.inference.models.sam3_adapter import SAM3Adapter


_MODEL_REGISTRY = {
    "sam3": SAM3Adapter,
    "rex-omni": RexOmniAdapter,
}


def build_model(model_cfg: Dict[str, Any]) -> DetectionModelAdapter:
    model_name = str(model_cfg.get("name", "sam3")).lower()
    if model_name not in _MODEL_REGISTRY:
        support = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise ValueError(f"Unsupported model '{model_name}'. Supported: {support}")
    model_cls = _MODEL_REGISTRY[model_name]
    return model_cls(model_cfg)
