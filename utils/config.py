from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict

from omegaconf import OmegaConf

from utils.config_loader import load_config


@dataclass
class RuntimeConfig:
    raw: Dict[str, Any]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)


def _to_plain_dict(cfg: Any) -> Dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refactored PV pipeline")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config. Supports _base_ inheritance.",
    )
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="Optional dotlist overrides, for example: inference.batch_size=8",
    )
    return parser.parse_args()


def load_runtime_config(config_path: str, opts: Any = None) -> RuntimeConfig:
    base_cfg = load_config(config_path)
    if opts:
        cli_cfg = OmegaConf.from_dotlist(opts)
        merged = OmegaConf.merge(base_cfg, cli_cfg)
    else:
        merged = base_cfg
    plain = _to_plain_dict(merged)
    return RuntimeConfig(raw=plain)
