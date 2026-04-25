from __future__ import annotations

import os

from omegaconf import OmegaConf


def load_config(config_path: str):
    """Recursively load YAML config with optional _base_ inheritance."""
    abs_path = os.path.abspath(config_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Config file not found: {abs_path}")

    current_cfg = OmegaConf.load(abs_path)

    if "_base_" in current_cfg:
        base_rel = current_cfg._base_
        if isinstance(base_rel, str):
            base_rel = [base_rel]

        base_cfg = OmegaConf.create()
        for base_path in base_rel:
            base_abs = os.path.join(os.path.dirname(abs_path), base_path)
            loaded_base_cfg = load_config(base_abs)
            base_cfg = OmegaConf.merge(base_cfg, loaded_base_cfg)

        merged_cfg = OmegaConf.merge(base_cfg, current_cfg)
        del merged_cfg["_base_"]
        return merged_cfg

    return current_cfg
