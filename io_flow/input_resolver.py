from __future__ import annotations

import glob
import os
from typing import Any, Dict, List


def _read_list_file(list_file: str) -> List[str]:
    items: List[str] = []
    with open(list_file, "r", encoding="utf-8") as f:
        for line in f:
            path = line.strip()
            if not path or path.startswith("#"):
                continue
            items.append(path)
    return items


def _resolve_with_root(path: str, root: str) -> str:
    if os.path.isabs(path):
        return path
    if root:
        return os.path.join(root, path)
    return path


def resolve_image_paths(data_cfg: Dict[str, Any]) -> List[str]:
    data_root = str(data_cfg.get("data_root", ""))

    images: List[str] = []
    image_path = data_cfg.get("image_path")
    if isinstance(image_path, str) and image_path:
        images.append(_resolve_with_root(image_path, data_root))
    elif isinstance(image_path, list):
        for path in image_path:
            if not path:
                continue
            images.append(_resolve_with_root(str(path), data_root))

    image_glob = str(data_cfg.get("image_glob", "")).strip()
    if image_glob:
        pattern = _resolve_with_root(image_glob, data_root)
        images.extend(glob.glob(pattern))

    image_list_file = str(data_cfg.get("image_list_file", "")).strip()
    if image_list_file:
        list_file = _resolve_with_root(image_list_file, data_root)
        images.extend(_read_list_file(list_file))

    # de-duplicate while keeping order
    uniq: List[str] = []
    seen = set()
    for path in images:
        norm = os.path.abspath(path)
        if norm in seen:
            continue
        seen.add(norm)
        uniq.append(norm)

    missing = [p for p in uniq if not os.path.exists(p)]
    if missing:
        miss_preview = "\n".join(missing[:10])
        raise FileNotFoundError(
            f"Found non-existing image paths (showing up to 10):\n{miss_preview}"
        )

    if not uniq:
        raise ValueError(
            "No images resolved from config. Please set one of: "
            "data.image_path, data.image_glob, data.image_list_file"
        )

    return uniq