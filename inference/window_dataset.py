from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from osgeo import gdal
from PIL import Image
from torch.utils.data import Dataset

from utils.common import GeoMeta


gdal.UseExceptions()


@dataclass
class WindowRecord:
    x: int
    y: int
    w: int
    h: int


def build_windows(
    image_width: int,
    image_height: int,
    slice_height: int,
    slice_width: int,
    overlap_height_ratio: float,
    overlap_width_ratio: float,
) -> List[WindowRecord]:
    slice_height = max(1, min(slice_height, image_height))
    slice_width = max(1, min(slice_width, image_width))

    stride_h = max(1, int(slice_height * (1.0 - overlap_height_ratio)))
    stride_w = max(1, int(slice_width * (1.0 - overlap_width_ratio)))

    y_points = list(range(0, max(1, image_height - slice_height + 1), stride_h))
    x_points = list(range(0, max(1, image_width - slice_width + 1), stride_w))

    if y_points[-1] != image_height - slice_height:
        y_points.append(image_height - slice_height)
    if x_points[-1] != image_width - slice_width:
        x_points.append(image_width - slice_width)

    y_points = sorted(set(max(0, y) for y in y_points))
    x_points = sorted(set(max(0, x) for x in x_points))

    windows: List[WindowRecord] = []
    for y in y_points:
        for x in x_points:
            w = min(slice_width, image_width - x)
            h = min(slice_height, image_height - y)
            windows.append(WindowRecord(x=x, y=y, w=w, h=h))
    return windows


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    if arr.size == 0:
        return arr.astype(np.uint8)
    arr_max = float(np.max(arr))
    arr_min = float(np.min(arr))
    if arr_max <= arr_min:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (arr - arr_min) / (arr_max - arr_min) * 255.0
    return scaled.astype(np.uint8)


def _array_to_rgb_pil(data: np.ndarray) -> Image.Image:
    if data.ndim == 2:
        data = np.stack([data, data, data], axis=-1)
    elif data.ndim == 3 and data.shape[0] in (1, 3, 4):
        data = np.transpose(data, (1, 2, 0))
    if data.ndim == 3 and data.shape[2] == 1:
        data = np.repeat(data, 3, axis=2)
    if data.ndim == 3 and data.shape[2] > 3:
        data = data[:, :, :3]
    data = _normalize_to_uint8(data)
    return Image.fromarray(data).convert("RGB")


def _load_prompt_boxes(image_path: str, prompt_source: str) -> List[List[float]]:
    if not prompt_source:
        return []

    prompt_file: Optional[str] = None
    if os.path.isfile(prompt_source) and prompt_source.lower().endswith(".txt"):
        prompt_file = prompt_source
    elif os.path.isdir(prompt_source):
        stem = Path(image_path).stem
        basename = os.path.basename(image_path)
        candidates = [
            os.path.join(prompt_source, stem + ".txt"),
            os.path.join(prompt_source, basename + ".txt"),
            os.path.join(prompt_source, basename.replace(Path(image_path).suffix, ".txt")),
        ]
        for cand in candidates:
            if os.path.exists(cand):
                prompt_file = cand
                break

    if not prompt_file or not os.path.exists(prompt_file):
        return []

    boxes: List[List[float]] = []
    with open(prompt_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 4:
                continue
            try:
                box = [float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])]
            except ValueError:
                continue
            area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
            if area < 500.0:
                continue
            boxes.append(box)
    return boxes


def _window_prompt_boxes(
    prompt_boxes: List[List[float]],
    win: WindowRecord,
    max_prompt_per_window: int,
) -> List[List[float]]:
    local_boxes: List[List[float]] = []
    x1 = win.x
    y1 = win.y
    x2 = win.x + win.w
    y2 = win.y + win.h

    for box in prompt_boxes:
        bx1, by1, bx2, by2 = box
        if bx2 < x1 or bx1 > x2 or by2 < y1 or by1 > y2:
            continue
        local = [
            max(bx1 - x1, 0.0),
            max(by1 - y1, 0.0),
            min(bx2 - x1, float(win.w)),
            min(by2 - y1, float(win.h)),
        ]
        local_boxes.append(local)

    if max_prompt_per_window > 0 and len(local_boxes) > max_prompt_per_window:
        local_boxes = random.sample(local_boxes, k=max_prompt_per_window)
    return local_boxes


class SlidingWindowDataset(Dataset):
    def __init__(
        self,
        image_path: str,
        slicing_cfg: Dict[str, Any],
        prompt_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.image_path = image_path
        self.prompt_cfg = prompt_cfg or {}
        self.enable_prompt = bool(self.prompt_cfg.get("enabled", False))
        self.max_prompt_per_window = int(self.prompt_cfg.get("max_prompt_per_window", 5))

        self._use_gdal = False
        self._projection_wkt: Optional[str] = None
        self._geotransform: Optional[List[float]] = None
        self._gdal_ds: Optional[gdal.Dataset] = None
        self._pil_img: Optional[Image.Image] = None

        image_width, image_height = self._probe_shape_and_geo()
        self.image_width = image_width
        self.image_height = image_height

        self.windows = build_windows(
            image_width=image_width,
            image_height=image_height,
            slice_height=int(slicing_cfg.get("slice_height", 1024)),
            slice_width=int(slicing_cfg.get("slice_width", 1024)),
            overlap_height_ratio=float(slicing_cfg.get("overlap_height_ratio", 0.25)),
            overlap_width_ratio=float(slicing_cfg.get("overlap_width_ratio", 0.25)),
        )

        prompt_source = str(self.prompt_cfg.get("source", ""))
        self.prompt_boxes = _load_prompt_boxes(image_path, prompt_source) if self.enable_prompt else []

    @property
    def geo_meta(self) -> GeoMeta:
        return GeoMeta(
            geotransform=self._geotransform,
            projection_wkt=self._projection_wkt,
            width=self.image_width,
            height=self.image_height,
        )

    def _probe_shape_and_geo(self) -> Tuple[int, int]:
        try:
            ds = gdal.Open(self.image_path)
        except RuntimeError:
            ds = None

        if ds is not None:
            self._use_gdal = True
            width = int(ds.RasterXSize)
            height = int(ds.RasterYSize)
            gt = ds.GetGeoTransform(can_return_null=True)
            self._geotransform = list(gt) if gt else None
            prj = ds.GetProjection()
            self._projection_wkt = prj if prj else None
            ds = None
            return width, height

        with Image.open(self.image_path) as img:
            width, height = img.size
        return int(width), int(height)

    def __len__(self) -> int:
        return len(self.windows)

    def _read_window(self, win: WindowRecord) -> Image.Image:
        if self._use_gdal:
            if self._gdal_ds is None:
                self._gdal_ds = gdal.Open(self.image_path)
            if self._gdal_ds is None:
                raise RuntimeError(f"Cannot open image with GDAL: {self.image_path}")
            arr = self._gdal_ds.ReadAsArray(win.x, win.y, win.w, win.h)
            if arr is None:
                raise RuntimeError(
                    f"Failed to read window x={win.x}, y={win.y}, w={win.w}, h={win.h}"
                )
            return _array_to_rgb_pil(arr)

        if self._pil_img is None:
            self._pil_img = Image.open(self.image_path).convert("RGB")
        return self._pil_img.crop((win.x, win.y, win.x + win.w, win.y + win.h))

    def __del__(self) -> None:
        self._gdal_ds = None
        if self._pil_img is not None:
            try:
                self._pil_img.close()
            except Exception:
                pass
            self._pil_img = None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        win = self.windows[idx]
        image = self._read_window(win)

        geometry_prompt: Optional[List[List[float]]] = None
        geometry_labels: Optional[List[bool]] = None
        if self.prompt_boxes:
            geometry_prompt = _window_prompt_boxes(
                prompt_boxes=self.prompt_boxes,
                win=win,
                max_prompt_per_window=self.max_prompt_per_window,
            )
            if geometry_prompt:
                geometry_labels = [True] * len(geometry_prompt)
            else:
                geometry_prompt = None

        return {
            "image": image,
            "offset": (win.x, win.y),
            "window_size": (win.w, win.h),
            "geometry_prompt": geometry_prompt,
            "geometry_labels": geometry_labels,
            "src": os.path.basename(self.image_path),
        }


def window_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "images": [item["image"] for item in batch],
        "offsets": [item["offset"] for item in batch],
        "window_sizes": [item["window_size"] for item in batch],
        "geometry_prompts": [item["geometry_prompt"] for item in batch],
        "geometry_labels": [item["geometry_labels"] for item in batch],
        "srcs": [item["src"] for item in batch],
    }
