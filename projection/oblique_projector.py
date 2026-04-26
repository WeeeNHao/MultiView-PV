from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from osgeo import gdal

from utils.common import Feature
from projection.collinearity import (
    build_rotation,
    compute_affine_transform,
    geo_to_image_xy,
    ground_to_photo,
    image_xy_to_geo,
    photo_to_ground,
    read_pose_csv,
)


def _flat_to_pairs(flat_xy: Sequence[float]) -> List[Tuple[float, float]]:
    if len(flat_xy) < 6 or len(flat_xy) % 2 != 0:
        return []
    return [(float(flat_xy[i]), float(flat_xy[i + 1])) for i in range(0, len(flat_xy), 2)]


def _pairs_to_flat(points: Sequence[Tuple[float, float]]) -> List[float]:
    out: List[float] = []
    for x, y in points:
        out.append(float(x))
        out.append(float(y))
    return out


def _bbox_from_segmentation(segmentation: List[List[float]]) -> List[float]:
    xs: List[float] = []
    ys: List[float] = []
    for ring in segmentation:
        pts = _flat_to_pairs(ring)
        for x, y in pts:
            xs.append(x)
            ys.append(y)
    if not xs or not ys:
        return [0.0, 0.0, 0.0, 0.0]
    return [min(xs), min(ys), max(xs), max(ys)]


def _image_name_candidates(image_path: str) -> List[str]:
    p = Path(image_path)
    return [p.name, p.stem + ".JPG", p.stem + ".jpg", p.stem + ".png", p.stem + ".tif", p.stem + ".TIF"]


class ObliqueProjector:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.pose_csv = str(cfg.get("pose_csv", "")).strip()
        self.dsm_path = str(cfg.get("dsm_path", "")).strip()

        if not self.pose_csv:
            raise ValueError("projection.oblique.pose_csv is required for oblique mode")
        if not self.dsm_path:
            raise ValueError("projection.oblique.dsm_path is required for oblique mode")
        if not os.path.exists(self.pose_csv):
            raise FileNotFoundError(f"pose_csv not found: {self.pose_csv}")
        if not os.path.exists(self.dsm_path):
            raise FileNotFoundError(f"dsm_path not found: {self.dsm_path}")

        self.pose_dict = read_pose_csv(self.pose_csv)

        self.focal = float(cfg.get("focal", 3713.29))
        self.cx = float(cfg.get("cx", 2647.02))
        self.cy = float(cfg.get("cy", 1969.28))
        self.avg_alt = float(cfg.get("avg_alt", 30.8))

        self.sample_interval = int(cfg.get("sample_interval", 50))
        self.max_alt_diff = float(cfg.get("max_alt_diff", 3.0))
        self.search_margin_px = int(cfg.get("search_margin_px", 200))
        self.min_control_points = int(cfg.get("min_control_points", 5))

        self.image_width = int(cfg.get("image_width", 5280))
        self.image_height = int(cfg.get("image_height", 3956))

        self.enable_slope_correction = bool(cfg.get("enable_slope_correction", False))

        self._dsm_ds = gdal.Open(self.dsm_path)
        if self._dsm_ds is None:
            raise RuntimeError(f"Failed to open DSM: {self.dsm_path}")

        self._dsm_w = int(self._dsm_ds.RasterXSize)
        self._dsm_h = int(self._dsm_ds.RasterYSize)
        self._dsm_geo = self._dsm_ds.GetGeoTransform()

        preload = bool(cfg.get("preload_dsm", False))
        self._dsm_array = self._dsm_ds.ReadAsArray() if preload else None

    def _read_dsm_value(self, col: int, row: int) -> Optional[float]:
        if col < 0 or row < 0 or col >= self._dsm_w or row >= self._dsm_h:
            return None
        if self._dsm_array is not None:
            return float(self._dsm_array[row, col])

        arr = self._dsm_ds.ReadAsArray(col, row, 1, 1)
        if arr is None:
            return None
        return float(arr[0][0])

    def _resolve_pose(self, image_path: str) -> Optional[List[str]]:
        for name in _image_name_candidates(image_path):
            if name in self.pose_dict:
                return self.pose_dict[name]
        return None

    def _project_points_direct_collinearity(
        self,
        points_xy: List[Tuple[float, float]],
        pose: Sequence[float],
    ) -> List[Tuple[float, float]]:
        xs, ys, zs, phi, omega, kappa = pose
        rot = build_rotation(phi, omega, kappa)

        out: List[Tuple[float, float]] = []
        for x, y in points_xy:
            px = x - self.cx
            py = self.cy - y
            gx, gy = photo_to_ground(px, py, self.focal, self.avg_alt, xs, ys, zs, rot)
            out.append((gx, gy))
        return out

    def _build_affine_pairs(
        self,
        bbox: Sequence[float],
        pose: Sequence[float],
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        xs, ys, zs, phi, omega, kappa = pose
        rot = build_rotation(phi, omega, kappa)

        x1, y1, x2, y2 = [float(v) for v in bbox]
        margin = float(self.search_margin_px)

        corners = [
            (x1 - margin, y1 - margin),
            (x1 - margin, y2 + margin),
            (x2 + margin, y2 + margin),
            (x2 + margin, y1 - margin),
        ]

        approx_ground: List[Tuple[float, float]] = []
        for ix, iy in corners:
            px = ix - self.cx
            py = self.cy - iy
            gx, gy = photo_to_ground(px, py, self.focal, self.avg_alt, xs, ys, zs, rot)
            approx_ground.append((gx, gy))

        gx_vals = [p[0] for p in approx_ground]
        gy_vals = [p[1] for p in approx_ground]
        gx_min, gx_max = min(gx_vals), max(gx_vals)
        gy_min, gy_max = min(gy_vals), max(gy_vals)

        c0, r0 = geo_to_image_xy(self._dsm_geo, gx_min, gy_max)
        c1, r1 = geo_to_image_xy(self._dsm_geo, gx_max, gy_min)

        col_min = max(0, min(int(min(c0, c1)), self._dsm_w - 1))
        col_max = max(0, min(int(max(c0, c1)), self._dsm_w - 1))
        row_min = max(0, min(int(min(r0, r1)), self._dsm_h - 1))
        row_max = max(0, min(int(max(r0, r1)), self._dsm_h - 1))

        if col_min >= col_max or row_min >= row_max:
            return [], []

        src_pts: List[Tuple[float, float]] = []
        dst_pts: List[Tuple[float, float]] = []

        for col in range(col_min, col_max + 1, max(1, self.sample_interval)):
            for row in range(row_min, row_max + 1, max(1, self.sample_interval)):
                z = self._read_dsm_value(col, row)
                if z is None:
                    continue
                if abs(z - self.avg_alt) > self.max_alt_diff:
                    continue

                gx, gy = image_xy_to_geo(self._dsm_geo, col, row)
                px, py = ground_to_photo(self.focal, gx, gy, z, xs, ys, zs, rot)
                ix = px + self.cx
                iy = self.cy - py

                if ix < 0 or iy < 0 or ix > self.image_width or iy > self.image_height:
                    continue
                if not (x1 <= ix <= x2 and y1 <= iy <= y2):
                    continue

                src_pts.append((ix, iy))
                dst_pts.append((gx, gy))

        return src_pts, dst_pts

    def _apply_slope_correction_placeholder(self, mapped: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        # Placeholder hook for future plane/slope correction.
        return mapped

    def project_feature(self, feature: Feature, image_path: str) -> Feature:
        out = dict(feature)
        seg = out.get("segmentation", [])
        if not isinstance(seg, list) or not seg:
            return out

        pose_raw = self._resolve_pose(image_path)
        if pose_raw is None:
            return out

        pose = [float(v) for v in pose_raw[:6]]
        bbox = out.get("bbox", [0.0, 0.0, 0.0, 0.0])

        src_pts, dst_pts = self._build_affine_pairs(bbox=bbox, pose=pose)
        use_affine = len(src_pts) >= self.min_control_points

        transformed_seg: List[List[float]] = []
        if use_affine:
            mat2, vec, _ = compute_affine_transform(src_pts, dst_pts)
            for ring in seg:
                points = _flat_to_pairs(ring)
                if len(points) < 3:
                    continue
                mapped = [(mat2[0, 0] * x + mat2[0, 1] * y + vec[0], mat2[1, 0] * x + mat2[1, 1] * y + vec[1]) for x, y in points]
                if self.enable_slope_correction:
                    mapped = self._apply_slope_correction_placeholder(mapped)
                transformed_seg.append(_pairs_to_flat(mapped))
        else:
            for ring in seg:
                points = _flat_to_pairs(ring)
                if len(points) < 3:
                    continue
                mapped = self._project_points_direct_collinearity(points, pose=pose)
                if self.enable_slope_correction:
                    mapped = self._apply_slope_correction_placeholder(mapped)
                transformed_seg.append(_pairs_to_flat(mapped))

        if transformed_seg:
            out["segmentation"] = transformed_seg
            out["bbox"] = _bbox_from_segmentation(transformed_seg)
            out["projection_method"] = "affine" if use_affine else "collinearity"

        return out

    def close(self) -> None:
        self._dsm_array = None
        self._dsm_ds = None
