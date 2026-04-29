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
        a1, a2, a3, b1, b2, b3, c1, c2, c3 = rot

        pts = np.array(points_xy, dtype=np.float64)
        px = pts[:, 0] - self.cx
        py = self.cy - pts[:, 1]

        den = c1 * px + c2 * py - c3 * self.focal
        den = np.where(np.abs(den) < 1e-12, 1e-12, den)
        gx = (self.avg_alt - zs) * (a1 * px + a2 * py - a3 * self.focal) / den + xs
        gy = (self.avg_alt - zs) * (b1 * px + b2 * py - b3 * self.focal) / den + ys

        return list(zip(gx.tolist(), gy.tolist()))

    def _build_affine_pairs(
        self,
        bbox: Sequence[float],
        pose: Sequence[float],
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        xs, ys, zs, phi, omega, kappa = pose
        rot = build_rotation(phi, omega, kappa)

        x1, y1, x2, y2 = [float(v) for v in bbox]
        margin = float(self.search_margin_px)

        # Vectorized corner projection
        corner_px = np.array([x1 - margin, x1 - margin, x2 + margin, x2 + margin]) - self.cx
        corner_py = self.cy - np.array([y1 - margin, y2 + margin, y2 + margin, y1 - margin])
        a1, a2, a3, b1, b2, b3, c1, c2, c3 = rot
        den = c1 * corner_px + c2 * corner_py - c3 * self.focal
        den = np.where(np.abs(den) < 1e-12, 1e-12, den)
        corner_gx = (self.avg_alt - zs) * (a1 * corner_px + a2 * corner_py - a3 * self.focal) / den + xs
        corner_gy = (self.avg_alt - zs) * (b1 * corner_px + b2 * corner_py - b3 * self.focal) / den + ys

        gx_min, gx_max = float(corner_gx.min()), float(corner_gx.max())
        gy_min, gy_max = float(corner_gy.min()), float(corner_gy.max())

        gt = self._dsm_geo
        c0, r0 = geo_to_image_xy(gt, gx_min, gy_max)
        c1r, r1r = geo_to_image_xy(gt, gx_max, gy_min)

        col_min = max(0, min(int(min(c0, c1r)), self._dsm_w - 1))
        col_max = max(0, min(int(max(c0, c1r)), self._dsm_w - 1))
        row_min = max(0, min(int(min(r0, r1r)), self._dsm_h - 1))
        row_max = max(0, min(int(max(r0, r1r)), self._dsm_h - 1))

        if col_min >= col_max or row_min >= row_max:
            return [], []

        step = max(1, self.sample_interval)

        # Read DSM block at once
        block_w = col_max - col_min + 1
        block_h = row_max - row_min + 1
        if self._dsm_array is not None:
            dsm_block = self._dsm_array[row_min:row_min + block_h, col_min:col_min + block_w]
        else:
            dsm_block = self._dsm_ds.ReadAsArray(col_min, row_min, block_w, block_h)
        if dsm_block is None:
            return [], []

        # Build sampled grid indices (local to block)
        local_cols = np.arange(0, block_w, step)
        local_rows = np.arange(0, block_h, step)
        cc, rr = np.meshgrid(local_cols, local_rows)
        cc_flat = cc.ravel()
        rr_flat = rr.ravel()

        # Read Z values with numpy indexing
        z_vals = dsm_block[rr_flat, cc_flat].astype(np.float64)

        # Filter by altitude
        alt_mask = np.abs(z_vals - self.avg_alt) <= self.max_alt_diff
        cc_valid = cc_flat[alt_mask]
        rr_valid = rr_flat[alt_mask]
        z_valid = z_vals[alt_mask]

        if len(z_valid) == 0:
            return [], []

        # Absolute DSM col/row
        abs_cols = (cc_valid + col_min).astype(np.float64)
        abs_rows = (rr_valid + row_min).astype(np.float64)

        # Vectorized imagexy2geo
        gx = gt[0] + abs_cols * gt[1] + abs_rows * gt[2]
        gy = gt[3] + abs_cols * gt[4] + abs_rows * gt[5]

        # Vectorized ground_to_photo
        dx = gx - xs
        dy = gy - ys
        dz = z_valid - zs
        den2 = a3 * dx + b3 * dy + c3 * dz
        den2 = np.where(np.abs(den2) < 1e-12, 1e-12, den2)
        px = -self.focal * (a1 * dx + b1 * dy + c1 * dz) / den2
        py = -self.focal * (a2 * dx + b2 * dy + c2 * dz) / den2

        # Photo coords to image coords
        img_x = px + self.cx
        img_y = self.cy - py

        # Filter: in image bounds and in bbox
        valid = (
            (img_x >= 0) & (img_y >= 0)
            & (img_x <= self.image_width) & (img_y <= self.image_height)
            & (img_x >= x1) & (img_x <= x2)
            & (img_y >= y1) & (img_y <= y2)
        )

        img_x = img_x[valid]
        img_y = img_y[valid]
        gx = gx[valid]
        gy = gy[valid]

        src_pts = list(zip(img_x.tolist(), img_y.tolist()))
        dst_pts = list(zip(gx.tolist(), gy.tolist()))

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
