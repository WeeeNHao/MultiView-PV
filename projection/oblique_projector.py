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
        self.method = str(cfg.get("method", "auto")).strip().lower()

        if not self.pose_csv:
            raise ValueError("projection.oblique.pose_csv is required for oblique mode")
        if not self.dsm_path:
            raise ValueError("projection.oblique.dsm_path is required for oblique mode")
        if self.method not in {"auto", "affine", "collinearity"}:
            raise ValueError("projection.oblique.method must be one of: auto, affine, collinearity")
        if not os.path.exists(self.pose_csv):
            raise FileNotFoundError(f"pose_csv not found: {self.pose_csv}")
        if not os.path.exists(self.dsm_path):
            raise FileNotFoundError(f"dsm_path not found: {self.dsm_path}")

        self.pose_dict = read_pose_csv(self.pose_csv)

        self.focal = float(cfg.get("focal", 3713.29))
        self.cx = float(cfg.get("cx", 2647.02))
        self.cy = float(cfg.get("cy", 1969.28))
        self.avg_alt = float(cfg.get("avg_alt", 30.0))
        
        self.ray_dsm_max_iter = int(cfg.get("ray_dsm_max_iter", 8))
        self.ray_dsm_tol = float(cfg.get("ray_dsm_tol", 1))
        self.ray_dsm_init_window = int(cfg.get("ray_dsm_init_window", 9))
        self.ray_dsm_fallback_avg_alt = bool(cfg.get("ray_dsm_fallback_avg_alt", False))
        self.enable_alt_filter = bool(cfg.get("enable_alt_filter", False))
        
        self.sample_interval = int(cfg.get("sample_interval", 50))
        self.max_alt_diff = float(cfg.get("max_alt_diff", 3.0))
        self.search_margin_px = int(cfg.get("search_margin_px", 200))
        self.min_control_points = int(cfg.get("min_control_points", 5))

        self.image_width = int(cfg.get("image_width", 5280))
        self.image_height = int(cfg.get("image_height", 3956))

        self.enable_slope_correction = bool(cfg.get("enable_slope_correction", False))

        self._dsm_ds = gdal.Open(self.dsm_path)
        self._dsm_ds_nodata = self._dsm_ds.GetRasterBand(1).GetNoDataValue() if self._dsm_ds is not None else None
        if self._dsm_ds is None:
            raise RuntimeError(f"Failed to open DSM: {self.dsm_path}")

        self._dsm_w = int(self._dsm_ds.RasterXSize)
        self._dsm_h = int(self._dsm_ds.RasterYSize)
        self._dsm_geo = self._dsm_ds.GetGeoTransform()

        preload = bool(cfg.get("preload_dsm", False))
        self._dsm_array = self._dsm_ds.ReadAsArray() if preload else None
        self._dsm_global_median = self._compute_dsm_center_median()
        # print(f"global DSM median: {self._dsm_global_median}")
        self._last_dsm_z: Optional[float] = None

    def _read_dsm_value(self, col: int, row: int) -> Optional[float]:
        if col < 0 or row < 0 or col >= self._dsm_w or row >= self._dsm_h:
            return None
        if self._dsm_array is not None:
            return float(self._dsm_array[row, col])

        arr = self._dsm_ds.ReadAsArray(col, row, 1, 1) # type: ignore
        if arr is None:
            return None
        return float(arr[0][0])

    def _resolve_pose(self, image_path: str) -> Optional[List[str]]:
        for name in _image_name_candidates(image_path):
            if name in self.pose_dict:
                return self.pose_dict[name]
        return None

    def _compute_dsm_center_median(self) -> Optional[float]:
        col = int(self._dsm_w // 2)
        row = int(self._dsm_h // 2)
        return self._dsm_window_median(col, row, self.ray_dsm_init_window)

    def _dsm_window_median(self, col: int, row: int, window: int) -> Optional[float]:
        if window <= 1:
            value = self._read_dsm_value(col, row)
            return float(value) if value is not None else None

        half = window // 2
        c0 = max(0, col - half)
        r0 = max(0, row - half)
        c1 = min(self._dsm_w, col + half + 1)
        r1 = min(self._dsm_h, row + half + 1)
        if c0 >= c1 or r0 >= r1:
            return None

        if self._dsm_array is not None:
            block = self._dsm_array[r0:r1, c0:c1]
        else:
            block = self._dsm_ds.ReadAsArray(c0, r0, c1 - c0, r1 - r0) # type: ignore
        if block is None:
            return None

        vals = block.astype(np.float64).ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return None
        return float(np.median(vals))

    def _estimate_initial_z(
        self,
        img_x: float,
        img_y: float,
        pose: Sequence[float],
    ) -> Optional[float]:
        z0 = self._last_dsm_z
        if z0 is None:
            z0 = self._dsm_global_median
        if z0 is None:
            return None

        xs, ys, zs, phi, omega, kappa = pose
        rot = build_rotation(phi, omega, kappa)
        px = float(img_x) - self.cx
        py = self.cy - float(img_y)
        gx, gy = photo_to_ground(px, py, self.focal, float(z0), xs, ys, zs, rot)
        col_f, row_f = geo_to_image_xy(self._dsm_geo, gx, gy)

        col = int(round(col_f))
        row = int(round(row_f))
        local = self._dsm_window_median(col, row, self.ray_dsm_init_window)
        return local if local is not None else float(z0)

    def _ray_dsm_intersection(
        self,
        img_x: float,
        img_y: float,
        pose: Sequence[float],
    ) -> Optional[Tuple[float, float, float]]:
        xs, ys, zs, phi, omega, kappa = pose
        rot = build_rotation(phi, omega, kappa)

        px = float(img_x) - self.cx
        py = self.cy - float(img_y)

        z0 = self._estimate_initial_z(img_x, img_y, pose)
        if z0 is None:
            return None
        z = float(z0)
        for _ in range(max(1, self.ray_dsm_max_iter)):
            gx, gy = photo_to_ground(px, py, self.focal, z, xs, ys, zs, rot)
            col_f, row_f = geo_to_image_xy(self._dsm_geo, gx, gy)
            col = int(round(col_f))
            row = int(round(row_f))
            dsm_z = self._read_dsm_value(col, row)
            if dsm_z is None or not np.isfinite(dsm_z):
                return None
            if abs(dsm_z - z) <= self.ray_dsm_tol:
                self._last_dsm_z = float(dsm_z)
                return gx, gy, float(dsm_z)
            z = float(dsm_z)

        return None

    def _project_point_fallback_avg_alt(
        self,
        img_x: float,
        img_y: float,
        pose: Sequence[float],
    ) -> Tuple[float, float]:
        xs, ys, zs, phi, omega, kappa = pose
        rot = build_rotation(phi, omega, kappa)
        px = float(img_x) - self.cx
        py = self.cy - float(img_y)
        return photo_to_ground(px, py, self.focal, self.avg_alt, xs, ys, zs, rot)

    def _project_points_direct_collinearity(
        self,
        points_xy: List[Tuple[float, float]],
        pose: Sequence[float],
    ) -> List[Tuple[float, float]]:
        mapped: List[Tuple[float, float]] = []
        for x, y in points_xy:
            hit = self._ray_dsm_intersection(x, y, pose)
            if hit is not None:
                mapped.append((hit[0], hit[1]))
                continue
            if self.ray_dsm_fallback_avg_alt and self.avg_alt is not None:
                mapped.append(self._project_point_fallback_avg_alt(x, y, pose))
        return mapped

    def _build_affine_pairs(
        self,
        bbox: Sequence[float],
        pose: Sequence[float],
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        margin = float(self.search_margin_px)

        corners = [
            (x1 - margin, y1 - margin),
            (x1 - margin, y2 + margin),
            (x2 + margin, y2 + margin),
            (x2 + margin, y1 - margin),
        ]
        corner_hits: List[Tuple[float, float]] = []
        for cx, cy in corners:
            hit = self._ray_dsm_intersection(cx, cy, pose)
            if hit is not None:
                corner_hits.append((hit[0], hit[1]))
            elif self.ray_dsm_fallback_avg_alt and self.avg_alt is not None:
                corner_hits.append(self._project_point_fallback_avg_alt(cx, cy, pose))

        if not corner_hits:
            return [], []

        gx_min = float(min(p[0] for p in corner_hits))
        gx_max = float(max(p[0] for p in corner_hits))
        gy_min = float(min(p[1] for p in corner_hits))
        gy_max = float(max(p[1] for p in corner_hits))

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
        if self.enable_alt_filter:
            alt_mask = np.abs(z_vals - self._dsm_global_median) <= self.max_alt_diff
            cc_valid = cc_flat[alt_mask]
            rr_valid = rr_flat[alt_mask]
            z_valid = z_vals[alt_mask]
        else:
            cc_valid = cc_flat
            rr_valid = rr_flat
            z_valid = z_vals

        if len(z_valid) == 0:
            return [], []

        # Absolute DSM col/row
        abs_cols = (cc_valid + col_min).astype(np.float64)
        abs_rows = (rr_valid + row_min).astype(np.float64)

        # Vectorized imagexy2geo
        gx = gt[0] + abs_cols * gt[1] + abs_rows * gt[2]
        gy = gt[3] + abs_cols * gt[4] + abs_rows * gt[5]

        xs, ys, zs, phi, omega, kappa = pose
        rot = build_rotation(phi, omega, kappa)
        a1, a2, a3, b1, b2, b3, c1, c2, c3 = rot

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
        if self.method == "affine":
            if len(src_pts) < 3:
                # raise ValueError("projection.oblique.method=affine requires at least 3 valid control points")
                out["projection_method"] = "affine_failed"
                return out
            use_affine = True
        elif self.method == "collinearity":
            use_affine = False
        else:
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
