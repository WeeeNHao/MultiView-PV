from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from osgeo import gdal
from sklearn.linear_model import RANSACRegressor

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
        if self.method not in {"auto", "affine", "collinearity", "slope_correction"}:
            raise ValueError("projection.oblique.method must be one of: auto, affine, collinearity, slope_correction")
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

        # Nearest-hit ray march. The search band spans the DSM's own elevation
        # range (plus a margin), so it carries no assumption about how many
        # height regimes the site has -- 004-CangFang has PV on two surfaces
        # ~9 m apart and the earlier band, anchored on a single global scalar,
        # could only reach one of them.
        self.ray_dsm_march_step = float(cfg.get("ray_dsm_march_step", 0.25))
        self.ray_dsm_z_margin = float(cfg.get("ray_dsm_z_margin", 2.0))
        self.ray_dsm_max_steps = int(cfg.get("ray_dsm_max_steps", 4096))
        # Deprecated: kept so existing configs load, no longer bounds the march.
        self.ray_dsm_march_up = float(cfg.get("ray_dsm_march_up", 4.0))
        self.ray_dsm_march_down = float(cfg.get("ray_dsm_march_down", 4.0))
        
        self.sample_interval = int(cfg.get("sample_interval", 50))
        self.max_alt_diff = float(cfg.get("max_alt_diff", 1.0))
        self.search_margin_px = int(cfg.get("search_margin_px", 200))
        self.min_control_points = int(cfg.get("min_control_points", 5))

        self.image_width = int(cfg.get("image_width", 5280))
        self.image_height = int(cfg.get("image_height", 3956))

        self.enable_slope_correction = bool(cfg.get("enable_slope_correction", False))

        sc_cfg = cfg.get("slope_correction", {}) or {}
        self.sc_ransac_threshold = float(sc_cfg.get("ransac_inlier_threshold", 0.3))
        self.sc_min_inliers = int(sc_cfg.get("min_inliers", 5))
        self.sc_max_tilt_deg = float(sc_cfg.get("max_tilt_deg", 60.0))
        self.sc_ransac_max_trials = int(sc_cfg.get("ransac_max_trials", 100))
        # RANSAC draws its candidate triplets at random. Left unseeded it used
        # the global RNG, so an ambiguous cloud gave a different plane on every
        # call -- the same feature's tilt varied by ~5 deg between two calls in
        # one process, which is what made the diagnostic's fit and
        # project_feature's fit disagree by 4-5 m.
        self.sc_ransac_random_state = int(sc_cfg.get("ransac_random_state", 0))
        # How many competing surfaces to peel off before giving up on finding
        # the one the feature actually rests on.
        self.sc_anchor_max_candidates = int(sc_cfg.get("anchor_max_candidates", 4))
        # Only sample cells this close to the feature's own ground point are
        # fitted. A panel is ~2.2 m across and rows repeat every ~4.8 m, so this
        # keeps the whole panel while excluding the neighbouring row. Set to 0
        # to disable and fit the whole sampled window.
        self.sc_anchor_radius_m = float(sc_cfg.get("anchor_radius_m", 2.0))

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
        self._dsm_geo_inv = None
        self._dsm_z_min, self._dsm_z_max = self._compute_dsm_z_range(
            cfg.get("ray_dsm_z_min", None), cfg.get("ray_dsm_z_max", None)
        )

    def _compute_dsm_z_range(
        self,
        z_min_cfg: Optional[float],
        z_max_cfg: Optional[float],
    ) -> Tuple[Optional[float], Optional[float]]:
        """Elevation range the ray march is allowed to search.

        Taken from the DSM itself so that no single nominal height is baked in.
        ``ray_dsm_z_min``/``ray_dsm_z_max`` override it for DSMs that extend
        well beyond the area of interest (a narrower band is cheaper)."""
        z_min = float(z_min_cfg) if z_min_cfg is not None else None
        z_max = float(z_max_cfg) if z_max_cfg is not None else None
        if z_min is not None and z_max is not None:
            return z_min, z_max

        stat_min = stat_max = None
        if self._dsm_array is not None:
            # Reduce in the array's own dtype with a `where` mask: materialising
            # a float64 copy of a preloaded DSM costs gigabytes.
            vals = self._dsm_array
            mask = np.isfinite(vals)
            if self._dsm_ds_nodata is not None:
                mask &= vals != self._dsm_ds_nodata
            if np.any(mask):
                stat_min = float(np.min(vals, where=mask, initial=np.inf))
                stat_max = float(np.max(vals, where=mask, initial=-np.inf))
        else:
            try:
                # approx_ok=1: uses overviews/subsampling, and GDAL caches the
                # result, so this stays cheap even on very large DSMs.
                stats = self._dsm_ds.GetRasterBand(1).GetStatistics(1, 1)
                stat_min, stat_max = float(stats[0]), float(stats[1])
            except Exception:
                stat_min = stat_max = None

        if stat_min is None or stat_max is None or not np.isfinite([stat_min, stat_max]).all():
            return z_min, z_max
        return (z_min if z_min is not None else stat_min,
                z_max if z_max is not None else stat_max)

    def _geo_to_image_xy_batch(
        self,
        gx: np.ndarray,
        gy: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._dsm_geo_inv is None:
            gt = self._dsm_geo
            m = np.array([[float(gt[1]), float(gt[2])],
                          [float(gt[4]), float(gt[5])]], dtype=np.float64)
            self._dsm_geo_inv = np.linalg.inv(m)
        inv = self._dsm_geo_inv
        dx = gx - float(self._dsm_geo[0])
        dy = gy - float(self._dsm_geo[3])
        return inv[0, 0] * dx + inv[0, 1] * dy, inv[1, 0] * dx + inv[1, 1] * dy

    def _read_dsm_values(self, col_f: np.ndarray, row_f: np.ndarray) -> np.ndarray:
        """DSM heights at many pixels at once; NaN off the raster or on nodata."""
        col = np.rint(col_f).astype(np.int64)
        row = np.rint(row_f).astype(np.int64)
        out = np.full(col.shape, np.nan, dtype=np.float64)
        inside = (col >= 0) & (row >= 0) & (col < self._dsm_w) & (row < self._dsm_h)
        if not np.any(inside):
            return out

        if self._dsm_array is not None:
            # Gather first, then let the float64 ``out`` upcast: converting the
            # whole preloaded DSM per call would copy gigabytes every ray.
            out[inside] = self._dsm_array[row[inside], col[inside]]
        else:
            for i in np.flatnonzero(inside):
                value = self._read_dsm_value(int(col[i]), int(row[i]))
                if value is not None:
                    out[i] = value

        out[~np.isfinite(out)] = np.nan
        if self._dsm_ds_nodata is not None:
            out[out == float(self._dsm_ds_nodata)] = np.nan
        return out

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

    def _ray_surface_probe(
        self,
        px: float,
        py: float,
        z: float,
        xs: float,
        ys: float,
        zs: float,
        rot: Sequence[float],
    ) -> Tuple[float, float, Optional[float]]:
        """Ground point where the ray sits at height ``z`` plus the DSM height
        there. dsm_z is None when the point is off the DSM or on nodata."""
        gx, gy = photo_to_ground(px, py, self.focal, z, xs, ys, zs, rot)
        col_f, row_f = geo_to_image_xy(self._dsm_geo, gx, gy)
        dsm_z = self._read_dsm_value(int(round(col_f)), int(round(row_f)))
        if dsm_z is None or not np.isfinite(dsm_z):
            return gx, gy, None
        if self._dsm_ds_nodata is not None and dsm_z == float(self._dsm_ds_nodata):
            return gx, gy, None
        return gx, gy, float(dsm_z)

    def _ray_surface_probe_batch(
        self,
        px: float,
        py: float,
        z_levels: np.ndarray,
        xs: float,
        ys: float,
        zs: float,
        rot: Sequence[float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``_ray_surface_probe`` for a whole column of ray heights at once.

        ``photo_to_ground`` is linear in z, so the ground track is just an
        affine function of ``z_levels`` -- the march costs one DSM gather
        instead of one Python iteration per step."""
        a1, a2, a3, b1, b2, b3, c1, c2, c3 = rot
        den = c1 * px + c2 * py - c3 * self.focal
        if abs(den) < 1e-12:
            gx = np.full(z_levels.shape, float(xs), dtype=np.float64)
            gy = np.full(z_levels.shape, float(ys), dtype=np.float64)
        else:
            kx = (a1 * px + a2 * py - a3 * self.focal) / den
            ky = (b1 * px + b2 * py - b3 * self.focal) / den
            dz = z_levels - float(zs)
            gx = dz * kx + float(xs)
            gy = dz * ky + float(ys)

        col_f, row_f = self._geo_to_image_xy_batch(gx, gy)
        return gx, gy, self._read_dsm_values(col_f, row_f)

    def _ray_dsm_intersection(
        self,
        img_x: float,
        img_y: float,
        pose: Sequence[float],
    ) -> Optional[Tuple[float, float, float]]:
        """Nearest-hit ray/DSM intersection.

        The camera ray ``P(z) = photo_to_ground(.., z)`` is a straight line, so a
        grazing ray can cross the DSM surface more than once (e.g. a panel edge
        and the ground behind it). The physically correct answer is the surface
        the camera actually sees: the intersection nearest the perspective
        centre, i.e. the FIRST crossing while marching the ray outward from the
        camera (decreasing z).

        The march is deterministic and makes no assumption about the site's
        height: it starts at the camera (capped just above the DSM's highest
        point) and sweeps down to just below its lowest, so every surface the
        ray can possibly meet is inside the search. An earlier version centred
        a +-4 m band on a single global scalar and could only reach one height
        regime -- on 004-CangFang, whose PV sits on two surfaces ~9 m apart,
        that returned a phantom surface 22 m below the roof the camera saw.
        See analysis/cangfang_two_heights/FINDINGS.md and
        analysis/.../id_1695/z_ref_bistability.md.
        """
        xs, ys, zs, phi, omega, kappa = pose
        rot = build_rotation(phi, omega, kappa)
        px = float(img_x) - self.cx
        py = self.cy - float(img_y)

        if self._dsm_z_min is None or self._dsm_z_max is None:
            return None
        margin = float(self.ray_dsm_z_margin)
        z_start = min(float(zs), float(self._dsm_z_max) + margin)
        z_floor = float(self._dsm_z_min) - margin
        if z_start <= z_floor:
            return None

        step = max(self.ray_dsm_march_step, 1e-6)
        n = min(int((z_start - z_floor) / step) + 2, max(2, self.ray_dsm_max_steps))
        z_levels = np.linspace(z_start, z_floor, n)      # camera -> outward
        gx, gy, dsm_z = self._ray_surface_probe_batch(px, py, z_levels, xs, ys, zs, rot)
        diff = z_levels - dsm_z                          # NaN wherever there is no DSM

        # First crossing from "ray above the surface" to "ray below it".
        above, below = diff[:-1], diff[1:]
        crossing = np.isfinite(above) & np.isfinite(below) & (above > 0.0) & (below <= 0.0)
        idx = np.flatnonzero(crossing)
        if idx.size == 0:
            return None
        k = int(idx[0])

        if below[k] == 0.0:
            self._last_dsm_z = float(dsm_z[k + 1])
            return float(gx[k + 1]), float(gy[k + 1]), float(dsm_z[k + 1])

        hit = self._bisect_ray_surface(px, py, xs, ys, zs, rot,
                                       float(z_levels[k]), float(z_levels[k + 1]))
        if hit is None:
            return None
        self._last_dsm_z = float(hit[2])
        return hit

    def _bisect_ray_surface(
        self,
        px: float,
        py: float,
        xs: float,
        ys: float,
        zs: float,
        rot: Sequence[float],
        z_above: float,
        z_below: float,
    ) -> Optional[Tuple[float, float, float]]:
        """Refine the crossing between ``z_above`` (ray above the surface) and
        ``z_below`` (ray below) by bisection on ray height."""
        gx = gy = None
        dsm_z: Optional[float] = None
        for _ in range(40):
            zm = 0.5 * (z_above + z_below)
            gx, gy, dsm_z = self._ray_surface_probe(px, py, zm, xs, ys, zs, rot)
            if dsm_z is None:
                return None
            diff = zm - dsm_z
            if abs(diff) <= self.ray_dsm_tol:
                return gx, gy, float(dsm_z)
            if diff > 0.0:
                z_above = zm
            else:
                z_below = zm
        if dsm_z is None or gx is None:
            return None
        return gx, gy, float(dsm_z)

    def _estimate_bbox_center_z(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        pose: Sequence[float],
    ) -> Optional[float]:
        bx = 0.5 * (float(x1) + float(x2))
        by = 0.5 * (float(y1) + float(y2))
        hit = self._ray_dsm_intersection(bx, by, pose)
        if hit is None:
            return None
        gx, gy, _ = hit
        col_f, row_f = geo_to_image_xy(self._dsm_geo, gx, gy)
        col = int(round(col_f))
        row = int(round(row_f))
        return self._dsm_window_median(col, row, self.ray_dsm_init_window)

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
            z_ref = self._estimate_bbox_center_z(x1, y1, x2, y2, pose)
            if z_ref is None:
                z_ref = self._dsm_global_median
            if z_ref is None:
                cc_valid = cc_flat
                rr_valid = rr_flat
                z_valid = z_vals
            else:
                alt_mask = np.abs(z_vals - z_ref) <= self.max_alt_diff
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

    def _sample_plane_points(
        self,
        bbox: Sequence[float],
        pose: Sequence[float],
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
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
            return None

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
            return None

        step = max(1, self.sample_interval)
        block_w = col_max - col_min + 1
        block_h = row_max - row_min + 1
        if self._dsm_array is not None:
            dsm_block = self._dsm_array[row_min:row_min + block_h, col_min:col_min + block_w]
        else:
            dsm_block = self._dsm_ds.ReadAsArray(col_min, row_min, block_w, block_h)
        if dsm_block is None:
            return None

        local_cols = np.arange(0, block_w, step)
        local_rows = np.arange(0, block_h, step)
        cc, rr = np.meshgrid(local_cols, local_rows)
        cc_flat = cc.ravel()
        rr_flat = rr.ravel()

        z_vals = dsm_block[rr_flat, cc_flat].astype(np.float64)
        finite_mask = np.isfinite(z_vals)
        if self._dsm_ds_nodata is not None:
            finite_mask &= z_vals != float(self._dsm_ds_nodata)
        cc_flat = cc_flat[finite_mask]
        rr_flat = rr_flat[finite_mask]
        z_vals = z_vals[finite_mask]
        if z_vals.size == 0:
            return None

        z_ref = self._estimate_bbox_center_z(x1, y1, x2, y2, pose)
        if z_ref is None:
            z_ref = self._dsm_global_median
        if z_ref is not None:
            alt_mask = np.abs(z_vals - z_ref) <= self.max_alt_diff
            cc_flat = cc_flat[alt_mask]
            rr_flat = rr_flat[alt_mask]
            z_vals = z_vals[alt_mask]
        if z_vals.size == 0:
            return None

        abs_cols = (cc_flat + col_min).astype(np.float64)
        abs_rows = (rr_flat + row_min).astype(np.float64)
        gx = gt[0] + abs_cols * gt[1] + abs_rows * gt[2]
        gy = gt[3] + abs_cols * gt[4] + abs_rows * gt[5]

        return gx, gy, z_vals

    def _fit_one_plane(
        self,
        xy: np.ndarray,
        gz: np.ndarray,
    ) -> Optional[Tuple[float, float, float, np.ndarray]]:
        """Single seeded RANSAC pass. Returns (a, b, c, inlier_mask)."""
        try:
            estimator = RANSACRegressor(
                residual_threshold=float(self.sc_ransac_threshold),
                max_trials=int(self.sc_ransac_max_trials),
                min_samples=3,
                random_state=int(self.sc_ransac_random_state),
            )
            estimator.fit(xy, gz)
        except Exception:
            return None

        coef = estimator.estimator_.coef_
        return (
            float(coef[0]),
            float(coef[1]),
            float(estimator.estimator_.intercept_),
            estimator.inlier_mask_,
        )

    def _fit_plane_ransac(
        self,
        gx: np.ndarray,
        gy: np.ndarray,
        gz: np.ndarray,
        anchor: Optional[Sequence[float]] = None,
    ) -> Optional[Tuple[float, float, float, int, float]]:
        """Fit the panel plane to the sampled DSM cells.

        The sample window is the ground bounding box of the bbox corners, which
        at grazing incidence is much wider than the panel and can straddle the
        neighbouring panel row. Rows are a sawtooth, so each is its own
        consensus set and a plain "largest set wins" vote picks whichever row
        contributed more cells -- for one XinXie feature that was the wrong row
        by 8% of the samples, putting the footprint 1.65 m off its panel.

        ``anchor`` is the ground point the feature's own centroid ray hits. When
        given, candidate surfaces are peeled off until one contains it, so the
        fit follows the feature rather than the sample count. Falls back to the
        largest surface if no candidate contains the anchor.
        """
        n = int(gz.size)
        if n < max(3, self.sc_min_inliers):
            return None

        if anchor is not None:
            # Restricting to the feature's own neighbourhood is what actually
            # removes the ambiguity: with the neighbouring row out of the cloud
            # there is only one consensus set left to find. Anchor containment
            # alone is too weak -- a plane cutting across both rows can pass
            # through the anchor while carrying nowhere near the panel's tilt.
            radius = float(self.sc_anchor_radius_m)
            if radius > 0.0:
                ax, ay = float(anchor[0]), float(anchor[1])
                near = ((gx - ax) ** 2 + (gy - ay) ** 2) <= radius * radius
                if int(np.count_nonzero(near)) >= max(3, self.sc_min_inliers):
                    local = self._search_plane(gx[near], gy[near], gz[near], anchor)
                    if local is not None:
                        return local

        return self._search_plane(gx, gy, gz, anchor)

    def _search_plane(
        self,
        gx: np.ndarray,
        gy: np.ndarray,
        gz: np.ndarray,
        anchor: Optional[Sequence[float]],
    ) -> Optional[Tuple[float, float, float, int, float]]:
        """Peel candidate surfaces until one contains ``anchor``."""
        n = int(gz.size)
        if n < max(3, self.sc_min_inliers):
            return None

        xy = np.column_stack([gx, gy])
        remaining = np.ones(n, dtype=bool)
        fallback: Optional[Tuple[float, float, float, int, float]] = None
        attempts = max(1, int(self.sc_anchor_max_candidates)) if anchor is not None else 1

        for _ in range(attempts):
            idx = np.flatnonzero(remaining)
            if idx.size < max(3, self.sc_min_inliers):
                break

            fitted = self._fit_one_plane(xy[idx], gz[idx])
            if fitted is None:
                break
            a, b, c, local_mask = fitted

            inlier_count = int(np.count_nonzero(local_mask))
            if inlier_count < self.sc_min_inliers:
                break

            tilt_rad = math.acos(1.0 / math.sqrt(1.0 + a * a + b * b))
            if math.degrees(tilt_rad) > self.sc_max_tilt_deg:
                # Unusable as an answer, but still a real surface: drop its
                # cells so the next pass can see past it.
                remaining[idx[local_mask]] = False
                continue

            sel = idx[local_mask]
            residuals = gz[sel] - (a * gx[sel] + b * gy[sel] + c)
            rmse = float(np.sqrt(np.mean(residuals * residuals))) if residuals.size > 0 else 0.0
            candidate = (a, b, c, inlier_count, rmse)

            if fallback is None:
                fallback = candidate
            if anchor is None:
                return candidate

            ax, ay, az = (float(v) for v in anchor[:3])
            if abs(a * ax + b * ay + c - az) <= self.sc_ransac_threshold:
                return candidate

            remaining[sel] = False

        return fallback

    def _ray_plane_intersect_batch(
        self,
        points_xy: Sequence[Tuple[float, float]],
        pose: Sequence[float],
        plane_abc: Tuple[float, float, float],
    ) -> Optional[np.ndarray]:
        if not points_xy:
            return None

        xs, ys, zs, phi, omega, kappa = pose
        rot = build_rotation(phi, omega, kappa)
        a1, a2, a3, b1, b2, b3, c1, c2, c3 = rot
        a, b, c = plane_abc

        pts = np.asarray(points_xy, dtype=np.float64)
        px = pts[:, 0] - self.cx
        py = self.cy - pts[:, 1]
        pz = -float(self.focal)

        dx = a1 * px + a2 * py + a3 * pz
        dy = b1 * px + b2 * py + b3 * pz
        dz = c1 * px + c2 * py + c3 * pz

        den = dz - a * dx - b * dy
        bad = np.abs(den) < 1e-9
        den = np.where(bad, 1.0, den)
        t = (c + a * float(xs) + b * float(ys) - float(zs)) / den

        gx = float(xs) + t * dx
        gy = float(ys) + t * dy
        gz = float(zs) + t * dz

        gx = np.where(bad, np.nan, gx)
        gy = np.where(bad, np.nan, gy)
        gz = np.where(bad, np.nan, gz)

        out = np.column_stack([gx, gy, gz])
        if not np.all(np.isfinite(out)):
            return None
        return out


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

        if self.method == "slope_correction":
            return self._project_feature_slope_correction(out, seg, bbox, pose)

        if self.method == "affine":
            use_affine = True
            src_pts, dst_pts = self._build_affine_pairs(bbox=bbox, pose=pose)
        elif self.method == "collinearity":
            use_affine = False
            src_pts, dst_pts = [], []
        else:
            src_pts, dst_pts = self._build_affine_pairs(bbox=bbox, pose=pose)
            use_affine = len(src_pts) >= self.min_control_points

        transformed_seg: List[List[float]] = []
        if use_affine:
            if len(src_pts) < 3:
                out["projection_method"] = "affine_failed"
                return out

            mat2, vec, _ = compute_affine_transform(src_pts, dst_pts)
            for ring in seg:
                points = _flat_to_pairs(ring)
                if len(points) < 3:
                    continue
                mapped = [(mat2[0, 0] * x + mat2[0, 1] * y + vec[0], mat2[1, 0] * x + mat2[1, 1] * y + vec[1]) for x, y in points]
                # if self.enable_slope_correction:
                #     mapped = self._apply_slope_correction_placeholder(mapped)
                transformed_seg.append(_pairs_to_flat(mapped))
        else:
            for ring in seg:
                points = _flat_to_pairs(ring)
                if len(points) < 3:
                    continue
                mapped = self._project_points_direct_collinearity(points, pose=pose)
                # if self.enable_slope_correction:
                #     mapped = self._apply_slope_correction_placeholder(mapped)
                transformed_seg.append(_pairs_to_flat(mapped))

        if transformed_seg:
            out["segmentation"] = transformed_seg
            out["bbox"] = _bbox_from_segmentation(transformed_seg)
            out["projection_method"] = "affine" if use_affine else "collinearity"

        return out

    def _project_feature_slope_correction(
        self,
        out: Feature,
        seg: List[List[float]],
        bbox: Sequence[float],
        pose: Sequence[float],
    ) -> Feature:
        sampled = self._sample_plane_points(bbox=bbox, pose=pose)
        if sampled is None:
            out["projection_method"] = "slope_correction_failed"
            return out
        gx, gy, gz = sampled
        if gz.size < self.sc_min_inliers:
            out["projection_method"] = "slope_correction_failed"
            return out

        # Where the feature itself sits: the sample window may cover several
        # panel rows, and only this one belongs to the feature being projected.
        x1, y1, x2, y2 = [float(v) for v in bbox]
        anchor = self._ray_dsm_intersection((x1 + x2) * 0.5, (y1 + y2) * 0.5, pose)

        plane = self._fit_plane_ransac(gx, gy, gz, anchor=anchor)
        if plane is None:
            out["projection_method"] = "slope_correction_failed"
            return out
        a, b, c, inlier_count, rmse = plane

        transformed_seg: List[List[float]] = []
        elevations: List[List[float]] = []
        for ring in seg:
            points = _flat_to_pairs(ring)
            if len(points) < 3:
                continue
            xyz = self._ray_plane_intersect_batch(points, pose, (a, b, c))
            if xyz is None or xyz.shape[0] == 0:
                continue
            flat: List[float] = []
            ring_z: List[float] = []
            for i in range(xyz.shape[0]):
                flat.append(float(xyz[i, 0]))
                flat.append(float(xyz[i, 1]))
                ring_z.append(float(xyz[i, 2]))
            transformed_seg.append(flat)
            elevations.append(ring_z)

        if not transformed_seg:
            out["projection_method"] = "slope_correction_failed"
            return out

        norm = math.sqrt(a * a + b * b + 1.0)
        tilt_rad = math.acos(1.0 / norm)
        azimuth_rad = math.atan2(a, b)

        out["segmentation"] = transformed_seg
        out["vertex_elevations"] = elevations
        out["bbox"] = _bbox_from_segmentation(transformed_seg)
        out["projection_method"] = "slope_correction"
        out["tilt_angle"] = float(tilt_rad)
        out["azimuth"] = float(azimuth_rad)
        out["plane_fit_inliers"] = int(inlier_count)
        out["plane_fit_rmse"] = float(rmse)
        return out

    def close(self) -> None:
        self._dsm_array = None
        self._dsm_ds = None
