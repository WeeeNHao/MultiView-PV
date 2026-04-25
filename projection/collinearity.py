from __future__ import annotations

from math import cos, sin
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def read_pose_csv(csv_path: str) -> Dict[str, List[str]]:
    photo_dict: Dict[str, List[str]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items = line.split(",")
            if len(items) < 7:
                continue
            photo_dict[items[0]] = items[1:]
    return photo_dict


def build_rotation(phi: float, omega: float, kappa: float) -> Tuple[float, ...]:
    a1 = cos(phi) * cos(kappa) - sin(phi) * sin(omega) * sin(kappa)
    a2 = -cos(phi) * sin(kappa) - sin(phi) * sin(omega) * cos(kappa)
    a3 = -sin(phi) * cos(omega)

    b1 = cos(omega) * sin(kappa)
    b2 = cos(omega) * cos(kappa)
    b3 = -sin(omega)

    c1 = sin(phi) * cos(kappa) + cos(phi) * sin(omega) * sin(kappa)
    c2 = -sin(phi) * sin(kappa) + cos(phi) * sin(omega) * cos(kappa)
    c3 = cos(phi) * cos(omega)
    return a1, a2, a3, b1, b2, b3, c1, c2, c3


def ground_to_photo(
    f: float,
    x: float,
    y: float,
    z: float,
    xs: float,
    ys: float,
    zs: float,
    rot: Sequence[float],
) -> Tuple[float, float]:
    a1, a2, a3, b1, b2, b3, c1, c2, c3 = rot
    den = a3 * (x - xs) + b3 * (y - ys) + c3 * (z - zs)
    if abs(den) < 1e-12:
        return 1e12, 1e12

    px = -f * (a1 * (x - xs) + b1 * (y - ys) + c1 * (z - zs)) / den
    py = -f * (a2 * (x - xs) + b2 * (y - ys) + c2 * (z - zs)) / den
    return px, py


def photo_to_ground(
    px: float,
    py: float,
    f: float,
    z: float,
    xs: float,
    ys: float,
    zs: float,
    rot: Sequence[float],
) -> Tuple[float, float]:
    a1, a2, a3, b1, b2, b3, c1, c2, c3 = rot
    den = c1 * px + c2 * py - c3 * f
    if abs(den) < 1e-12:
        return xs, ys

    xg = (z - zs) * (a1 * px + a2 * py - a3 * f) / den + xs
    yg = (z - zs) * (b1 * px + b2 * py - b3 * f) / den + ys
    return xg, yg


def geo_to_image_xy(geo_transform: Sequence[float], x: float, y: float) -> Tuple[float, float]:
    a = np.array(
        [[float(geo_transform[1]), float(geo_transform[2])], [float(geo_transform[4]), float(geo_transform[5])]],
        dtype=np.float64,
    )
    b = np.array([x - float(geo_transform[0]), y - float(geo_transform[3])], dtype=np.float64)
    col, row = np.linalg.solve(a, b)
    return float(col), float(row)


def image_xy_to_geo(geo_transform: Sequence[float], col: float, row: float) -> Tuple[float, float]:
    gx = float(geo_transform[0]) + col * float(geo_transform[1]) + row * float(geo_transform[2])
    gy = float(geo_transform[3]) + col * float(geo_transform[4]) + row * float(geo_transform[5])
    return gx, gy


def compute_affine_transform(
    src_pts: Iterable[Tuple[float, float]],
    dst_pts: Iterable[Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    src = np.array(list(src_pts), dtype=np.float64)
    dst = np.array(list(dst_pts), dtype=np.float64)
    if len(src) < 3 or len(src) != len(dst):
        raise ValueError("Need at least 3 paired points for affine transform")

    n = src.shape[0]
    m = np.zeros((2 * n, 6), dtype=np.float64)
    b = np.zeros((2 * n,), dtype=np.float64)

    for i in range(n):
        x, y = src[i]
        xg, yg = dst[i]
        m[2 * i] = [x, y, 1, 0, 0, 0]
        m[2 * i + 1] = [0, 0, 0, x, y, 1]
        b[2 * i] = xg
        b[2 * i + 1] = yg

    params, _, _, _ = np.linalg.lstsq(m, b, rcond=None)
    a, b1, e = params[0], params[1], params[2]
    c, d, f = params[3], params[4], params[5]

    mat2 = np.array([[a, b1], [c, d]], dtype=np.float64)
    vec = np.array([e, f], dtype=np.float64)
    mat3 = np.array([[a, b1, e], [c, d, f], [0.0, 0.0, 1.0]], dtype=np.float64)
    return mat2, vec, mat3
