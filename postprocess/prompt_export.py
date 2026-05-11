from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from osgeo import gdal, ogr, osr
from shapely.geometry import Polygon
from tqdm import tqdm

from io_flow.input_resolver import resolve_image_paths
from projection.collinearity import (
    build_rotation,
    geo_to_image_xy,
    ground_to_photo,
    image_xy_to_geo,
    photo_to_ground,
    read_pose_csv,
)

try:
    from rtree import index as _rtree_index
except Exception:
    _rtree_index = None


def _as_dict(cfg: Dict[str, Any], key: str) -> Dict[str, Any]:
    obj = cfg.get(key, {})
    if isinstance(obj, dict):
        return obj
    return {}


def _resolve_mode(cfg: Dict[str, Any]) -> str:
    prompt_cfg = _as_dict(_as_dict(cfg, "postprocess"), "prompt_export")
    mode = str(prompt_cfg.get("mode", "auto")).strip().lower()
    if mode in {"oblique", "dom"}:
        return mode

    projection_mode = str(_as_dict(cfg, "projection").get("mode", "auto")).strip().lower()
    if projection_mode == "dom":
        return "dom"
    return "oblique"


def _normalize_image_name(name_or_path: str) -> str:
    return os.path.basename(str(name_or_path).strip())


def _oblique_image_ground_bbox(
    pose_params: List[str],
    image_width: int,
    image_height: int,
    focal: float,
    cx: float,
    cy: float,
    avg_alt: float,
) -> Optional[Tuple[float, float, float, float]]:
    try:
        xs, ys, zs = float(pose_params[0]), float(pose_params[1]), float(pose_params[2])
        phi, omega, kappa = float(pose_params[3]), float(pose_params[4]), float(pose_params[5])
    except Exception:
        return None

    rot = build_rotation(phi, omega, kappa)
    corners_px = [(0.0, 0.0), (float(image_width), 0.0), (float(image_width), float(image_height)), (0.0, float(image_height))]
    ground_pts: List[Tuple[float, float]] = []
    for ix, iy in corners_px:
        px = ix - cx
        py = cy - iy
        gx, gy = photo_to_ground(px, py, focal, avg_alt, xs, ys, zs, rot)
        if abs(gx) > 1e11 or abs(gy) > 1e11:
            return None
        ground_pts.append((gx, gy))

    xs_g = [p[0] for p in ground_pts]
    ys_g = [p[1] for p in ground_pts]
    return min(xs_g), min(ys_g), max(xs_g), max(ys_g)


class _ObliqueFootprintIndex:
    def __init__(self, boxes: List[Tuple[str, Tuple[float, float, float, float]]]) -> None:
        self._boxes = boxes
        self._names: List[str] = [x[0] for x in boxes]
        self._rtree = None
        if _rtree_index is not None and boxes:
            idx = _rtree_index.Index()
            for i, (_, box) in enumerate(boxes):
                idx.insert(i, box)
            self._rtree = idx

    def query(self, bbox: Tuple[float, float, float, float]) -> List[str]:
        if self._rtree is not None:
            ids = list(self._rtree.intersection(bbox))
            return [self._names[i] for i in ids]

        x1, y1, x2, y2 = bbox
        out: List[str] = []
        for name, (bx1, by1, bx2, by2) in self._boxes:
            if x2 < bx1 or bx2 < x1 or y2 < by1 or by2 < y1:
                continue
            out.append(name)
        return out


def _oblique_prompt_bbox(
    pose_params: List[str],
    dsm: gdal.Dataset,
    dsm_transform: Tuple[float, ...],
    geom: ogr.Geometry,
    oblique_cfg: Dict[str, Any],
) -> List[float]:
    f = float(oblique_cfg.get("focal", 3713.29))
    cx = float(oblique_cfg.get("cx", 2647.02))
    cy = float(oblique_cfg.get("cy", 1969.28))
    default_alt = float(oblique_cfg.get("avg_alt", 37.5))

    xs, ys, zs = float(pose_params[0]), float(pose_params[1]), float(pose_params[2])
    phi, omega, kappa = float(pose_params[3]), float(pose_params[4]), float(pose_params[5])
    rot = build_rotation(phi, omega, kappa)

    ring = geom.GetGeometryRef(0)
    if ring is None:
        raise ValueError("invalid polygon ring")
    poly = Polygon(ring.GetPoints())
    rotated_rec = poly.minimum_rotated_rectangle
    coords = rotated_rec.exterior.coords
    if len(coords) < 4:
        raise ValueError("invalid rotated rectangle")

    corners = coords[:4]
    dsm_pixels = [geo_to_image_xy(dsm_transform, x, y) for x, y in corners]

    def get_z(col: float, row: float) -> float:
        try:
            arr = dsm.ReadAsArray(int(col), int(row), 1, 1)
            return float(arr[0][0])
        except Exception:
            return default_alt

    geo_points: List[Tuple[float, float, float]] = []
    for col, row in dsm_pixels:
        z = get_z(col, row)
        gx, gy = image_xy_to_geo(dsm_transform, col, row)
        geo_points.append((gx, gy, z))

    img_pts: List[Tuple[float, float]] = []
    for gx, gy, z in geo_points:
        u, v = ground_to_photo(f, gx, gy, z, xs, ys, zs, rot)
        img_pts.append((u + cx, cy - v))

    xs_px = [p[0] for p in img_pts]
    ys_px = [p[1] for p in img_pts]
    return [min(xs_px), min(ys_px), max(xs_px), max(ys_px)]


def _oblique_feature_geo_points(
    geom: ogr.Geometry,
    dsm: gdal.Dataset,
    dsm_transform: Tuple[float, ...],
    default_alt: float,
) -> Optional[List[Tuple[float, float, float]]]:
    ring = geom.GetGeometryRef(0)
    if ring is None:
        return None
    poly = Polygon(ring.GetPoints())
    rotated_rec = poly.minimum_rotated_rectangle
    coords = rotated_rec.exterior.coords
    if len(coords) < 4:
        return None

    corners = coords[:4]
    dsm_pixels = [geo_to_image_xy(dsm_transform, x, y) for x, y in corners]

    def get_z(col: float, row: float) -> float:
        try:
            arr = dsm.ReadAsArray(int(col), int(row), 1, 1)
            return float(arr[0][0])
        except Exception:
            return default_alt

    geo_points: List[Tuple[float, float, float]] = []
    for col, row in dsm_pixels:
        z = get_z(col, row)
        gx, gy = image_xy_to_geo(dsm_transform, col, row)
        geo_points.append((gx, gy, z))

    return geo_points


def _oblique_prompt_bbox_from_geo_points(
    geo_points: List[Tuple[float, float, float]],
    pose: Tuple[float, float, float, Tuple[float, ...]],
    oblique_cfg: Dict[str, Any],
) -> List[float]:
    f = float(oblique_cfg.get("focal", 3713.29))
    cx = float(oblique_cfg.get("cx", 2647.02))
    cy = float(oblique_cfg.get("cy", 1969.28))
    xs, ys, zs, rot = pose

    img_pts: List[Tuple[float, float]] = []
    for gx, gy, z in geo_points:
        u, v = ground_to_photo(f, gx, gy, z, xs, ys, zs, rot)
        img_pts.append((u + cx, cy - v))

    xs_px = [p[0] for p in img_pts]
    ys_px = [p[1] for p in img_pts]
    return [min(xs_px), min(ys_px), max(xs_px), max(ys_px)]


def _export_oblique_prompts(cfg: Dict[str, Any], shp_path: str, prompt_cfg: Dict[str, Any]) -> Dict[str, Any]:
    projection_cfg = _as_dict(cfg, "projection")
    oblique_cfg = _as_dict(projection_cfg, "oblique")
    output_cfg = _as_dict(cfg, "output")

    pose_csv = str(oblique_cfg.get("pose_csv", "")).strip()
    dsm_path = str(oblique_cfg.get("dsm_path", "")).strip()
    if not pose_csv or not os.path.exists(pose_csv):
        raise FileNotFoundError(f"Cannot find pose_csv: {pose_csv}")
    if not dsm_path or not os.path.exists(dsm_path):
        raise FileNotFoundError(f"Cannot find dsm_path: {dsm_path}")

    output_dir = str(prompt_cfg.get("output_dir", "")).strip()
    if not output_dir:
        final_merged_shp = str(output_cfg.get("final_merged_shp", "outputs/final_merged.shp"))
        output_dir = os.path.splitext(final_merged_shp)[0] + "_bbox_prompts"
    os.makedirs(output_dir, exist_ok=True)

    pose_dict = read_pose_csv(pose_csv)
    dsm_ds = gdal.Open(dsm_path)
    if dsm_ds is None:
        raise FileNotFoundError(f"Cannot open DSM: {dsm_path}")
    dsm_transform = dsm_ds.GetGeoTransform()

    min_size = float(prompt_cfg.get("min_size", 50.0))
    include_intersections = bool(prompt_cfg.get("include_intersections", True))
    use_spatial_index = bool(prompt_cfg.get("use_spatial_index", True))

    image_width = int(oblique_cfg.get("image_width", 5280))
    image_height = int(oblique_cfg.get("image_height", 3956))
    focal = float(oblique_cfg.get("focal", 3713.29))
    cx = float(oblique_cfg.get("cx", 2647.02))
    cy = float(oblique_cfg.get("cy", 1969.28))
    avg_alt = float(oblique_cfg.get("avg_alt", 37.5))

    img_paths = resolve_image_paths(_as_dict(cfg, "data"))
    candidate_names = [_normalize_image_name(p) for p in img_paths] if img_paths else list(pose_dict.keys())
    candidate_names = [n for n in candidate_names if n in pose_dict]

    footprint_boxes: List[Tuple[str, Tuple[float, float, float, float]]] = []
    if include_intersections:
        for name in tqdm(candidate_names, desc="Building oblique footprint index", leave=False):
            bbox = _oblique_image_ground_bbox(
                pose_params=pose_dict[name],
                image_width=image_width,
                image_height=image_height,
                focal=focal,
                cx=cx,
                cy=cy,
                avg_alt=avg_alt,
            )
            if bbox is not None:
                footprint_boxes.append((name, bbox))

    footprint_index = _ObliqueFootprintIndex(footprint_boxes) if (include_intersections and use_spatial_index) else None

    pose_cache: Dict[str, Tuple[float, float, float, Tuple[float, ...]]] = {}
    for name in candidate_names:
        pose_params = pose_dict.get(name)
        if not pose_params:
            continue
        try:
            xs, ys, zs = float(pose_params[0]), float(pose_params[1]), float(pose_params[2])
            phi, omega, kappa = float(pose_params[3]), float(pose_params[4]), float(pose_params[5])
        except Exception:
            continue
        rot = build_rotation(phi, omega, kappa)
        pose_cache[name] = (xs, ys, zs, rot)

    results: Dict[str, List[List[float]]] = defaultdict(list)
    shp_ds = ogr.Open(shp_path)
    if shp_ds is None:
        raise FileNotFoundError(f"Cannot open {shp_path}")
    layer = shp_ds.GetLayer()

    for feat in tqdm(layer, total=layer.GetFeatureCount(), desc="Exporting oblique prompts", leave=False):
        geom = feat.GetGeometryRef()
        if geom is None:
            continue

        if float(feat.GetFieldAsString("con")) < float(prompt_cfg.get("min_confidence", 0.5)):
            continue
        
        src = _normalize_image_name(feat.GetFieldAsString("src")) if feat.GetFieldIndex("src") != -1 else ""
        target_set = set()
        if src and src in pose_dict:
            target_set.add(src)

        if include_intersections:
            minx, maxx, miny, maxy = geom.GetEnvelope()
            query_box = (float(minx), float(miny), float(maxx), float(maxy))
            if footprint_index is not None:
                hit_names = footprint_index.query(query_box)
            else:
                # Fallback: linear scan when index is disabled.
                hit_names = []
                for name, (bx1, by1, bx2, by2) in footprint_boxes:
                    if query_box[2] < bx1 or bx2 < query_box[0] or query_box[3] < by1 or by2 < query_box[1]:
                        continue
                    hit_names.append(name)
            for name in hit_names:
                if name in pose_dict:
                    target_set.add(name)

        if not target_set:
            continue
        targets: List[str] = list(target_set)

        geo_points = _oblique_feature_geo_points(
            geom=geom,
            dsm=dsm_ds,
            dsm_transform=dsm_transform,
            default_alt=avg_alt,
        )
        if not geo_points:
            continue

        for photo in targets:
            try:
                pose = pose_cache.get(photo)
                if pose is None:
                    continue
                bbox = _oblique_prompt_bbox_from_geo_points(
                    geo_points=geo_points,
                    pose=pose,
                    oblique_cfg=oblique_cfg,
                )
                if (bbox[2] - bbox[0]) < min_size or (bbox[3] - bbox[1]) < min_size:
                    continue
                if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > image_width or bbox[3] > image_height:
                    continue
                results[photo].append(bbox)
            except Exception:
                continue

    file_count = 0
    box_count = 0
    for img_name, bboxes in results.items():
        base = os.path.splitext(img_name)[0]
        out_path = os.path.join(output_dir, f"{base}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            for b in bboxes:
                f.write(f"{b[0]},{b[1]},{b[2]},{b[3]}\n")
        file_count += 1
        box_count += len(bboxes)

    return {"mode": "oblique", "output": output_dir, "files": file_count, "boxes": box_count}


def _geo_to_pixel(geo_transform: Tuple[float, ...], x: float, y: float) -> Tuple[float, float]:
    det = geo_transform[1] * geo_transform[5] - geo_transform[2] * geo_transform[4]
    if abs(det) < 1e-12:
        raise ValueError("Invalid geotransform: determinant is zero")
    dx = x - geo_transform[0]
    dy = y - geo_transform[3]
    pixel_x = (geo_transform[5] * dx - geo_transform[2] * dy) / det
    pixel_y = (-geo_transform[4] * dx + geo_transform[1] * dy) / det
    return pixel_x, pixel_y


def _create_coord_transform(shp_layer: ogr.Layer, dom_ds: gdal.Dataset) -> Optional[osr.CoordinateTransformation]:
    shp_srs = shp_layer.GetSpatialRef()
    dom_wkt = dom_ds.GetProjection()
    if shp_srs is None or not dom_wkt:
        return None
    dom_srs = osr.SpatialReference()
    dom_srs.ImportFromWkt(dom_wkt)
    try:
        if bool(shp_srs.IsSame(dom_srs)):
            return None
    except Exception:
        pass
    return osr.CoordinateTransformation(shp_srs, dom_srs)


def _feature_to_dom_bbox(
    geom: ogr.Geometry,
    geo_transform: Tuple[float, ...],
    coord_transform: Optional[osr.CoordinateTransformation],
    width: int,
    height: int,
    clip_to_image: bool,
    normalized: bool,
) -> Optional[List[float]]:
    geom = geom.Clone()
    if coord_transform is not None:
        geom.Transform(coord_transform)

    min_x, max_x, min_y, max_y = geom.GetEnvelope()
    p0 = _geo_to_pixel(geo_transform, min_x, min_y)
    p1 = _geo_to_pixel(geo_transform, min_x, max_y)
    p2 = _geo_to_pixel(geo_transform, max_x, min_y)
    p3 = _geo_to_pixel(geo_transform, max_x, max_y)

    xs = [p0[0], p1[0], p2[0], p3[0]]
    ys = [p0[1], p1[1], p2[1], p3[1]]
    x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)

    if clip_to_image:
        x_min = max(0.0, min(x_min, float(width)))
        y_min = max(0.0, min(y_min, float(height)))
        x_max = max(0.0, min(x_max, float(width)))
        y_max = max(0.0, min(y_max, float(height)))
    else:
        if x_max < 0 or y_max < 0 or x_min > width or y_min > height:
            return None

    if x_max <= x_min or y_max <= y_min:
        return None

    if normalized:
        x_min /= width
        y_min /= height
        x_max /= width
        y_max /= height

    return [x_min, y_min, x_max, y_max]


def _export_dom_prompts(cfg: Dict[str, Any], shp_path: str, prompt_cfg: Dict[str, Any]) -> Dict[str, Any]:
    data_cfg = _as_dict(cfg, "data")
    output_cfg = _as_dict(cfg, "output")

    img_paths = resolve_image_paths(data_cfg)
    if not img_paths:
        raise ValueError("No DOM image found from data config")
    dom_path = img_paths[0]

    output_txt = str(prompt_cfg.get("output_txt", "")).strip()
    if not output_txt:
        final_merged_shp = str(output_cfg.get("final_merged_shp", "outputs/final_merged.shp"))
        output_txt = os.path.splitext(final_merged_shp)[0] + "_bbox_prompts.txt"
    os.makedirs(os.path.dirname(output_txt) or ".", exist_ok=True)

    normalized = bool(prompt_cfg.get("normalized", False))
    clip_to_image = bool(prompt_cfg.get("clip_to_image", True))
    min_size = float(prompt_cfg.get("min_size", 1.0))
    reproject_when_needed = bool(prompt_cfg.get("reproject_when_needed", False))

    shp_ds = ogr.Open(shp_path)
    if shp_ds is None:
        raise FileNotFoundError(f"Cannot open {shp_path}")
    layer = shp_ds.GetLayer()

    dom_ds = gdal.Open(dom_path)
    if dom_ds is None:
        raise FileNotFoundError(f"Cannot open {dom_path}")

    geo_transform = dom_ds.GetGeoTransform()
    width = dom_ds.RasterXSize
    height = dom_ds.RasterYSize
    coord_transform = _create_coord_transform(layer, dom_ds) if reproject_when_needed else None

    bboxes: List[List[float]] = []
    for feat in tqdm(layer, total=layer.GetFeatureCount(), desc="Exporting dom prompts", leave=False):
        geom = feat.GetGeometryRef()
        if geom is None:
            continue
        bbox = _feature_to_dom_bbox(
            geom=geom,
            geo_transform=geo_transform,
            coord_transform=coord_transform,
            width=width,
            height=height,
            clip_to_image=clip_to_image,
            normalized=normalized,
        )
        if bbox is None:
            continue
        if (bbox[2] - bbox[0]) < min_size or (bbox[3] - bbox[1]) < min_size:
            continue
        if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > width or bbox[3] > height:
            continue
        bboxes.append(bbox)

    with open(output_txt, "w", encoding="utf-8") as f:
        for b in bboxes:
            f.write(f"{b[0]:.6f},{b[1]:.6f},{b[2]:.6f},{b[3]:.6f}\n")

    return {"mode": "dom", "output": output_txt, "boxes": len(bboxes)}


def maybe_export_bbox_prompts(cfg: Dict[str, Any], shp_path: str) -> Optional[Dict[str, Any]]:
    post_cfg = _as_dict(cfg, "postprocess")
    prompt_cfg = _as_dict(post_cfg, "prompt_export")
    if not bool(prompt_cfg.get("enabled", False)):
        return None

    mode = _resolve_mode(cfg)
    if mode == "dom":
        return _export_dom_prompts(cfg=cfg, shp_path=shp_path, prompt_cfg=prompt_cfg)
    return _export_oblique_prompts(cfg=cfg, shp_path=shp_path, prompt_cfg=prompt_cfg)
