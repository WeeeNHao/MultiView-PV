from __future__ import annotations

import argparse
import os
from typing import List, Optional, Tuple

from osgeo import gdal, ogr, osr

from config_loader import load_config
from input_resolver import resolve_image_paths


def geo_to_pixel(geo_transform: Tuple[float, ...], x: float, y: float) -> Tuple[float, float]:
    det = geo_transform[1] * geo_transform[5] - geo_transform[2] * geo_transform[4]
    if abs(det) < 1e-12:
        raise ValueError("Invalid geotransform: determinant is zero.")

    dx = x - geo_transform[0]
    dy = y - geo_transform[3]
    pixel_x = (geo_transform[5] * dx - geo_transform[2] * dy) / det
    pixel_y = (-geo_transform[4] * dx + geo_transform[1] * dy) / det
    return pixel_x, pixel_y


def create_coord_transform(shp_layer: ogr.Layer, dom_ds: gdal.Dataset) -> Optional[osr.CoordinateTransformation]:
    shp_srs = shp_layer.GetSpatialRef()
    dom_wkt = dom_ds.GetProjection()
    dom_srs = osr.SpatialReference()
    if dom_wkt:
        dom_srs.ImportFromWkt(dom_wkt)
    else:
        dom_srs = None

    if shp_srs is None or dom_srs is None:
        return None

    try:
        if bool(shp_srs.IsSame(dom_srs)):
            return None
    except Exception:
        pass

    return osr.CoordinateTransformation(shp_srs, dom_srs)


def feature_to_bbox(
    geom: ogr.Geometry,
    geo_transform: Tuple[float, ...],
    coord_transform: Optional[osr.CoordinateTransformation],
    img_width: int,
    img_height: int,
    clip_to_image: bool,
    normalized: bool,
) -> Optional[List[float]]:
    geom = geom.Clone()
    if coord_transform is not None:
        geom.Transform(coord_transform)

    min_x, max_x, min_y, max_y = geom.GetEnvelope()
    p0 = geo_to_pixel(geo_transform, min_x, min_y)
    p1 = geo_to_pixel(geo_transform, min_x, max_y)
    p2 = geo_to_pixel(geo_transform, max_x, min_y)
    p3 = geo_to_pixel(geo_transform, max_x, max_y)

    xs = [p0[0], p1[0], p2[0], p3[0]]
    ys = [p0[1], p1[1], p2[1], p3[1]]

    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)

    if clip_to_image:
        x_min = max(0.0, min(x_min, float(img_width)))
        y_min = max(0.0, min(y_min, float(img_height)))
        x_max = max(0.0, min(x_max, float(img_width)))
        y_max = max(0.0, min(y_max, float(img_height)))
    else:
        if x_max < 0 or y_max < 0 or x_min > img_width or y_min > img_height:
            return None

    if x_max <= x_min or y_max <= y_min:
        return None

    if normalized:
        x_min /= img_width
        y_min /= img_height
        x_max /= img_width
        y_max /= img_height

    return [x_min, y_min, x_max, y_max]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to DOM RuntimeConfig YAML")
    parser.add_argument("--shp", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--normalized", action="store_true", default=False)
    parser.add_argument("--clip-to-image", action="store_true", default=True)
    parser.add_argument("--min-size", type=float, default=1.0)
    parser.add_argument("--reproject-when-needed", action="store_true", default=False)
    args = parser.parse_args()

    cfg = load_config(args.config)
    img_paths = resolve_image_paths(cfg.get("data", {}))
    if not img_paths:
        raise ValueError(f"No DOM image found in config: {args.config}")
    dom_path = img_paths[0]

    shp_ds = ogr.Open(args.shp)
    if shp_ds is None:
        raise FileNotFoundError(f"Cannot open {args.shp}")
    layer = shp_ds.GetLayer()

    dom_ds = gdal.Open(dom_path)
    if dom_ds is None:
        raise FileNotFoundError(f"Cannot open {dom_path}")

    geo_transform = dom_ds.GetGeoTransform()
    width = dom_ds.RasterXSize
    height = dom_ds.RasterYSize
    coord_transform = create_coord_transform(layer, dom_ds) if args.reproject_when_needed else None

    bboxes = []
    for feature in layer:
        geom = feature.GetGeometryRef()
        if geom is None:
            continue

        bbox = feature_to_bbox(
            geom, geo_transform, coord_transform, width, height, args.clip_to_image, args.normalized
        )
        if bbox is None:
            continue

        if bbox[2] - bbox[0] < args.min_size or bbox[3] - bbox[1] < args.min_size:
            continue
        bboxes.append(bbox)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for b in bboxes:
            f.write(f"{b[0]:.6f},{b[1]:.6f},{b[2]:.6f},{b[3]:.6f}\\n")


if __name__ == "__main__":
    main()
