from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from osgeo import gdal, ogr
from shapely.geometry import Polygon
from tqdm import tqdm

from cli.select_views import ImageFootprintIndex
from utils.config_loader import load_config
from io_flow.input_resolver import resolve_image_paths
from projection.collinearity import (
    build_rotation,
    geo_to_image_xy,
    ground_to_photo,
    image_xy_to_geo,
    read_pose_csv,
)


def generate_bbox_prompt(
    pose_params: List[str],
    dsm: gdal.Dataset,
    dsm_transform: Tuple[float, ...],
    geom: ogr.Geometry,
    proj_cfg: Dict[str, Any],
) -> List[float]:
    obl_cfg = proj_cfg.get("oblique", {})
    f = float(obl_cfg.get("focal", 3713.29))
    cx = float(obl_cfg.get("cx", 2647.02))
    cy = float(obl_cfg.get("cy", 1969.28))
    default_alt = float(obl_cfg.get("avg_alt", 37.5))
    
    xs, ys, zs = float(pose_params[0]), float(pose_params[1]), float(pose_params[2])
    phi, omega, kappa = float(pose_params[3]), float(pose_params[4]), float(pose_params[5])
    rot = build_rotation(phi, omega, kappa)

    poly = Polygon(geom.GetGeometryRef(0).GetPoints())
    rotated_rec = poly.minimum_rotated_rectangle
    coords = rotated_rec.exterior.coords
    
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]
    x4, y4 = coords[3]

    c0, r0 = geo_to_image_xy(dsm_transform, x1, y1)
    c1, r1 = geo_to_image_xy(dsm_transform, x2, y2)
    c2, r2 = geo_to_image_xy(dsm_transform, x3, y3)
    c3, r3 = geo_to_image_xy(dsm_transform, x4, y4)

    def get_z(col, row):
        try:
            return float(dsm.ReadAsArray(int(col), int(row), 1, 1)[0][0])
        except Exception:
            return default_alt

    z0 = get_z(c0, r0)
    z1 = get_z(c1, r1)
    z2 = get_z(c2, r2)
    z3 = get_z(c3, r3)

    gx0, gy0 = image_xy_to_geo(dsm_transform, c0, r0)
    gx1, gy1 = image_xy_to_geo(dsm_transform, c1, r1)
    gx2, gy2 = image_xy_to_geo(dsm_transform, c2, r2)
    gx3, gy3 = image_xy_to_geo(dsm_transform, c3, r3)

    u0, v0 = ground_to_photo(f, gx0, gy0, z0, xs, ys, zs, rot)
    u1, v1 = ground_to_photo(f, gx1, gy1, z1, xs, ys, zs, rot)
    u2, v2 = ground_to_photo(f, gx2, gy2, z2, xs, ys, zs, rot)
    u3, v3 = ground_to_photo(f, gx3, gy3, z3, xs, ys, zs, rot)

    px1, py1 = u0 + cx, cy - v0
    px2, py2 = u1 + cx, cy - v1
    px3, py3 = u2 + cx, cy - v2
    px4, py4 = u3 + cx, cy - v3

    return [
        min(px1, px2, px3, px4),
        min(py1, py2, py3, py4),
        max(px1, px2, px3, px4),
        max(py1, py2, py3, py4),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to RuntimeConfig YAML")
    parser.add_argument("--shp_file", required=True)
    parser.add_argument("--output_folder", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    proj_cfg = cfg.get("projection", {})
    obl_cfg = proj_cfg.get("oblique", {})
    
    pos_file = str(obl_cfg.get("pose_csv", ""))
    dsm_file = str(obl_cfg.get("dsm_path", ""))
    
    if not pos_file or not os.path.exists(pos_file):
        raise FileNotFoundError(f"Cannot find pose_csv: {pos_file}")
    if not dsm_file or not os.path.exists(dsm_file):
        raise FileNotFoundError(f"Cannot find dsm_path: {dsm_file}")

    pos_dict = read_pose_csv(pos_file)
    dsm_ds = gdal.Open(dsm_file)
    if not dsm_ds:
        raise FileNotFoundError(f"Cannot open DSM: {dsm_file}")
    dsm_transform = dsm_ds.GetGeoTransform()

    img_paths = resolve_image_paths(cfg.get("data", {}))
    img_index = ImageFootprintIndex(img_paths, proj_cfg)

    results = defaultdict(list)
    shp_ds = ogr.Open(args.shp_file)
    layer = shp_ds.GetLayer()

    for feature in tqdm(layer, total=layer.GetFeatureCount(), desc="Generating BBox Prompts"):
        geom = feature.GetGeometryRef()
        if not geom:
            continue
        
        src = feature.GetFieldAsString("src") if feature.GetFieldIndex("src") != -1 else ""
        tgt_photos = set()
        if src:
            tgt_photos.add(src)

        minx, maxx, miny, maxy = geom.GetEnvelope()
        intersected = img_index.query([minx, miny, maxx, maxy])
        for img_path in intersected:
            img_name = os.path.basename(img_path)
            if img_name != src and img_name in pos_dict:
                tgt_photos.add(img_name)

        for photo in tgt_photos:
            if photo not in pos_dict:
                continue
            try:
                bbox = generate_bbox_prompt(pos_dict[photo], dsm_ds, dsm_transform, geom, proj_cfg)
                if any(x < 0 for x in bbox):
                    continue
                if bbox[2] - bbox[0] < 50 or bbox[3] - bbox[1] < 50:
                    continue
                results[photo].append(bbox)
            except Exception:
                pass

    os.makedirs(args.output_folder, exist_ok=True)
    for img_name, bboxes in results.items():
        base = os.path.splitext(img_name)[0]
        out_path = os.path.join(args.output_folder, f"{base}.txt")
        with open(out_path, "w") as f:
            for b in bboxes:
                f.write(f"{b[0]},{b[1]},{b[2]},{b[3]}\\n")


if __name__ == "__main__":
    main()
