from __future__ import annotations

import argparse
import os
import random
from typing import Any, Dict, List, Set

from osgeo import ogr
from rtree import index
from tqdm import tqdm

from utils.config_loader import load_config
from io_flow.input_resolver import resolve_image_paths
from projection.collinearity import build_rotation, photo_to_ground, read_pose_csv


def _normalize_src(src: str) -> str:
    if not src:
        return ""
    return os.path.basename(str(src)).strip().lower()


def _bbox_from_geom(geom: ogr.Geometry) -> List[float]:
    min_x, max_x, min_y, max_y = geom.GetEnvelope()
    return [min_x, min_y, max_x, max_y]


def _iou(g1: ogr.Geometry, g2: ogr.Geometry) -> float:
    if g1 is None or g2 is None:
        return 0.0
    inter = g1.Intersection(g2)
    if inter is None or inter.IsEmpty():
        return 0.0
    union = g1.Union(g2)
    if union is None or union.IsEmpty():
        return 0.0
    return float(inter.Area() / union.Area())


class ImageFootprintIndex:
    def __init__(self, img_paths: List[str], proj_cfg: Dict[str, Any]):
        self.idx = index.Index()
        self.image_files: List[str] = []
        
        print("Building on-the-fly image footprint index...")
        if not img_paths:
            print("Warning: No images found for index building.")
            return
            
        obl_cfg = proj_cfg.get("oblique", {})
        pose_csv = obl_cfg.get("pose_csv", "")
        if not pose_csv or not os.path.exists(pose_csv):
            raise FileNotFoundError(f"pose_csv not found: {pose_csv}")
            
        pose_dict = read_pose_csv(pose_csv)
        
        avg_alt = float(obl_cfg.get("avg_alt", 37.5))
        f = float(obl_cfg.get("focal", 3713.29))
        cx = float(obl_cfg.get("cx", 2647.02))
        cy = float(obl_cfg.get("cy", 1969.28))
        img_width = int(obl_cfg.get("image_width", 5280))
        img_height = int(obl_cfg.get("image_height", 3956))
        
        for p in tqdm(img_paths, desc="Indexing footprints"):
            name = os.path.basename(p)
            params = pose_dict.get(name)
            if not params:
                continue
            
            xs, ys, zs = float(params[0]), float(params[1]), float(params[2])
            phi, omega, kappa = float(params[3]), float(params[4]), float(params[5])
            rot = build_rotation(phi, omega, kappa)
            
            w_exp, h_exp = 0, 0
            x1, y1 = photo_to_ground(-cx - w_exp, cy - h_exp, f, avg_alt, xs, ys, zs, rot)
            x2, y2 = photo_to_ground(-cx - w_exp, cy - img_height + h_exp, f, avg_alt, xs, ys, zs, rot)
            x3, y3 = photo_to_ground(img_width - cx + w_exp, cy + h_exp, f, avg_alt, xs, ys, zs, rot)
            x4, y4 = photo_to_ground(img_width - cx + w_exp, cy - img_height - h_exp, f, avg_alt, xs, ys, zs, rot)
            
            xmin, xmax = min(x1, x2, x3, x4), max(x1, x2, x3, x4)
            ymin, ymax = min(y1, y2, y3, y4), max(y1, y2, y3, y4)
            bbox = (xmin, ymin, xmax, ymax)
            
            self.image_files.append(p)
            self.idx.insert(len(self.image_files) - 1, bbox)
            
    def query(self, bbox: List[float]) -> List[str]:
        hits = list(self.idx.intersection(tuple(bbox)))
        return [self.image_files[i] for i in hits]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to RuntimeConfig YAML")
    parser.add_argument("--merged_shp", required=True)
    parser.add_argument("--gt_shp", required=True)
    parser.add_argument("--output_shp", required=True)
    parser.add_argument("--view_num", type=int, required=True)
    parser.add_argument("--src_field", default="src")
    parser.add_argument("--min_iou_gt_match", type=float, default=0.0)
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    img_paths = resolve_image_paths(cfg.get("data", {}))
    proj_cfg = cfg.get("projection", {})

    driver = ogr.GetDriverByName("ESRI Shapefile")
    merged_ds = ogr.Open(args.merged_shp, 0)
    gt_ds = ogr.Open(args.gt_shp, 0)

    if merged_ds is None:
        raise FileNotFoundError(f"Cannot open {args.merged_shp}")
    if gt_ds is None:
        raise FileNotFoundError(f"Cannot open {args.gt_shp}")

    merged_layer = merged_ds.GetLayer(0)
    gt_layer = gt_ds.GetLayer(0)

    merged_features = []
    for feat in merged_layer:
        geom = feat.GetGeometryRef()
        if not geom or geom.IsEmpty():
            continue
        merged_features.append({
            "fid": feat.GetFID(),
            "geom": geom.Clone(),
            "bbox": _bbox_from_geom(geom),
            "src": _normalize_src(feat.GetField(args.src_field))
        })

    gt_features = []
    for feat in gt_layer:
        geom = feat.GetGeometryRef()
        if not geom or geom.IsEmpty():
            continue
        gt_features.append({
            "fid": feat.GetFID(),
            "geom": geom.Clone(),
            "bbox": _bbox_from_geom(geom),
        })

    gt_rtree = index.Index()
    for i, item in enumerate(gt_features):
        gt_rtree.insert(i, tuple(item["bbox"]))

    img_index = ImageFootprintIndex(img_paths, proj_cfg)
    rng = random.Random(args.random_seed)

    selected_views_by_gt: Dict[int, Set[str]] = {}
    for gt_item in tqdm(gt_features, desc="GT View Selection"):
        gt_fid = gt_item["fid"]
        candidate_paths = img_index.query(gt_item["bbox"])
        candidate_views = {_normalize_src(p) for p in candidate_paths if _normalize_src(p)}

        if not candidate_views:
            selected_views_by_gt[gt_fid] = set()
            continue

        if len(candidate_views) <= args.view_num:
            selected_views_by_gt[gt_fid] = set(candidate_views)
        else:
            selected_views_by_gt[gt_fid] = set(rng.sample(sorted(candidate_views), k=args.view_num))

    selected_merged_fids: Set[int] = set()
    for m_item in tqdm(merged_features, desc="Merged Filtering"):
        m_geom = m_item["geom"]
        src = m_item["src"]
        if not src:
            continue

        best_gt_fid = None
        best_iou = 0.0
        for gt_idx in gt_rtree.intersection(tuple(m_item["bbox"])):
            gt_item = gt_features[gt_idx]
            if not m_geom.Intersects(gt_item["geom"]):
                continue
            iou = _iou(m_geom, gt_item["geom"])
            if iou > best_iou:
                best_iou = iou
                best_gt_fid = gt_item["fid"]

        if best_gt_fid is None or best_iou < args.min_iou_gt_match:
            continue

        if src in selected_views_by_gt.get(best_gt_fid, set()):
            selected_merged_fids.add(m_item["fid"])

    if os.path.exists(args.output_shp):
        driver.DeleteDataSource(args.output_shp)
    out_ds = driver.CreateDataSource(args.output_shp)
    out_layer = out_ds.CreateLayer("selected_views", srs=merged_layer.GetSpatialRef(), geom_type=merged_layer.GetGeomType())

    merged_defn = merged_layer.GetLayerDefn()
    for i in range(merged_defn.GetFieldCount()):
        out_layer.CreateField(merged_defn.GetFieldDefn(i))

    merged_layer.ResetReading()
    written = 0
    for feat in merged_layer:
        if feat.GetFID() not in selected_merged_fids:
            continue
        out_feat = ogr.Feature(out_layer.GetLayerDefn())
        out_feat.SetGeometry(feat.GetGeometryRef().Clone())
        for i in range(merged_defn.GetFieldCount()):
            name = merged_defn.GetFieldDefn(i).GetName()
            out_feat.SetField(name, feat.GetField(name))
        out_layer.CreateFeature(out_feat)
        written += 1

    print(f"Selected {written} features from {len(merged_features)}.")


if __name__ == "__main__":
    main()
