#!/usr/bin/env python
"""Search the pixel-vote baseline (M2) for its best achievable configuration.

Why this exists: M2 reports obj_F1 = 0.0000 in the main tables, which reads as a
strawman unless we can show it is the *best* the method can do. The default
``cell_size = 0.2 m`` is provably hopeless -- GT modules are 1.10 x 2.20 m with a
median inter-module gap of 5.9 cm (BeiOu) / 3.8 cm (XinXie), so 100% of gaps are
narrower than one cell and adjacent modules cannot be separated at all. This
sweep asks whether a finer grid plus a stricter vote threshold recovers
instances, before we claim pixel voting cannot.

Run it on BeiOu: it is the only station whose projected footprints are free of
the oversized-affine tail (0.0% above 2x module area, vs 20.2% at XinXie and
22.1% at CangFang). Those outsized polygons bridge the gaps with full vote
support, so a sweep there cannot separate "too coarse" from "bad projection".

Faithfulness: the real pipeline applies ``per_image_nms`` while collecting the
per-image shapefiles (pipeline.py), and only the deduplicated features reach the
vote grid -- BeiOu's own run logs 150610 -> 60062 at XinXie. The on-disk
per-image files are pre-NMS, so reading them raw would let sliding-window
duplicates of a single view satisfy ``min_votes`` on their own and silently
change what the threshold means. This script therefore replicates that NMS.

    python scripts/sweep_pixel_vote.py --station 001-BeiOu
    python scripts/sweep_pixel_vote.py --station 001-BeiOu \
        --cell-sizes 0.05,0.02 --min-votes 2,4,6 --out sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("TQDM_DISABLE", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from osgeo import gdal, ogr  # noqa: E402

ogr.DontUseExceptions()
gdal.UseExceptions()

from evaluation.geom import area_overlap_metrics, dissolve_features  # noqa: E402
from evaluation.metrics import evaluate_feature_lists  # noqa: E402
from utils.config_loader import load_config  # noqa: E402
from io_flow.shp_io import (  # noqa: E402
    export_features_to_shapefile,
    read_features_from_shapefile,
)
from postprocess.nms import nms_features  # noqa: E402
from postprocess.pixel_fusion import (  # noqa: E402
    MAX_GRID_CELLS,
    _accumulate_votes,
    _geom_to_bbox,
    _geom_to_segmentation,
    _polygonize,
    robust_bounds,
)

# Beyond this many predictions the object-level matcher is too slow to be worth
# it, and the configuration is hopeless anyway -- report counts only.
EVAL_PRED_CAP = 20000


def load_features(proj_dir: str, nms_iou: float, use_geometry_iou: bool) -> list:
    """Read per-image projected shapefiles, applying the pipeline's per-image NMS."""
    files = sorted(
        os.path.join(proj_dir, f)
        for f in os.listdir(proj_dir)
        if f.endswith(".shp")
    )
    if not files:
        raise FileNotFoundError(f"no shapefiles under {proj_dir}")

    before = after = 0
    out: list = []
    for i, path in enumerate(files, 1):
        feats = read_features_from_shapefile(shp_path=path)
        before += len(feats)
        if nms_iou > 0:
            feats = nms_features(
                features=feats,
                score_field="con_weight",
                iou_threshold=nms_iou,
                use_geometry_iou=use_geometry_iou,
                backend="auto",
            )
        after += len(feats)
        out.extend(feats)
        if i % 50 == 0 or i == len(files):
            print(f"  [load] {i}/{len(files)} files  raw={before} kept={after}", flush=True)
    print(f"  [load] per_image_nms: {before} -> {after} "
          f"({(before - after) / max(before, 1) * 100:.1f}% removed)", flush=True)
    return out


def vote_grid(features: list, cell_size: float):
    """Accumulate the vote grid once; every min_votes reuses it.

    Mirrors fuse_pixel_vote: robust bounds first (one 295 km outlier at XinXie
    is enough to demand a 306 TiB array here), then the MAX_GRID_CELLS guard.
    """
    b, features, dropped = robust_bounds(features)
    if b is None:
        return None, None, None
    if dropped:
        print(f"  [grid] dropped {dropped} coordinate outlier(s)", flush=True)
    minx, miny, maxx, maxy = b
    minx -= cell_size
    miny -= cell_size
    maxx += cell_size
    maxy += cell_size
    bounds = (minx, miny, maxx, maxy)
    width = int(np.ceil((maxx - minx) / cell_size))
    height = int(np.ceil((maxy - miny) / cell_size))
    if float(width) * float(height) > MAX_GRID_CELLS:
        # A sweep that silently coarsens its own cell size measures nothing, so
        # refuse rather than mirror the production fallback here.
        raise MemoryError(
            f"cell_size={cell_size} over extent {maxx - minx:.0f}x{maxy - miny:.0f} m "
            f"needs {width * height / 1e9:.1f}e9 cells (> MAX_GRID_CELLS). "
            "Bounds are still contaminated -- inspect the projection output."
        )
    votes = _accumulate_votes(
        features=features,
        bounds=bounds,
        cell_size=cell_size,
        width=width,
        height=height,
        score_field="con_weight",
        use_score_weight=False,
    )
    return votes, bounds, (width, height)


def instances_from_mask(mask, bounds, cell_size, min_area: float) -> list:
    geoms = _polygonize(mask, bounds, cell_size)
    out = []
    for idx, geom in enumerate(geoms):
        area = float(geom.Area())
        if area < min_area:
            continue
        out.append({
            "id": idx + 1,
            "geom": geom,
            "segmentation": _geom_to_segmentation(geom),
            "bbox": _geom_to_bbox(geom),
            "label": 1,
            "src": "pixel_vote",
            "area": area,
            "con_sem": 0.0,
            "con_pv": 0.0,
            "con_weight": 1.0,
            "score": 1.0,
        })
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--station", default="001-BeiOu")
    p.add_argument("--zs-root", default="/data/dataset/PV/ZS_PV")
    p.add_argument("--variant", default="m2",
                   help="which baseline's shared_proj to sweep (m2 or m3)")
    p.add_argument("--cell-sizes", default="0.2,0.1,0.05,0.02")
    p.add_argument("--min-votes", default="2,3,4,6,8")
    p.add_argument("--min-areas", default="0.0,0.6",
                   help="drop components below this area (m^2); 0.6 ~ 25%% of a module")
    p.add_argument("--nms-iou", type=float, default=None,
                   help="per-image NMS IoU. Default: read from the station's own "
                        "oblique_views.yaml. Do not hardcode -- BeiOu uses 0.25 "
                        "while XinXie and CangFang use 0.2, and assuming one "
                        "value leaves ~3%% more polygons in the vote grid than "
                        "production has, which shifts the tuned min_votes.")
    p.add_argument("--nms-geometry-iou", action="store_true",
                   help="oblique_views.yaml uses bbox IoU; set this to compare")
    p.add_argument("--out", default=None)
    p.add_argument("--export-dir", default=None,
                   help="Also write each configuration's polygons as a shapefile "
                        "here, so the result can be inspected in QGIS. The sweep "
                        "otherwise produces metrics only.")
    args = p.parse_args(argv)

    # The sweep must reproduce the pipeline's own per-image NMS exactly; a
    # different threshold changes how many polygons reach the vote grid and
    # therefore where the min_votes optimum sits.
    nms_iou = args.nms_iou
    if nms_iou is None:
        station_cfg = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", args.station.split("-", 1)[1], "oblique_views.yaml")
        try:
            cfg = load_config(station_cfg)
            nms_iou = float(cfg.postprocess.per_image_nms.iou_threshold)
            print(f"[sweep] per_image_nms.iou_threshold={nms_iou} (from {station_cfg})")
        except Exception as exc:  # noqa: BLE001
            nms_iou = 0.25
            print(f"[sweep] !! could not read {station_cfg} ({exc}); "
                  f"falling back to {nms_iou}", file=sys.stderr)

    proj_dir = os.path.join(args.zs_root, f"{args.station}-exp2",
                            args.variant, "shared_proj")
    gt_path = f"/data/dataset/PV/{args.station}/gt/pv.shp"
    out_csv = args.out or os.path.join(args.zs_root, "eval_exp2",
                                       f"pixel_vote_sweep_{args.station}.csv")

    cells = [float(x) for x in args.cell_sizes.split(",") if x.strip()]
    votes_list = [float(x) for x in args.min_votes.split(",") if x.strip()]
    areas_list = [float(x) for x in args.min_areas.split(",") if x.strip()]

    print(f"[sweep] station={args.station} variant={args.variant}")
    print(f"[sweep] proj_dir={proj_dir}")
    print(f"[sweep] cells={cells} min_votes={votes_list} min_areas={areas_list}")

    # Carry the CRS onto the exports, otherwise QGIS loads them unreferenced and
    # they will not overlay the imagery.
    proj_wkt = None
    if args.export_dir:
        os.makedirs(args.export_dir, exist_ok=True)
        first = next((os.path.join(proj_dir, f) for f in sorted(os.listdir(proj_dir))
                      if f.endswith(".shp")), None)
        if first:
            ds = ogr.Open(first)
            if ds is not None:
                sr = ds.GetLayer(0).GetSpatialRef()
                if sr is not None:
                    proj_wkt = sr.ExportToWkt()
            ds = None
        print(f"[sweep] exporting shapefiles to {args.export_dir} "
              f"(CRS {'from input' if proj_wkt else 'MISSING'})")

    gts = read_features_from_shapefile(shp_path=gt_path, score_field="con_weight")
    gt_union = dissolve_features(gts)
    print(f"[sweep] GT features = {len(gts)}", flush=True)

    feats = load_features(proj_dir, nms_iou, args.nms_geometry_iou)
    print(f"[sweep] features into the vote grid = {len(feats)}", flush=True)

    rows: List[dict] = []
    for cell in cells:
        t0 = time.time()
        votes, bounds, wh = vote_grid(feats, cell)
        if votes is None:
            print(f"[sweep] cell={cell}: no bounds, skip")
            continue
        print(f"[sweep] cell={cell:.3f} grid={wh[0]}x{wh[1]} "
              f"({wh[0] * wh[1] / 1e6:.1f}M cells) accumulated in {time.time() - t0:.1f}s",
              flush=True)

        for mv in votes_list:
            mask = votes >= mv
            if not mask.any():
                print(f"    min_votes={mv}: empty mask")
                continue
            t1 = time.time()
            geoms_all = instances_from_mask(mask, bounds, cell, 0.0)
            poly_s = time.time() - t1

            for ma in areas_list:
                preds = [f for f in geoms_all if f["area"] >= ma]
                if args.export_dir and preds:
                    name = (f"pv_{args.station}_{args.variant}"
                            f"_c{cell:g}_mv{mv:g}_a{ma:g}.shp")
                    export_features_to_shapefile(
                        features=preds,
                        out_shp=os.path.join(args.export_dir, name),
                        projection_wkt=proj_wkt,
                    )
                row = {
                    "station": args.station, "variant": args.variant,
                    "cell_size": cell, "min_votes": mv, "min_area": ma,
                    "n_pred": len(preds), "n_gt": len(gts),
                }
                if 0 < len(preds) <= EVAL_PRED_CAP:
                    r = evaluate_feature_lists(preds=preds, gts=gts,
                                               primary_iou=0.5,
                                               score_field="con_weight")
                    pred_union = dissolve_features(preds)
                    a = area_overlap_metrics(pred_union, gt_union)
                    row.update({
                        "obj_Prec": f"{r.precision:.4f}", "obj_Rec": f"{r.recall:.4f}",
                        "obj_F1": f"{r.f1:.4f}", "AP95": f"{r.ap95:.4f}",
                        "AP975": f"{r.ap975:.4f}", "obj_mIoU": f"{r.mean_matched_iou:.4f}",
                        "area_IoU": f"{a['area_iou']:.4f}",
                        "over_seg": r.over_seg_count, "under_seg": r.under_seg_count,
                    })
                    print(f"    cell={cell:<5} mv={mv:<4} minA={ma:<4} "
                          f"n={len(preds):<7} F1={r.f1:.4f} areaIoU={a['area_iou']:.4f} "
                          f"over/under={r.over_seg_count}/{r.under_seg_count} "
                          f"({poly_s:.0f}s poly)", flush=True)
                else:
                    row.update({k: "" for k in
                                ("obj_Prec", "obj_Rec", "obj_F1", "AP95", "AP975",
                                 "obj_mIoU", "area_IoU", "over_seg", "under_seg")})
                    print(f"    cell={cell:<5} mv={mv:<4} minA={ma:<4} "
                          f"n={len(preds):<7} (skipped eval, >{EVAL_PRED_CAP})", flush=True)
                rows.append(row)

        del votes

    if not rows:
        print("[sweep] nothing produced", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    fields = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[sweep] -> {out_csv} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
