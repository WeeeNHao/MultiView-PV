#!/usr/bin/env python
"""Static preflight checks for one station before launching the experiments.

Verifies everything that would otherwise fail hours into a run: input files,
SAM3 weights, CRS agreement between DSM / GT / the configured EPSG, and pose
coverage of the images actually on disk. Also prints the nadir/oblique
direction split that table 7 depends on.

Exit code is non-zero if any FAIL-level check trips; WARN-level findings are
reported but do not block.

    python scripts/exp_preflight.py --station 004-CangFang \
        --config configs/CangFang/oblique_views.yaml \
        --data-root /data/dataset/PV/004-CangFang/
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from osgeo import gdal, ogr, osr  # noqa: E402

gdal.UseExceptions()
ogr.DontUseExceptions()

from utils.config import load_runtime_config  # noqa: E402

# Images tilted less than this are treated as nadir; the rest are oblique and
# get bucketed into four azimuths by Kappa. Measured clusters sit at ~0 deg and
# ~20-45 deg, so the boundary is not sensitive.
NADIR_TILT_DEG = 10.0

_FAILURES: List[str] = []
_WARNINGS: List[str] = []


def ok(msg: str) -> None:
    print(f"  [ OK ] {msg}")


def warn(msg: str) -> None:
    _WARNINGS.append(msg)
    print(f"  [WARN] {msg}")


def fail(msg: str) -> None:
    _FAILURES.append(msg)
    print(f"  [FAIL] {msg}")


def _epsg_of_raster(path: str) -> Optional[str]:
    ds = gdal.Open(path)
    srs = osr.SpatialReference(wkt=ds.GetProjection())
    return srs.GetAttrValue("AUTHORITY", 1)


def _epsg_of_vector(path: str) -> Tuple[Optional[str], Optional[str], int]:
    ds = ogr.Open(path)
    layer = ds.GetLayer()
    srs = layer.GetSpatialRef()
    code = srs.GetAttrValue("AUTHORITY", 1) if srs else None
    name = srs.GetName() if srs else None
    return code, name, layer.GetFeatureCount()


def check_files(cfg, data_root: str, image_glob: str, station: str) -> List[str]:
    print("\n[1] input files")

    model_cfg = cfg.get("model", {})
    for key in ("checkpoint_path", "bpe_path"):
        path = str(model_cfg.get(key, ""))
        if path and os.path.isfile(path):
            ok(f"model.{key}: {path}")
        else:
            fail(f"model.{key} missing: {path!r}")

    oblique = cfg.get("projection", {}).get("oblique", {})
    for key in ("pose_csv", "dsm_path"):
        path = str(oblique.get(key, ""))
        if path and os.path.isfile(path):
            ok(f"projection.oblique.{key}: {path}")
        else:
            fail(f"projection.oblique.{key} missing: {path!r}")

    images = sorted(glob.glob(os.path.join(data_root, image_glob)))
    if images:
        ok(f"images: {len(images)} matched {image_glob}")
    else:
        fail(f"no images matched {os.path.join(data_root, image_glob)}")

    gt = f"/data/dataset/PV/{station}/gt/pv.shp"
    if os.path.isfile(gt):
        ok(f"ground truth: {gt}")
    else:
        fail(f"ground truth missing: {gt}")

    dom = os.path.join(data_root, "dom", "DOM.tif")
    if os.path.isfile(dom):
        ok(f"DOM: {dom}")
    else:
        warn(f"DOM missing ({dom}) -- tables 1/2 M1 and table 5 need it")

    return images


def check_crs(cfg, data_root: str, station: str) -> None:
    """DSM, GT and the configured EPSG must agree.

    The oblique path has no geotransform to inherit a CRS from, so it stamps
    outputs with projection.oblique.epsg -- which silently defaults to 4550.
    A mismatch here produces shapefiles whose .prj lies about the coordinates.
    """
    print("\n[2] CRS agreement")

    oblique = cfg.get("projection", {}).get("oblique", {})
    cfg_epsg = str(oblique.get("epsg", 4550))

    dsm = str(oblique.get("dsm_path", ""))
    dsm_epsg = _epsg_of_raster(dsm) if os.path.isfile(dsm) else None

    gt = f"/data/dataset/PV/{station}/gt/pv.shp"
    gt_epsg, gt_name, gt_n = _epsg_of_vector(gt) if os.path.isfile(gt) else (None, None, 0)

    print(f"       config epsg={cfg_epsg}  DSM={dsm_epsg}  GT={gt_epsg} ({gt_name})")

    if dsm_epsg and dsm_epsg != cfg_epsg:
        fail(
            f"projection.oblique.epsg={cfg_epsg} but DSM is EPSG:{dsm_epsg}. "
            f"Set epsg: {dsm_epsg} in the station base config."
        )
    elif dsm_epsg:
        ok(f"config EPSG matches DSM ({cfg_epsg})")

    if gt_epsg and dsm_epsg and gt_epsg != dsm_epsg:
        # A bare unit code (9001) means the .prj carries an ESRI-style name with
        # no EPSG authority; compare names instead of blocking on it.
        if gt_name and dsm_epsg and str(gt_name).replace("_", " ").lower() not in ("", "unknown"):
            warn(f"GT authority code is {gt_epsg}, not {dsm_epsg} -- name is {gt_name!r}, "
                 "coordinates are likely fine but verify the overlay in QGIS")
        else:
            fail(f"GT EPSG:{gt_epsg} != DSM EPSG:{dsm_epsg}")
    elif gt_epsg:
        ok(f"GT matches DSM ({gt_epsg}), {gt_n} features")


def check_poses(cfg, images: List[str]) -> None:
    print("\n[3] pose coverage and camera directions")

    pose_csv = str(cfg.get("projection", {}).get("oblique", {}).get("pose_csv", ""))
    if not os.path.isfile(pose_csv) or not images:
        warn("skipped (missing pose.csv or images)")
        return

    with open(pose_csv, newline="", encoding="utf-8") as f:
        poses: Dict[str, dict] = {r["Image Name"]: r for r in csv.DictReader(f)}

    names = [os.path.basename(p) for p in images]
    matched = [n for n in names if n in poses]
    missing = len(names) - len(matched)

    if missing:
        fail(f"{missing}/{len(names)} images have no pose entry")
    else:
        ok(f"all {len(names)} images have poses ({len(poses)} rows in pose.csv)")

    def tilt_deg(row: dict) -> float:
        return math.degrees(math.hypot(float(row["Phi"]), float(row["Omega"])))

    nadir = [n for n in matched if tilt_deg(poses[n]) < NADIR_TILT_DEG]
    oblique_imgs = [n for n in matched if tilt_deg(poses[n]) >= NADIR_TILT_DEG]

    buckets = Counter()
    for n in oblique_imgs:
        kappa = math.degrees(float(poses[n]["Kappa"]))
        buckets[int(math.floor(kappa / 45.0)) * 45] += 1

    print(f"       nadir={len(nadir)}  oblique={len(oblique_imgs)}")
    print(f"       oblique azimuth buckets: {dict(sorted(buckets.items()))}")

    if len(nadir) == 0:
        fail("no nadir images -- table 7 needs a nadir subset")
    elif len(buckets) < 4:
        warn(f"only {len(buckets)} oblique azimuth clusters (table 7 assumes 4)")
    else:
        ok("nadir + 4 oblique directions present (table 7 feasible)")



def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--station", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--image-glob", default="images/*.JPG")
    args = p.parse_args(argv)

    print(f"=== preflight: {args.station} ===")
    print(f"    config    : {args.config}")
    print(f"    data root : {args.data_root}")

    cfg = load_runtime_config(args.config).raw

    images = check_files(cfg, args.data_root, args.image_glob, args.station)
    check_crs(cfg, args.data_root, args.station)
    check_poses(cfg, images)

    print("\n=== summary ===")
    if _FAILURES:
        print(f"  {len(_FAILURES)} FAIL, {len(_WARNINGS)} WARN")
        for m in _FAILURES:
            print(f"    FAIL: {m}")
        return 1
    print(f"  all checks passed ({len(_WARNINGS)} WARN)")
    for m in _WARNINGS:
        print(f"    WARN: {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
