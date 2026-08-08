#!/usr/bin/env python
"""Group a station's images by camera direction for the table 7 increment study.

Images are classified from pose.csv: tilt = hypot(Phi, Omega) separates nadir
from oblique, and oblique images are then bucketed into four azimuths by Kappa.
The oblique buckets are ordered by a fixed random seed, as the draft requires,
and named O1..O4 in that order.

Two outputs, both driven off the same classification:

  --emit lists  writes one image-list file per cumulative view set, for runs
                that re-infer (table 7 at t=1).
  --emit links  builds a directory of symlinks into an existing per-image
                shapefile cache, so a cumulative view set can be postprocessed
                without re-running inference at all.

    python scripts/split_directions.py --station 004-CangFang \
        --emit links --proj-dir .../iter_0/shared/proj --out-root .../dirs
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import random
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

NADIR_TILT_DEG = 10.0
DEFAULT_SEED = 42

# Cumulative view sets, in the order table 7 lists them. "tdom" carries no
# perspective images at all -- it is the DOM-only row and produces an empty
# image set here by design.
SET_NAMES = ["d0_tdom", "d1_nadir", "d2_o1", "d3_o2", "d4_o3", "d5_o4"]


def classify(pose_csv: str, images: List[str], seed: int) -> Tuple[List[str], List[List[str]]]:
    """Return (nadir images, [O1 images, ..., O4 images])."""
    with open(pose_csv, newline="", encoding="utf-8") as f:
        poses: Dict[str, dict] = {r["Image Name"]: r for r in csv.DictReader(f)}

    nadir: List[str] = []
    by_azimuth: Dict[int, List[str]] = defaultdict(list)

    for path in images:
        row = poses.get(os.path.basename(path))
        if row is None:
            continue
        tilt = math.degrees(math.hypot(float(row["Phi"]), float(row["Omega"])))
        if tilt < NADIR_TILT_DEG:
            nadir.append(path)
        else:
            kappa = math.degrees(float(row["Kappa"]))
            by_azimuth[int(math.floor(kappa / 45.0)) * 45].append(path)

    # Fixed-seed ordering so O1..O4 is reproducible across stations and reruns.
    keys = sorted(by_azimuth)
    random.Random(seed).shuffle(keys)
    return sorted(nadir), [sorted(by_azimuth[k]) for k in keys]


def cumulative_sets(nadir: List[str], obliques: List[List[str]]) -> List[Tuple[str, List[str]]]:
    sets: List[Tuple[str, List[str]]] = [(SET_NAMES[0], [])]
    acc = list(nadir)
    sets.append((SET_NAMES[1], list(acc)))
    for i, group in enumerate(obliques[:4]):
        acc = acc + group
        sets.append((SET_NAMES[2 + i], list(acc)))
    return sets


def _shp_stem_to_image(name: str) -> str:
    """`images_DJI_0001_V__r2.shp` -> `DJI_0001_V.JPG` style stem."""
    stem = os.path.splitext(os.path.basename(name))[0]
    stem = re.sub(r"__r\d+$", "", stem)
    # pipeline._safe_image_name prefixes the parent directory name.
    return stem.split("_", 1)[1] if "_" in stem else stem


def emit_lists(sets, out_root: str) -> None:
    os.makedirs(out_root, exist_ok=True)
    for name, paths in sets:
        path = os.path.join(out_root, f"{name}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(paths) + ("\n" if paths else ""))
        print(f"  {name:10s} {len(paths):5d} images -> {path}")


def emit_links(sets, proj_dir: str, out_root: str) -> None:
    """Symlink the per-image shapefile parts belonging to each view set.

    Postprocess only ever scans per_image_shp_dir for *.shp, so pointing it at
    one of these directories restricts fusion to that view set without touching
    inference.
    """
    shps = sorted(glob.glob(os.path.join(proj_dir, "*.shp")))
    if not shps:
        print(f"no shapefiles under {proj_dir}", file=sys.stderr)
        raise SystemExit(1)

    by_stem: Dict[str, List[str]] = defaultdict(list)
    for shp in shps:
        by_stem[_shp_stem_to_image(shp)].append(shp)

    for name, paths in sets:
        dest = os.path.join(out_root, name, "proj")
        os.makedirs(dest, exist_ok=True)
        linked = 0
        wanted = {os.path.splitext(os.path.basename(p))[0] for p in paths}
        for stem in wanted:
            for shp in by_stem.get(stem, []):
                base = os.path.splitext(shp)[0]
                # A shapefile is a file set; link every sidecar alongside .shp.
                for part in glob.glob(base + ".*"):
                    link = os.path.join(dest, os.path.basename(part))
                    if not os.path.lexists(link):
                        os.symlink(part, link)
                linked += 1
        print(f"  {name:10s} {len(paths):5d} images -> {linked:5d} shp linked  {dest}")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--station", required=True)
    p.add_argument("--data-root", default=None)
    p.add_argument("--image-glob", default="images/*.JPG")
    p.add_argument("--pose-csv", default=None)
    p.add_argument("--emit", choices=["lists", "links"], required=True)
    p.add_argument("--proj-dir", default=None, help="required for --emit links")
    p.add_argument("--out-root", required=True)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = p.parse_args(argv)

    data_root = args.data_root or f"/data/dataset/PV/{args.station}/"
    pose_csv = args.pose_csv or os.path.join(data_root, "CAM", "pose.csv")
    images = sorted(glob.glob(os.path.join(data_root, args.image_glob)))

    nadir, obliques = classify(pose_csv, images, args.seed)
    print(f"=== {args.station}: {len(images)} images, "
          f"nadir={len(nadir)}, oblique groups={[len(g) for g in obliques]} (seed={args.seed})")

    sets = cumulative_sets(nadir, obliques)

    if args.emit == "lists":
        emit_lists(sets, args.out_root)
    else:
        if not args.proj_dir:
            p.error("--emit links requires --proj-dir")
        emit_links(sets, args.proj_dir, args.out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
