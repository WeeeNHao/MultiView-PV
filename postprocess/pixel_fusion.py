"""Pixel-level object-space fusion (draft table 0, rows M2 and M3).

The instance-level path (``postprocess/merge.py``) keeps whole candidate
polygons and resolves overlap by association / NMS. The baselines it is
compared against fuse in raster space instead:

  * **M2 -- pixel voting.** Every projected candidate from every perspective
    view burns into a shared object-space grid. Cells carrying enough support
    survive; connected components of the surviving mask become instances.
  * **M3 -- pixel-level late fusion.** Same grid, but the TDOM detections are
    burned in alongside the perspective ones, so the two sources are combined
    as masks rather than as objects.

Both drop the module-geometry prior: cells are weighted by raw SAM~3
confidence (or counted uniformly), never by the area / rectangularity /
aspect-ratio scores that the proposed method relies on.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from osgeo import gdal, ogr

from utils.common import FeatureList

gdal.UseExceptions()

_LOG = logging.getLogger(__name__)

# Cap on grid cells (~1.5 GB as float32). A whole station at a few centimetres
# would otherwise allocate tens of gigabytes; the cell size is widened instead.
MAX_GRID_CELLS = 400_000_000


def _feature_geometry(feature: Dict[str, Any]) -> Optional[ogr.Geometry]:
    geom = feature.get("geom")
    if geom is not None:
        return geom

    seg = feature.get("segmentation")
    if not seg:
        return None

    poly = ogr.Geometry(ogr.wkbPolygon)
    for ring_data in seg if isinstance(seg, list) else []:
        vals = [float(v) for v in ring_data]
        if len(vals) < 6 or len(vals) % 2 != 0:
            continue
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for i in range(0, len(vals), 2):
            ring.AddPoint(vals[i], vals[i + 1])
        if (vals[0], vals[1]) != (vals[-2], vals[-1]):
            ring.AddPoint(vals[0], vals[1])
        if ring.GetPointCount() >= 4:
            poly.AddGeometry(ring)

    if poly.IsEmpty():
        return None
    feature["geom"] = poly
    return poly


def _bounds(features: FeatureList) -> Optional[Tuple[float, float, float, float]]:
    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    for feat in features:
        geom = _feature_geometry(feat)
        if geom is None:
            continue
        x0, x1, y0, y1 = geom.GetEnvelope()
        minx, maxx = min(minx, x0), max(maxx, x1)
        miny, maxy = min(miny, y0), max(maxy, y1)
    if minx > maxx or miny > maxy:
        return None
    return minx, miny, maxx, maxy


def robust_bounds(
    features: FeatureList,
    iqr_k: float = 10.0,
) -> Tuple[Optional[Tuple[float, float, float, float]], FeatureList, int]:
    """Bounds over the bulk of the features, plus the features inside them.

    Plain min/max is not usable here: a *single* degenerate projection sets the
    extent for the whole grid. At XinXie one feature out of 150 610 (0.001%)
    landed 295 km from the site -- everything else was within 71 m -- which grew
    the extent to 62 x 541 km. The cell size then hit MAX_GRID_CELLS and was
    silently widened from 0.20 m to 9.2 m, larger than a whole PV module, and the
    entire array fused into two connected components. The baseline never ran at
    the configured resolution and nothing in the logs said so.

    Rejection is a Tukey fence on the feature centres, [Q1 - k*IQR, Q3 + k*IQR]
    per axis, with a deliberately loose k. A percentile trim does not work here:
    with one outlier in 150 610 the 99.95th percentile still interpolates most of
    the way to it. The IQR is set by the bulk of the data and is unmoved by any
    number of extreme points, so the fence holds however few outliers there are.

    k=10 is far wider than any real site -- XinXie's modules span 122 m, giving a
    fence of roughly +-600 m -- so this only ever rejects gross failures, never
    the edge of a legitimate array.

    Returns (bounds, kept_features, n_dropped). ``iqr_k <= 0`` restores the old
    min/max behaviour.
    """
    entries = []
    for feat in features:
        geom = _feature_geometry(feat)
        if geom is None:
            continue
        x0, x1, y0, y1 = geom.GetEnvelope()
        entries.append((feat, x0, x1, y0, y1))

    if not entries:
        return None, [], 0
    if iqr_k <= 0.0 or len(entries) < 20:
        b = _bounds([e[0] for e in entries])
        return b, [e[0] for e in entries], 0

    cx = np.array([(e[1] + e[2]) * 0.5 for e in entries])
    cy = np.array([(e[3] + e[4]) * 0.5 for e in entries])

    def _fence(v: np.ndarray) -> Tuple[float, float]:
        q1, q3 = np.percentile(v, [25.0, 75.0])
        # Floor the spread so a degenerate (all-identical) axis cannot collapse
        # the fence onto a single coordinate.
        spread = max(float(q3 - q1), 1.0)
        return float(q1) - iqr_k * spread, float(q3) + iqr_k * spread

    lo_x, hi_x = _fence(cx)
    lo_y, hi_y = _fence(cy)

    kept: FeatureList = []
    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    dropped = 0
    for (feat, x0, x1, y0, y1), ex, ey in zip(entries, cx, cy):
        if not (lo_x <= ex <= hi_x and lo_y <= ey <= hi_y):
            dropped += 1
            continue
        kept.append(feat)
        minx, maxx = min(minx, x0), max(maxx, x1)
        miny, maxy = min(miny, y0), max(maxy, y1)

    if not kept:
        return None, [], dropped
    return (minx, miny, maxx, maxy), kept, dropped


def _accumulate_votes(
    features: FeatureList,
    bounds: Tuple[float, float, float, float],
    cell_size: float,
    width: int,
    height: int,
    score_field: str,
    use_score_weight: bool,
) -> np.ndarray:
    """Burn every polygon into the grid, summing support per cell."""
    minx, miny, _, maxy = bounds
    geotransform = (minx, cell_size, 0.0, maxy, 0.0, -cell_size)

    votes = np.zeros((height, width), dtype=np.float32)

    # Rasterising one polygon at a time would mean thousands of GDAL calls, so
    # features are grouped by burn value and each group is burned in one pass.
    # With uniform weights that collapses to a single pass.
    groups: Dict[float, List[ogr.Geometry]] = {}
    for feat in features:
        geom = _feature_geometry(feat)
        if geom is None:
            continue
        if use_score_weight:
            burn = round(float(feat.get(score_field, 0.0) or 0.0), 3)
        else:
            burn = 1.0
        if burn <= 0.0:
            continue
        groups.setdefault(burn, []).append(geom)

    driver = gdal.GetDriverByName("MEM")
    mem_drv = ogr.GetDriverByName("Memory")

    for burn, geoms in groups.items():
        ras = driver.Create("", width, height, 1, gdal.GDT_Float32)
        ras.SetGeoTransform(geotransform)

        src = mem_drv.CreateDataSource("burn")
        layer = src.CreateLayer("burn", geom_type=ogr.wkbPolygon)
        defn = layer.GetLayerDefn()
        for geom in geoms:
            f = ogr.Feature(defn)
            f.SetGeometry(geom)
            layer.CreateFeature(f)
            f = None

        # MERGE_ALG=ADD is what makes this a vote: without it overlapping
        # polygons merely set the cell, so N agreeing views would count once.
        gdal.RasterizeLayer(
            ras, [1], layer, burn_values=[burn], options=["MERGE_ALG=ADD"]
        )
        votes += ras.GetRasterBand(1).ReadAsArray()

        src = None
        ras = None

    return votes


def _polygonize(
    mask: np.ndarray,
    bounds: Tuple[float, float, float, float],
    cell_size: float,
) -> List[ogr.Geometry]:
    """Connected components of a binary mask, as polygons."""
    minx, _, _, maxy = bounds
    height, width = mask.shape

    ras = gdal.GetDriverByName("MEM").Create("", width, height, 1, gdal.GDT_Byte)
    ras.SetGeoTransform((minx, cell_size, 0.0, maxy, 0.0, -cell_size))
    band = ras.GetRasterBand(1)
    band.WriteArray(mask.astype(np.uint8))
    band.SetNoDataValue(0)

    src = ogr.GetDriverByName("Memory").CreateDataSource("poly")
    layer = src.CreateLayer("poly", geom_type=ogr.wkbPolygon)
    layer.CreateField(ogr.FieldDefn("dn", ogr.OFTInteger))

    # The mask itself is the validity mask, so only set cells are emitted.
    gdal.Polygonize(band, band, layer, 0, [], callback=None)

    geoms: List[ogr.Geometry] = []
    for feat in layer:
        if int(feat.GetField("dn")) == 0:
            continue
        geom = feat.GetGeometryRef()
        if geom is not None and not geom.IsEmpty():
            geoms.append(geom.Clone())

    src = None
    ras = None
    return geoms


def fuse_pixel_vote(features: FeatureList, cfg: Dict[str, Any]) -> FeatureList:
    """Raster vote + connected components -> instances.

    Config keys (under ``postprocess.multiview``):
      ``cell_size``        grid resolution in map units (default 0.2 m)
      ``min_votes``        support a cell needs to survive (default 2.0)
      ``use_score_weight`` weight each burn by confidence instead of counting
      ``min_area``         drop components smaller than this (map units^2)
    """
    if not features:
        return []

    cell_size = float(cfg.get("cell_size", 0.2))
    min_votes = float(cfg.get("min_votes", 2.0))
    use_score_weight = bool(cfg.get("use_score_weight", False))
    score_field = str(cfg.get("score_field", "con_weight"))
    min_area = float(cfg.get("min_area", 0.0))

    bounds_iqr_k = float(cfg.get("bounds_iqr_k", 10.0))
    bounds, features, n_dropped = robust_bounds(features, bounds_iqr_k)
    if bounds is None:
        return []
    if n_dropped:
        _LOG.warning(
            "pixel_vote: dropped %d of %d features as coordinate outliers before "
            "gridding (bounds_iqr_k=%.1f); a single one can otherwise widen the "
            "cell past the module size",
            n_dropped, n_dropped + len(features), bounds_iqr_k,
        )

    minx, miny, maxx, maxy = bounds
    # Pad by one cell so polygons on the edge are not clipped.
    minx -= cell_size
    miny -= cell_size
    maxx += cell_size
    maxy += cell_size
    bounds = (minx, miny, maxx, maxy)

    width = int(np.ceil((maxx - minx) / cell_size))
    height = int(np.ceil((maxy - miny) / cell_size))
    if width <= 0 or height <= 0:
        return []

    if width * height > MAX_GRID_CELLS:
        scale = np.sqrt(width * height / MAX_GRID_CELLS)
        requested = cell_size
        cell_size *= scale
        width = int(np.ceil((maxx - minx) / cell_size))
        height = int(np.ceil((maxy - miny) / cell_size))
        # Loud on purpose. This silently rewrote the configured resolution for a
        # whole experimental campaign: XinXie ran at 9.2 m instead of the 0.20 m
        # in its config, and every log and summary still reported 0.20.
        _LOG.warning(
            "pixel_vote: grid over MAX_GRID_CELLS, cell_size widened %.4f -> "
            "%.4f m (extent %.1f x %.1f m). Results are NOT at the configured "
            "resolution.",
            requested, cell_size, maxx - minx, maxy - miny,
        )

    votes = _accumulate_votes(
        features=features,
        bounds=bounds,
        cell_size=cell_size,
        width=width,
        height=height,
        score_field=score_field,
        use_score_weight=use_score_weight,
    )

    mask = votes >= min_votes
    if not mask.any():
        return []

    geoms = _polygonize(mask, bounds, cell_size)

    out: FeatureList = []
    for idx, geom in enumerate(geoms):
        area = float(geom.Area())
        if area < min_area:
            continue
        # Confidence is the mean support inside the component, normalised so a
        # cell at exactly the threshold scores 0 and heavily-voted cores tend
        # to 1. There is no geometry prior here by construction.
        out.append(
            {
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
            }
        )
    return out


def _geom_to_segmentation(geom: ogr.Geometry) -> List[List[float]]:
    rings: List[List[float]] = []
    for i in range(geom.GetGeometryCount()):
        ring = geom.GetGeometryRef(i)
        flat: List[float] = []
        for j in range(ring.GetPointCount()):
            x, y, *_ = ring.GetPoint(j)
            flat.extend([float(x), float(y)])
        if len(flat) >= 6:
            rings.append(flat)
    return rings


def _geom_to_bbox(geom: ogr.Geometry) -> List[float]:
    x0, x1, y0, y1 = geom.GetEnvelope()
    return [float(x0), float(y0), float(x1), float(y1)]
