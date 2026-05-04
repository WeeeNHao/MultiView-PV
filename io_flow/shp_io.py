from __future__ import annotations

import os
import time
from typing import Any, List, Optional, Sequence, Tuple

from osgeo import ogr
from tqdm import tqdm

from utils.common import Feature, FeatureList


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _flat_to_pairs(flat_xy: Sequence[float]) -> List[Tuple[float, float]]:
    if len(flat_xy) < 6 or len(flat_xy) % 2 != 0:
        return []
    return [(float(flat_xy[i]), float(flat_xy[i + 1])) for i in range(0, len(flat_xy), 2)]


def _segmentation_to_polygon(segmentation: Any) -> Optional[ogr.Geometry]:
    if not isinstance(segmentation, list) or not segmentation:
        return None

    poly = ogr.Geometry(ogr.wkbPolygon)
    for ring_data in segmentation:
        pts = _flat_to_pairs(ring_data)
        if len(pts) < 3:
            continue

        ring = ogr.Geometry(ogr.wkbLinearRing)
        for x, y in pts:
            ring.AddPoint(x, y)
        if pts[0] != pts[-1]:
            ring.AddPoint(pts[0][0], pts[0][1])

        if ring.GetPointCount() >= 4:
            poly.AddGeometry(ring)

    if poly.IsEmpty():
        return None
    return poly


def _bbox_to_polygon(bbox: Sequence[float]) -> Optional[ogr.Geometry]:
    if len(bbox) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox]
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(x1, y1)
    ring.AddPoint(x2, y1)
    ring.AddPoint(x2, y2)
    ring.AddPoint(x1, y2)
    ring.AddPoint(x1, y1)
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    return poly


def export_features_to_shapefile(
    features: FeatureList,
    out_shp: str,
    projection_wkt: Optional[str] = None,
) -> None:
    _ensure_parent_dir(out_shp)

    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(out_shp):
        driver.DeleteDataSource(out_shp)

    ds = driver.CreateDataSource(out_shp)
    if ds is None:
        raise RuntimeError(f"Cannot create output shapefile: {out_shp}")

    layer = ds.CreateLayer("pv", srs=None, geom_type=ogr.wkbPolygon)

    fields = [
        ("id", ogr.OFTInteger),
        ("label", ogr.OFTInteger),
        ("src", ogr.OFTString),
        ("con", ogr.OFTReal),
        ("con_sem", ogr.OFTReal),
        ("con_pv", ogr.OFTReal),
        ("con_weight", ogr.OFTReal),
        ("area", ogr.OFTReal),
        ("aspect_rat", ogr.OFTReal),
        ("area_sc", ogr.OFTReal),
        ("ratio_sc", ogr.OFTReal),
        ("shape_sc", ogr.OFTReal),
    ]
    for name, ftype in fields:
        layer.CreateField(ogr.FieldDefn(name, ftype))

    for idx, feature in enumerate(tqdm(features, desc="Writing shapefile", leave=False, position=2)):
        geom = _segmentation_to_polygon(feature.get("segmentation"))
        if geom is None:
            geom = _bbox_to_polygon(feature.get("bbox", []))
        if geom is None or geom.IsEmpty():
            continue

        out_feat = ogr.Feature(layer.GetLayerDefn())
        out_feat.SetGeometry(geom)
        out_feat.SetField("id", idx + 1)
        out_feat.SetField("label", int(feature.get("label", 0)))
        out_feat.SetField("src", str(feature.get("src", ""))[:240])

        con_sem = float(feature.get("con_sem", feature.get("score", 0.0)))
        con_pv = float(feature.get("con_pv", 0.0))
        con_weight = float(feature.get("con_weight", con_sem))

        out_feat.SetField("con", con_weight)
        out_feat.SetField("con_sem", con_sem)
        out_feat.SetField("con_pv", con_pv)
        out_feat.SetField("con_weight", con_weight)
        out_feat.SetField("area", float(feature.get("area", 0.0)))
        out_feat.SetField("aspect_rat", float(feature.get("aspect_ratio", 0.0)))
        out_feat.SetField("area_sc", float(feature.get("area_score", 0.0)))
        out_feat.SetField("ratio_sc", float(feature.get("ratio_score", 0.0)))
        out_feat.SetField("shape_sc", float(feature.get("shape_score", 0.0)))

        layer.CreateFeature(out_feat)
        out_feat = None

    ds = None

    if projection_wkt:
        prj = os.path.splitext(out_shp)[0] + ".prj"
        with open(prj, "w", encoding="utf-8") as f:
            f.write(projection_wkt)


def _geom_to_segmentation(geom: ogr.Geometry) -> List[List[float]]:
    out: List[List[float]] = []
    if geom is None or geom.IsEmpty():
        return out

    gtype = geom.GetGeometryType()
    if gtype in (ogr.wkbPolygon, ogr.wkbPolygon25D):
        for i in range(geom.GetGeometryCount()):
            ring = geom.GetGeometryRef(i)
            if ring is None:
                continue
            flat: List[float] = []
            point_count = ring.GetPointCount()
            for pidx in range(point_count):
                flat.append(float(ring.GetX(pidx)))
                flat.append(float(ring.GetY(pidx)))
            if len(flat) >= 6:
                out.append(flat)
        return out

    if gtype in (ogr.wkbMultiPolygon, ogr.wkbMultiPolygon25D):
        for i in range(geom.GetGeometryCount()):
            sub = geom.GetGeometryRef(i)
            out.extend(_geom_to_segmentation(sub))
        return out

    return out


def read_features_from_shapefile(
    shp_path: str,
    score_field: str = "con_weight",
    sem_field: str = "con_sem",
    pv_field: str = "con_pv",
    label_field: str = "label",
    src_field: str = "src",
) -> FeatureList:
    ds = ogr.Open(shp_path, 0)
    if ds is None:
        raise FileNotFoundError(f"Cannot open shapefile: {shp_path}")

    layer = ds.GetLayer(0)
    defn = layer.GetLayerDefn()
    field_names = {defn.GetFieldDefn(i).GetName() for i in range(defn.GetFieldCount())}

    feature_count = layer.GetFeatureCount()
    features: FeatureList = []
    for feat in tqdm(layer, total=feature_count, desc="Reading shapefile", leave=False):
        geom = feat.GetGeometryRef()
        if geom is None or geom.IsEmpty():
            continue
        seg = _geom_to_segmentation(geom)
        env = geom.GetEnvelope()
        bbox = [float(env[0]), float(env[2]), float(env[1]), float(env[3])]

        con_sem = float(feat.GetField(sem_field)) if sem_field in field_names else 0.0
        con_pv = float(feat.GetField(pv_field)) if pv_field in field_names else 0.0
        if score_field in field_names:
            con_weight = float(feat.GetField(score_field))
        elif "con" in field_names:
            con_weight = float(feat.GetField("con"))
        else:
            con_weight = con_sem

        label = int(feat.GetField(label_field)) if label_field in field_names else 0
        src = str(feat.GetField(src_field)) if src_field in field_names else ""

        features.append(
            {
                "bbox": bbox,
                "segmentation": seg,
                "geom": geom.Clone(),
                "label": label,
                "src": src,
                "con_sem": con_sem,
                "con_pv": con_pv,
                "con_weight": con_weight,
                "score": con_weight,
                "area": float(geom.Area()),
            }
        )

    ds = None
    return features