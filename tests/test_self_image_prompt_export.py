"""The image-space self re-prompt path (design doc §4.5, table 3 arm 2).

Each image re-prompts SAM~3 with the pixel bboxes of its *own* previous-round
masks -- no object-space fusion, no geometry prior, no cross-view broadcast.
Those per-image results live in ``iter_{t-1}/shared/infer`` under names like
``images_DJI_20251013154527_0001_V__r0.shp``: the image glob's directory is
folded into the stem and the writing rank is appended.

``inference/window_dataset.py`` looks prompts up as ``<image stem>.txt``. If the
exporter writes the *shapefile's* stem instead, not one prompt matches and
nothing raises -- every image silently falls back to text-prompt-only
inference, which is exactly the t=0 behaviour. The ablation arm would then
measure nothing while looking like it ran, and the table would report a number
that means "no feedback" under a row labelled "self re-prompt".

These tests pin the filename mapping and the two filters that must match the
object-space path (``min_confidence``, ``min_size``) so the arms stay
comparable.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from osgeo import ogr  # noqa: E402

ogr.DontUseExceptions()

from postprocess.prompt_export import (  # noqa: E402
    _export_self_image_prompts,
    maybe_export_bbox_prompts,
)

IMAGE = "DJI_20251013154527_0001_V.JPG"
STEM = "DJI_20251013154527_0001_V"
SHP = "images_DJI_20251013154527_0001_V__r0.shp"


def _write_pixel_shp(path, polys):
    """``polys``: list of ``(x0, y0, x1, y1, con)`` in image pixel coordinates."""
    driver = ogr.GetDriverByName("ESRI Shapefile")
    ds = driver.CreateDataSource(path)
    layer = ds.CreateLayer("pv", srs=None, geom_type=ogr.wkbPolygon)
    layer.CreateField(ogr.FieldDefn("con", ogr.OFTReal))
    for x0, y0, x1, y1, con in polys:
        feat = ogr.Feature(layer.GetLayerDefn())
        feat.SetField("con", con)
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)):
            ring.AddPoint(float(x), float(y))
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        feat.SetGeometry(poly)
        layer.CreateFeature(feat)
    ds = None


def _build(tmp_path, shp_specs, image_names=(IMAGE,)):
    """Lay out a station: ``images/*.JPG`` plus an ``infer/`` shapefile dir."""
    images_dir = tmp_path / "station" / "images"
    images_dir.mkdir(parents=True)
    for name in image_names:
        (images_dir / name).write_bytes(b"")

    infer_dir = tmp_path / "infer"
    infer_dir.mkdir()
    for shp_name, polys in shp_specs.items():
        _write_pixel_shp(str(infer_dir / shp_name), polys)

    out_dir = tmp_path / "prompts"
    cfg = {"data": {"data_root": str(tmp_path / "station"),
                    "image_glob": "images/*.JPG"}}
    prompt_cfg = {
        "enabled": True,
        "mode": "self_image",
        "per_image_raw_shp_dir": str(infer_dir),
        "output_dir": str(out_dir),
        "min_size": 50.0,
        "min_confidence": 0.5,
    }
    return cfg, prompt_cfg, out_dir


def _boxes(out_dir, stem=STEM):
    path = out_dir / f"{stem}.txt"
    if not path.exists():
        return None
    return [[float(v) for v in line.split(",")]
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_prompt_file_is_named_after_the_image_not_the_shapefile(tmp_path):
    cfg, prompt_cfg, out = _build(tmp_path, {SHP: [(100, 200, 400, 500, 0.9)]})

    _export_self_image_prompts(cfg=cfg, prompt_cfg=prompt_cfg)

    assert (out / f"{STEM}.txt").exists()
    assert not (out / "images_DJI_20251013154527_0001_V__r0.txt").exists()


def test_exported_box_is_the_polygon_pixel_envelope(tmp_path):
    cfg, prompt_cfg, out = _build(tmp_path, {SHP: [(100, 200, 400, 500, 0.9)]})

    _export_self_image_prompts(cfg=cfg, prompt_cfg=prompt_cfg)

    assert _boxes(out) == [[100.0, 200.0, 400.0, 500.0]]


def test_features_below_min_confidence_are_not_exported(tmp_path):
    cfg, prompt_cfg, out = _build(tmp_path, {SHP: [
        (100, 200, 400, 500, 0.9),
        (600, 700, 900, 1000, 0.4),
    ]})

    _export_self_image_prompts(cfg=cfg, prompt_cfg=prompt_cfg)

    assert _boxes(out) == [[100.0, 200.0, 400.0, 500.0]]


def test_boxes_thinner_than_min_size_are_not_exported(tmp_path):
    cfg, prompt_cfg, out = _build(tmp_path, {SHP: [
        (100, 200, 400, 500, 0.9),
        (600, 700, 900, 740, 0.9),   # 300 x 40, below min_size=50
    ]})

    _export_self_image_prompts(cfg=cfg, prompt_cfg=prompt_cfg)

    assert _boxes(out) == [[100.0, 200.0, 400.0, 500.0]]


def test_image_without_a_shapefile_gets_no_prompt_file(tmp_path):
    other = "DJI_20251013154527_0002_V.JPG"
    cfg, prompt_cfg, out = _build(
        tmp_path,
        {SHP: [(100, 200, 400, 500, 0.9)]},
        image_names=(IMAGE, other),
    )

    _export_self_image_prompts(cfg=cfg, prompt_cfg=prompt_cfg)

    assert _boxes(out) == [[100.0, 200.0, 400.0, 500.0]]
    assert not (out / "DJI_20251013154527_0002_V.txt").exists()


def test_reports_how_many_files_and_boxes_it_wrote(tmp_path):
    """The count is the only guard against a silently empty export."""
    cfg, prompt_cfg, out = _build(tmp_path, {SHP: [
        (100, 200, 400, 500, 0.9),
        (600, 700, 900, 1000, 0.9),
    ]})

    info = _export_self_image_prompts(cfg=cfg, prompt_cfg=prompt_cfg)

    assert info["mode"] == "self_image"
    assert info["files"] == 1
    assert info["boxes"] == 2


def test_reports_the_image_count_so_a_short_export_is_detectable(tmp_path):
    """Fewer prompt files than images is the silent-failure signature.

    The count is only checkable if the exporter says how many images it looked
    at, so it has to come back alongside the file count.
    """
    cfg, prompt_cfg, _ = _build(
        tmp_path,
        {SHP: [(100, 200, 400, 500, 0.9)]},
        image_names=(IMAGE, "DJI_20251013154527_0002_V.JPG"),
    )

    info = _export_self_image_prompts(cfg=cfg, prompt_cfg=prompt_cfg)

    assert info["images"] == 2
    assert info["files"] == 1


def test_mode_self_image_is_routed_by_maybe_export_bbox_prompts(tmp_path):
    """The pipeline only ever calls maybe_export_bbox_prompts.

    ``shp_path`` is deliberately bogus: this mode reads the per-image results,
    never the fused shapefile, so routing must not touch it.
    """
    cfg, prompt_cfg, out = _build(tmp_path, {SHP: [(100, 200, 400, 500, 0.9)]})
    cfg["postprocess"] = {"prompt_export": prompt_cfg}

    info = maybe_export_bbox_prompts(cfg=cfg, shp_path="/nonexistent/fused.shp")

    assert info["mode"] == "self_image"
    assert (out / f"{STEM}.txt").exists()
