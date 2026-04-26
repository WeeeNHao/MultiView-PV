from __future__ import annotations

import glob
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import DistInfo, FeatureList
from config import RuntimeConfig
from inference.distributed import (
    barrier_if_needed,
    cleanup_distributed,
    get_dist_info,
    is_main_process,
    maybe_init_distributed,
    split_items_for_rank,
)
from inference.models import build_model
from inference.runner import InferenceRunner
from input_resolver import resolve_image_paths
from log_utils import PipelineRunLogger
from postprocess import fuse_multiview_features, merge_image_with_dom_features
from shp_io import export_features_to_shapefile, read_features_from_shapefile
from projection import project_and_score_features


def _as_dict(cfg: Any, key: str) -> Dict[str, Any]:
    obj = cfg.get(key, {})
    if isinstance(obj, dict):
        return obj
    return {}


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _collect_rank_outputs(per_image_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(per_image_dir, "*.shp")))


def _read_first_projection_wkt(per_image_dir: str) -> Optional[str]:
    prj_files = sorted(glob.glob(os.path.join(per_image_dir, "*.prj")))
    if not prj_files:
        return None
    with open(prj_files[0], "r", encoding="utf-8") as f:
        return f.read()


def _safe_image_name(image_path: str) -> str:
    path = Path(image_path)
    parent = path.parent.name or "root"
    return f"{parent}_{path.stem}"


def run_pipeline(runtime_cfg: RuntimeConfig) -> None:
    run_start = time.perf_counter()
    cfg = runtime_cfg.raw

    dist_cfg = _as_dict(cfg, "distributed")
    distributed_enabled = bool(dist_cfg.get("enabled", True))

    if distributed_enabled:
        info = maybe_init_distributed(backend=dist_cfg.get("backend"))
    else:
        info = get_dist_info()
        info = DistInfo(rank=0, world_size=1, local_rank=0)

    data_cfg = _as_dict(cfg, "data")
    output_cfg = _as_dict(cfg, "output")
    model_cfg = _as_dict(cfg, "model")
    inference_cfg = _as_dict(cfg, "inference")
    projection_cfg = _as_dict(cfg, "projection")
    post_cfg = _as_dict(cfg, "postprocess")

    per_image_dir = str(output_cfg.get("per_image_shp_dir", "outputs/per_image"))
    final_merged_shp = str(output_cfg.get("final_merged_shp", "outputs/final_merged.shp"))
    logging_cfg = _as_dict(output_cfg, "logging")
    logs_dir = str(
        logging_cfg.get(
            "log_dir",
            os.path.join(os.path.dirname(final_merged_shp) or ".", "logs"),
        )
    )
    logging_enabled = bool(logging_cfg.get("enabled", True))

    _ensure_dir(per_image_dir)
    _ensure_dir(os.path.dirname(final_merged_shp) or ".")
    _ensure_dir(logs_dir)

    image_paths = resolve_image_paths(data_cfg)
    rank_images = split_items_for_rank(image_paths, info)

    run_logger = PipelineRunLogger(
        log_dir=logs_dir,
        rank=info.rank,
        world_size=info.world_size,
        enabled=logging_enabled,
    )

    run_logger.info(
        "pipeline started",
        total_images=len(image_paths),
        assigned_images=len(rank_images),
        world_size=info.world_size,
    )
    run_logger.event(
        "pipeline_start",
        total_images=len(image_paths),
        assigned_images=len(rank_images),
    )

    model = None
    runner = None
    run_status = "success"
    run_error = ""

    try:
        with run_logger.stage("build_model"):
            model = build_model(model_cfg)
            runner = InferenceRunner(model=model, inference_cfg=inference_cfg)

        for image_path in rank_images:
            image_name = _safe_image_name(image_path)
            run_logger.info("processing image", image=image_name)

            with run_logger.stage("infer_and_project_image", image=image_name):
                features_px, geo_meta = runner.run_image(image_path=image_path)
                features_map = project_and_score_features(
                    features=features_px,
                    geo_meta=geo_meta,
                    projection_cfg=projection_cfg,
                    image_path=image_path,
                )

            rank_suffix = f"__r{info.rank}" if info.world_size > 1 else ""
            per_image_shp = os.path.join(per_image_dir, f"{image_name}{rank_suffix}.shp")
            with run_logger.stage(
                "write_per_image_shp",
                image=image_name,
                feature_count=len(features_map),
            ):
                export_features_to_shapefile(
                    features=features_map,
                    out_shp=per_image_shp,
                    projection_wkt=geo_meta.projection_wkt,
                )

        with run_logger.stage("barrier"):
            barrier_if_needed()

        if not is_main_process(info):
            return

        run_logger.info("collecting per-image outputs")
        with run_logger.stage("collect_per_image_outputs"):
            shp_files = _collect_rank_outputs(per_image_dir)
            all_features: FeatureList = []
            for shp in shp_files:
                all_features.extend(read_features_from_shapefile(shp_path=shp))
        run_logger.info("collected features", count=len(all_features))
        run_logger.event("feature_count", stage="collect", count=len(all_features))

        multiview_cfg = _as_dict(post_cfg, "multiview")
        if bool(multiview_cfg.get("enabled", True)):
            before = len(all_features)
            with run_logger.stage("multiview_fusion"):
                all_features = fuse_multiview_features(all_features, multiview_cfg)
            after = len(all_features)
            run_logger.log_count_change("multiview_fusion", before, after)
            run_logger.info("after multiview fusion", count=after, delta=after - before)

        from postprocess.filter import filter_features
        filter_cfg = _as_dict(post_cfg, "filter")
        if bool(filter_cfg.get("enabled", False)):
            before = len(all_features)
            with run_logger.stage("feature_filter"):
                all_features = filter_features(all_features, filter_cfg)
            after = len(all_features)
            run_logger.log_count_change("feature_filter", before, after)
            run_logger.info("after filtering", count=after, delta=after - before)

        dom_merge_cfg = _as_dict(post_cfg, "dom_merge")
        dom_shp = str(data_cfg.get("dom_shp", "")).strip()
        if bool(dom_merge_cfg.get("enabled", False)) and dom_shp and os.path.exists(dom_shp):
            before = len(all_features)
            with run_logger.stage("dom_merge"):
                dom_score_field = str(dom_merge_cfg.get("dom_score_field", "con"))
                dom_features = read_features_from_shapefile(
                    shp_path=dom_shp,
                    score_field=dom_score_field,
                    sem_field=dom_score_field,
                    pv_field="con_pv",
                    label_field=str(dom_merge_cfg.get("dom_label_field", "label")),
                    src_field=str(dom_merge_cfg.get("dom_src_field", "src")),
                )
                all_features = merge_image_with_dom_features(
                    image_features=all_features,
                    dom_features=dom_features,
                    cfg=dom_merge_cfg,
                )
            after = len(all_features)
            run_logger.log_count_change("dom_merge", before, after, dom_shp=dom_shp)
            run_logger.info("after dom merge", count=after, delta=after - before)

        projection_wkt = _read_first_projection_wkt(per_image_dir)
        with run_logger.stage("write_final_output", feature_count=len(all_features)):
            export_features_to_shapefile(
                features=all_features,
                out_shp=final_merged_shp,
                projection_wkt=projection_wkt,
            )
        run_logger.info("final merged output", out_shp=final_merged_shp, count=len(all_features))
    except Exception as exc:
        run_status = "failed"
        run_error = str(exc)
        run_logger.event("pipeline_error", error=run_error)
        raise
    finally:
        if model is not None:
            with run_logger.stage("model_close"):
                model.close()
        if distributed_enabled:
            with run_logger.stage("cleanup_distributed"):
                cleanup_distributed()

        total_elapsed_ms = (time.perf_counter() - run_start) * 1000.0
        summary_path = os.path.join(
            logs_dir,
            f"pipeline_summary_{run_logger.run_id}_rank{info.rank}.json",
        )
        run_logger.write_summary(
            summary_path=summary_path,
            status=run_status,
            error=run_error,
            total_elapsed_ms=total_elapsed_ms,
            assigned_images=len(rank_images),
            total_images=len(image_paths),
            final_merged_shp=final_merged_shp if is_main_process(info) else "",
        )
        run_logger.info("pipeline finished", status=run_status, total_elapsed_ms=round(total_elapsed_ms, 2))
