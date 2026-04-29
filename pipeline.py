from __future__ import annotations

import glob
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional
from tqdm import tqdm

from utils.common import DistInfo, FeatureList
from utils.config import RuntimeConfig
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
from io_flow.input_resolver import resolve_image_paths
from utils.log_utils import PipelineRunLogger
from postprocess import (
    fuse_multiview_features,
    maybe_export_bbox_prompts,
    merge_image_with_dom_features,
    select_views_from_features,
)
from postprocess.nms import nms_features
from io_flow.shp_io import export_features_to_shapefile, read_features_from_shapefile
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


def _project_and_export_image(
    features_px: FeatureList,
    projection_cfg: Dict[str, Any],
    geo_meta: Any,
    image_path: str,
    per_image_shp: str,
) -> int:
    features_map = project_and_score_features(
        features=features_px,
        geo_meta=geo_meta,
        projection_cfg=projection_cfg,
        image_path=image_path,
        progress_position=2,
    )
    export_features_to_shapefile(
        features=features_map,
        out_shp=per_image_shp,
        projection_wkt=geo_meta.projection_wkt,
    )
    return len(features_map)


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
    pipeline_cfg = _as_dict(cfg, "pipeline")
    run_inference = bool(pipeline_cfg.get("run_inference", True))

    per_image_dir = str(output_cfg.get("per_image_shp_dir", "outputs/per_image"))
    final_merged_shp = str(output_cfg.get("final_merged_shp", "outputs/final_merged.shp"))
    trace_cfg = _as_dict(output_cfg, "trace_exports")
    trace_enabled = bool(trace_cfg.get("enabled", True))
    trace_collected_shp = str(trace_cfg.get("collected_shp", "")).strip()
    trace_selected_shp = str(trace_cfg.get("selected_shp", "")).strip()

    final_base, final_ext = os.path.splitext(final_merged_shp)
    if not final_ext:
        final_ext = ".shp"
    if not trace_collected_shp:
        trace_collected_shp = f"{final_base}_collected{final_ext}"
    if not trace_selected_shp:
        trace_selected_shp = f"{final_base}_selected{final_ext}"

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
    if trace_enabled:
        _ensure_dir(os.path.dirname(trace_collected_shp) or ".")
        _ensure_dir(os.path.dirname(trace_selected_shp) or ".")

    if run_inference:
        image_paths = resolve_image_paths(data_cfg)
        rank_images = split_items_for_rank(image_paths, info)
    else:
        image_paths = []
        rank_images = []

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
        run_inference=run_inference,
    )
    run_logger.event(
        "pipeline_start",
        total_images=len(image_paths),
        assigned_images=len(rank_images),
        run_inference=run_inference,
    )

    model = None
    runner = None
    run_status = "success"
    run_error = ""

    try:
        if run_inference:
            with run_logger.stage("build_model"):
                model = build_model(model_cfg)
                runner = InferenceRunner(model=model, inference_cfg=inference_cfg)
    
            projection_workers = int(pipeline_cfg.get("projection_workers", 2))
            projection_workers = max(1, projection_workers)

            pbar = tqdm(
                rank_images,
                desc=f"Dataloader [rank={info.rank}]",
                position=0,
            )
            run_logger.set_tqdm(pbar)

            pending: Optional[tuple[str, int, Future[int]]] = None
            with ThreadPoolExecutor(max_workers=projection_workers, thread_name_prefix="proj") as pool:
                for image_path in pbar:
                    image_name = _safe_image_name(image_path)

                    with run_logger.stage("infer_image", image=image_name):
                        features_px, geo_meta = runner.run_image(
                            image_path=image_path,
                            progress_position=1,
                        )
                    num_features_before = len(features_px)

                    if pending is not None:
                        prev_image_name, prev_before, prev_future = pending
                        with run_logger.stage("wait_project_and_write", image=prev_image_name):
                            prev_after = prev_future.result()
                        run_logger.update_postfix(
                            img=prev_image_name.split("_", 1)[-1],
                            n=prev_after,
                            d=prev_after - prev_before,
                        )

                    rank_suffix = f"__r{info.rank}" if info.world_size > 1 else ""
                    per_image_shp = os.path.join(per_image_dir, f"{image_name}{rank_suffix}.shp")
                    future = pool.submit(
                        _project_and_export_image,
                        features_px,
                        projection_cfg,
                        geo_meta,
                        image_path,
                        per_image_shp,
                    )
                    pending = (image_name, num_features_before, future)

                if pending is not None:
                    prev_image_name, prev_before, prev_future = pending
                    with run_logger.stage("wait_project_and_write", image=prev_image_name):
                        prev_after = prev_future.result()
                    run_logger.update_postfix(
                        img=prev_image_name.split("_", 1)[-1],
                        n=prev_after,
                        d=prev_after - prev_before,
                    )

            run_logger.clear_tqdm()
            run_logger.info(
                "inference complete",
                images=len(rank_images),
                stage_stats={
                    "infer_image": run_logger.stage_stats.get("infer_image", {}),
                    "wait_project_and_write": run_logger.stage_stats.get("wait_project_and_write", {}),
                },
            )

            # Free GPU memory immediately after inference
            with run_logger.stage("model_close"):
                if model is not None:
                    model.close()
                    model = None
                runner = None
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    

        else:
            run_logger.info("skip inference and projection", per_image_dir=per_image_dir)
            if not _collect_rank_outputs(per_image_dir):
                raise FileNotFoundError(
                    f"No per-image shapefiles found under: {per_image_dir}. "
                    "Set pipeline.run_inference=true or prepare per-image shp files first."
                )


        with run_logger.stage("barrier"):
            barrier_if_needed()
        if not is_main_process(info):
            return

        projection_wkt = _read_first_projection_wkt(per_image_dir)

        per_image_nms_cfg = _as_dict(post_cfg, "per_image_nms")
        per_image_nms_enabled = bool(per_image_nms_cfg.get("enabled", False))
        score_field = str(per_image_nms_cfg.get("score_field", "con_weight"))
        iou_threshold = float(per_image_nms_cfg.get("iou_threshold", 0.5))
        use_geometry_iou = bool(per_image_nms_cfg.get("use_geometry_iou", False))
        nms_backend = str(per_image_nms_cfg.get("backend", "auto"))

        run_logger.info("collecting per-image outputs")
        nms_before_total = 0
        nms_after_total = 0
        nms_elapsed = 0.0
        if per_image_nms_enabled and not os.path.exists(trace_collected_shp):
            with run_logger.stage("collect_per_image_outputs"):
                shp_files = _collect_rank_outputs(per_image_dir)
                all_features: FeatureList = []
                pbar_shp = tqdm(shp_files, desc="Reading shapefiles")
                for shp in pbar_shp:
                    per_image_features = read_features_from_shapefile(shp_path=shp)
                    if per_image_nms_enabled:
                        nms_before_total += len(per_image_features)
                        nms_start = time.perf_counter()
                        per_image_features = nms_features(
                            features=per_image_features,
                            score_field=score_field,
                            iou_threshold=iou_threshold,
                            use_geometry_iou=use_geometry_iou,
                            backend=nms_backend,
                        )
                        nms_elapsed += time.perf_counter() - nms_start
                        nms_after_total += len(per_image_features)
                    all_features.extend(per_image_features)
                    pbar_shp.set_postfix(total=len(all_features), file=Path(shp).stem)
            run_logger.info("collected features", count=len(all_features))
            run_logger.event("feature_count", stage="collect", count=len(all_features))
            run_logger.log_count_change("per_image_nms", nms_before_total, nms_after_total)
            run_logger.info("after per-image nms", count=nms_after_total, delta=nms_after_total - nms_before_total)
            run_logger.info("per-image nms time", time=nms_elapsed)

        if trace_enabled and not os.path.exists(trace_collected_shp):
            with run_logger.stage(
                "write_collected_trace",
                out_shp=trace_collected_shp,
                feature_count=len(all_features),
            ):
                export_features_to_shapefile(
                    features=all_features,
                    out_shp=trace_collected_shp,
                    projection_wkt=projection_wkt,
                )
            run_logger.info("collected trace output", out_shp=trace_collected_shp, count=len(all_features))

        if os.path.exists(trace_collected_shp):
            all_features = read_features_from_shapefile(shp_path=trace_collected_shp)
            run_logger.info("collected trace exists, loaded features", count=len(all_features))

        
        view_selection_cfg = _as_dict(post_cfg, "view_selection")
        if bool(view_selection_cfg.get("enabled", False)):
            start_time = time.time()
            before = len(all_features)
            with run_logger.stage("view_selection"):
                all_features = select_views_from_features(all_features, view_selection_cfg)
            after = len(all_features)
            run_logger.log_count_change("view_selection", before, after)
            run_logger.info("after view selection", count=after, delta=after - before)
            run_logger.info("view selection time", time=time.time() - start_time)
        if trace_enabled:
            with run_logger.stage(
                "write_view_selection_trace",
                out_shp=trace_selected_shp,
                feature_count=len(all_features),
            ):
                export_features_to_shapefile(
                    features=all_features,
                    out_shp=trace_selected_shp,
                    projection_wkt=projection_wkt,
                )
            run_logger.info("view selection trace output", out_shp=trace_selected_shp, count=len(all_features))
        

        multiview_cfg = _as_dict(post_cfg, "multiview")
        if bool(multiview_cfg.get("enabled", True)):
            start_time = time.time()
            before = len(all_features)
            with run_logger.stage("multiview_fusion"):
                all_features = fuse_multiview_features(all_features, multiview_cfg)
            after = len(all_features)
            run_logger.log_count_change("multiview_fusion", before, after)
            run_logger.info("after multiview fusion", count=after, delta=after - before)
            run_logger.info("multiview fusion time", time=time.time() - start_time)

        dom_merge_cfg = _as_dict(post_cfg, "dom_merge")
        dom_shp = str(data_cfg.get("dom_shp", "")).strip()
        if bool(dom_merge_cfg.get("enabled", False)) and dom_shp and os.path.exists(dom_shp):
            dom_score_field = str(dom_merge_cfg.get("dom_score_field", "con"))

            with run_logger.stage("dom_read"):
                dom_features = read_features_from_shapefile(
                    shp_path=dom_shp,
                    score_field=dom_score_field,
                    sem_field=dom_score_field,
                    pv_field="con_pv",
                    label_field=str(dom_merge_cfg.get("dom_label_field", "label")),
                    src_field=str(dom_merge_cfg.get("dom_src_field", "src")),
                )
            run_logger.info("dom features loaded", count=len(dom_features))

            # Merge DOM features with image features
            before = len(all_features)
            with run_logger.stage("dom_merge"):
                all_features = merge_image_with_dom_features(
                    image_features=all_features,
                    dom_features=dom_features,
                    cfg=dom_merge_cfg,
                )
            after = len(all_features)
            run_logger.log_count_change("dom_merge", before, after, dom_shp=dom_shp)
            run_logger.info("after dom merge", count=after, delta=after - before)

 
        with run_logger.stage("write_final_output", feature_count=len(all_features)):
            export_features_to_shapefile(
                features=all_features,
                out_shp=final_merged_shp,
                projection_wkt=projection_wkt,
            )
        run_logger.info("final merged output", out_shp=final_merged_shp, count=len(all_features))

        with run_logger.stage("export_bbox_prompts", shp=final_merged_shp):
            prompt_info = maybe_export_bbox_prompts(cfg=cfg, shp_path=final_merged_shp)
        if prompt_info:
            run_logger.info("bbox prompts exported", **prompt_info)
            
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

        total_elapsed_mins = (time.perf_counter() - run_start) / 60.0
        summary_path = os.path.join(
            logs_dir,
            f"pipeline_summary_{run_logger.run_id}_rank{info.rank}.json",
        )
        run_logger.write_summary(
            summary_path=summary_path,
            status=run_status,
            error=run_error,
            total_elapsed_mins=total_elapsed_mins,
            assigned_images=len(rank_images),
            total_images=len(image_paths),
            final_merged_shp=final_merged_shp if is_main_process(info) else "",
        )
        run_logger.info("pipeline finished", status=run_status, total_elapsed_mins=round(total_elapsed_mins, 2))
