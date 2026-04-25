from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from refactor_v2.common import DistInfo, FeatureList
from refactor_v2.config import RuntimeConfig
from refactor_v2.inference.distributed import (
    barrier_if_needed,
    cleanup_distributed,
    get_dist_info,
    is_main_process,
    maybe_init_distributed,
    split_items_for_rank,
)
from refactor_v2.inference.models import build_model
from refactor_v2.inference.runner import InferenceRunner
from refactor_v2.io import (
    export_features_to_shapefile,
    read_features_from_shapefile,
    resolve_image_paths,
)
from refactor_v2.postprocess import fuse_multiview_features, merge_image_with_dom_features
from refactor_v2.projection import project_and_score_features


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

    per_image_dir = str(output_cfg.get("per_image_shp_dir", "outputs/refactor_v2/per_image"))
    final_merged_shp = str(output_cfg.get("final_merged_shp", "outputs/refactor_v2/final_merged.shp"))

    _ensure_dir(per_image_dir)
    _ensure_dir(os.path.dirname(final_merged_shp) or ".")

    image_paths = resolve_image_paths(data_cfg)
    rank_images = split_items_for_rank(image_paths, info)

    print(
        f"[Rank {info.rank}] total_images={len(image_paths)}, "
        f"assigned_images={len(rank_images)}, world_size={info.world_size}"
    )

    model = build_model(model_cfg)
    runner = InferenceRunner(model=model, inference_cfg=inference_cfg)

    try:
        for image_path in rank_images:
            image_name = _safe_image_name(image_path)
            print(f"[Rank {info.rank}] processing {image_name}")

            features_px, geo_meta = runner.run_image(image_path=image_path)
            features_map = project_and_score_features(
                features=features_px,
                geo_meta=geo_meta,
                projection_cfg=projection_cfg,
                image_path=image_path,
            )

            rank_suffix = f"__r{info.rank}" if info.world_size > 1 else ""
            per_image_shp = os.path.join(per_image_dir, f"{image_name}{rank_suffix}.shp")
            export_features_to_shapefile(
                features=features_map,
                out_shp=per_image_shp,
                projection_wkt=geo_meta.projection_wkt,
            )

        barrier_if_needed()

        if not is_main_process(info):
            return

        print("[Rank 0] collecting per-image outputs...")
        shp_files = _collect_rank_outputs(per_image_dir)
        all_features: FeatureList = []
        for shp in shp_files:
            all_features.extend(read_features_from_shapefile(shp_path=shp))

        print(f"[Rank 0] collected features: {len(all_features)}")

        multiview_cfg = _as_dict(post_cfg, "multiview")
        if bool(multiview_cfg.get("enabled", True)):
            all_features = fuse_multiview_features(all_features, multiview_cfg)
            print(f"[Rank 0] after multiview fusion: {len(all_features)}")

        from refactor_v2.postprocess.filter import filter_features
        filter_cfg = _as_dict(post_cfg, "filter")
        if bool(filter_cfg.get("enabled", False)):
            all_features = filter_features(all_features, filter_cfg)
            print(f"[Rank 0] after filtering: {len(all_features)}")

        dom_merge_cfg = _as_dict(post_cfg, "dom_merge")
        dom_shp = str(data_cfg.get("dom_shp", "")).strip()
        if bool(dom_merge_cfg.get("enabled", False)) and dom_shp and os.path.exists(dom_shp):
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
            print(f"[Rank 0] after DOM merge: {len(all_features)}")

        projection_wkt = _read_first_projection_wkt(per_image_dir)
        export_features_to_shapefile(
            features=all_features,
            out_shp=final_merged_shp,
            projection_wkt=projection_wkt,
        )
        print(f"[Rank 0] final merged output: {final_merged_shp}")
    finally:
        model.close()
        if distributed_enabled:
            cleanup_distributed()
