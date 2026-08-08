#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/CangFang/oblique_views.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/dataset/PV/ZS_PV/004-CangFang-0518}"
DATA_ROOT="${DATA_ROOT:-/data/dataset/PV/004-CangFang/}"
IMAGE_GLOB="${IMAGE_GLOB:-images/*.JPG}"
ROUNDS="${ROUNDS:-4}"
VIEW_MIN="${VIEW_MIN:-1}"
VIEW_MAX="${VIEW_MAX:-15}"
DISTRIBUTED="${DISTRIBUTED:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"
REUSE_PER_IMAGE="${REUSE_PER_IMAGE:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
POSTPROCESS_ONLY="${POSTPROCESS_ONLY:-0}"
TRACE_EXPORTS_ENABLED="${TRACE_EXPORTS_ENABLED:-0}"
AUTO_POSTPROCESS_ON_CACHE="${AUTO_POSTPROCESS_ON_CACHE:-1}"
ITER_MIN="${ITER_MIN:-1}"
ITER_MAX="${ITER_MAX:-$((ROUNDS - 1))}"

if [[ "$ROUNDS" -lt 1 ]]; then
    echo "ROUNDS must be >= 1" >&2
    exit 1
fi

if [[ "$VIEW_MIN" -gt "$VIEW_MAX" ]]; then
    echo "VIEW_MIN must be <= VIEW_MAX" >&2
    exit 1
fi

if [[ "$ITER_MIN" -lt 1 ]]; then
    echo "ITER_MIN must be >= 1" >&2
    exit 1
fi

if [[ "$ITER_MAX" -lt "$ITER_MIN" ]]; then
    echo "ITER_MAX must be >= ITER_MIN" >&2
    exit 1
fi

run_pipeline() {
    local -a opts=("$@")
    local force_single=0

    for opt in "${opts[@]}"; do
        if [[ "$opt" == "distributed.enabled=false" ]]; then
            force_single=1
            break
        fi
    done

    if [[ "$DISTRIBUTED" == "0" ]]; then
        opts+=("distributed.enabled=false")
        force_single=1
    fi

    if [[ "$force_single" == "0" && "$DISTRIBUTED" == "1" && "$NPROC_PER_NODE" -gt 1 ]]; then
        "$TORCHRUN_BIN" --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" \
            "$ROOT_DIR/cli/run_pipeline.py" --config "$CONFIG_PATH" "${opts[@]}"
    else
        "$PYTHON_BIN" "$ROOT_DIR/cli/run_pipeline.py" --config "$CONFIG_PATH" "${opts[@]}"
    fi
}

has_shp_files() {
    local dir="$1"
    compgen -G "${dir}/*.shp" > /dev/null
}

ITER0_CACHE_DIR="${OUTPUT_ROOT}/iter_0/shared"
ITER0_INFER_DIR="${ITER0_CACHE_DIR}/infer"
ITER0_PROJ_DIR="${ITER0_CACHE_DIR}/proj"

prepare_iter0_cache() {
    if [[ "$SKIP_EXISTING" == "1" ]] && compgen -G "${ITER0_PROJ_DIR}/*.shp" > /dev/null; then
        echo "[cangfang] reuse iter0 cache: ${ITER0_PROJ_DIR}"
        return
    fi

    mkdir -p "$ITER0_INFER_DIR" "$ITER0_PROJ_DIR" "$ITER0_CACHE_DIR"

    local -a opts=(
        "pipeline.run_inference=true"
        "pipeline.run_projection=true"
        "pipeline.run_postprocess=false"
        "data.data_root=${DATA_ROOT}"
        "data.image_glob=${IMAGE_GLOB}"
        "output.per_image_raw_shp_dir=${ITER0_INFER_DIR}"
        "output.per_image_shp_dir=${ITER0_PROJ_DIR}"
        "output.final_merged_shp=${ITER0_CACHE_DIR}/iter0_placeholder.shp"
    )

    echo "[cangfang] build iter0 cache"
    run_pipeline "${opts[@]}"
}

run_iter0_postprocess() {
    local view_num="$1"
    local view_dir="${OUTPUT_ROOT}/iter_0/views/view_${view_num}"
    local prompt_dir="${view_dir}/prompts"
    local final_shp="${view_dir}/view_${view_num}.shp"

    if [[ "$SKIP_EXISTING" == "1" && -f "$final_shp" ]]; then
        echo "[cangfang] iter=0 view=${view_num} exists, skip"
        return
    fi

    mkdir -p "$view_dir" "$prompt_dir"

    local -a opts=(
        "pipeline.run_inference=false"
        "pipeline.run_projection=false"
        "pipeline.run_postprocess=true"
        "distributed.enabled=false"
        "output.per_image_shp_dir=${ITER0_PROJ_DIR}"
        "output.per_image_raw_shp_dir=${ITER0_INFER_DIR}"
        "postprocess.view_selection.view_num=${view_num}"
        "postprocess.prompt_export.enabled=true"
        "postprocess.prompt_export.output_dir=${prompt_dir}"
        "inference.prompt.enabled=false"
        "inference.prompt.source="
        "output.final_merged_shp=${final_shp}"
    )

    echo "[cangfang] iter=0 view=${view_num} postprocess"
    run_pipeline "${opts[@]}"
}

run_iter_with_prompt() {
    local iter_idx="$1"
    local view_num="$2"
    local prev_idx=$((iter_idx - 1))
    local prev_prompt_dir="${OUTPUT_ROOT}/iter_${prev_idx}/views/view_${view_num}/prompts"
    local view_dir="${OUTPUT_ROOT}/iter_${iter_idx}/views/view_${view_num}"
    local prompt_dir="${view_dir}/prompts"
    local final_shp="${view_dir}/view_${view_num}.shp"

    if [[ "$SKIP_EXISTING" == "1" && -f "$final_shp" ]]; then
        echo "[cangfang] iter=${iter_idx} view=${view_num} exists, skip"
        return
    fi

    local cache_root
    if [[ "$REUSE_PER_IMAGE" == "1" ]]; then
        cache_root="${OUTPUT_ROOT}/shared/view_${view_num}"
    else
        cache_root="$view_dir"
    fi

    local infer_dir="${cache_root}/infer"
    local proj_dir="${cache_root}/proj"

    mkdir -p "$infer_dir" "$proj_dir" "$prompt_dir" "$view_dir"

    local -a opts=(
        "pipeline.run_inference=true"
        "pipeline.run_projection=true"
        "pipeline.run_postprocess=true"
        "data.data_root=${DATA_ROOT}"
        "data.image_glob=${IMAGE_GLOB}"
        "postprocess.view_selection.view_num=${view_num}"
        "postprocess.prompt_export.enabled=true"
        "postprocess.prompt_export.output_dir=${prompt_dir}"
        "inference.prompt.enabled=true"
        "inference.prompt.source=${prev_prompt_dir}"
        "inference.prompt.strict_window_prompt=false"
        "inference.prompt.max_prompt_per_window=5"
        "output.trace_exports.enabled=${TRACE_EXPORTS_ENABLED}"
        "output.per_image_raw_shp_dir=${infer_dir}"
        "output.per_image_shp_dir=${proj_dir}"
        "output.final_merged_shp=${final_shp}"
    )

    echo "[cangfang] iter=${iter_idx} view=${view_num}"
    run_pipeline "${opts[@]}"
}

# prepare_iter0_cache

# for ((view=VIEW_MIN; view<=VIEW_MAX; view++)); do
# 	run_iter0_postprocess "$view"
# done

for ((iter=ITER_MIN; iter<=ITER_MAX; iter++)); do
    for ((view=VIEW_MIN; view<=VIEW_MAX; view++)); do
        run_iter_with_prompt "$iter" "$view"
    done
done

echo "[cangfang] all rounds finished"
