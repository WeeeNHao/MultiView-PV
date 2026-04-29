#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/XinXie/oblique_views.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/dataset/PV/ZS_PV/003-XinXie}"
DATA_ROOT="${DATA_ROOT:-/data/dataset/PV/003-XinXie/}"
IMAGE_GLOB="${IMAGE_GLOB:-images/*.JPG}"
ROUNDS="${ROUNDS:-3}"
VIEW_NUM="${VIEW_NUM:-5}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-iter}"
DISTRIBUTED="${DISTRIBUTED:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"

if [[ "$ROUNDS" -lt 1 ]]; then
  echo "ROUNDS must be >= 1" >&2
  exit 1
fi

run_round() {
  local iter_idx="$1"
  local prompt_source="$2"
  local enable_prompt="$3"
  local iter_dir="${OUTPUT_ROOT}/${RUN_NAME_PREFIX}_${iter_idx}"
  local view_dir="${iter_dir}/views"
  local infer_dir="${view_dir}/infer"
  local prompt_dir="${iter_dir}/prompts"
  local final_shp="${view_dir}/view_${VIEW_NUM}.shp"
  local collected_shp="${view_dir}/view_${VIEW_NUM}_collected.shp"
  local selected_shp="${view_dir}/view_${VIEW_NUM}_selected.shp"
  local multiview_shp="${view_dir}/view_${VIEW_NUM}_multiview.shp"

  mkdir -p "$infer_dir" "$prompt_dir" "$view_dir"

  local -a opts=(
    "pipeline.run_inference=true"
    "data.data_root=${DATA_ROOT}"
    "data.image_glob=${IMAGE_GLOB}"
    "postprocess.per_image_nms.enabled=true"
    "postprocess.per_image_nms.backend=auto"
    "postprocess.per_image_nms.score_field=con_weight"
    "postprocess.per_image_nms.iou_threshold=0.2"
    "postprocess.per_image_nms.use_geometry_iou=false"
    "postprocess.view_selection.enabled=true"
    "postprocess.view_selection.view_num=${VIEW_NUM}"
    "postprocess.view_selection.iou_threshold=0.2"
    "postprocess.view_selection.use_geometry_iou=false"
    "postprocess.view_selection.random_seed=42"
    "postprocess.multiview.enabled=true"
    "postprocess.multiview.strategy=nms_keep_max"
    "postprocess.multiview.score_field=con_weight"
    "postprocess.multiview.iou_threshold=0.2"
    "postprocess.prompt_export.enabled=true"
    "postprocess.prompt_export.mode=oblique"
    "postprocess.prompt_export.output_dir=${prompt_dir}"
    "postprocess.prompt_export.min_size=50"
    "postprocess.prompt_export.include_intersections=true"
    "postprocess.prompt_export.use_spatial_index=true"
    "output.per_image_shp_dir=${infer_dir}"
    "output.final_merged_shp=${final_shp}"
    "output.trace_exports.enabled=true"
    "output.trace_exports.collected_shp=${collected_shp}"
    "output.trace_exports.selected_shp=${selected_shp}"
    "output.trace_exports.multiview_shp=${multiview_shp}"
  )

  if [[ "$enable_prompt" == "1" ]]; then
    opts+=(
      "inference.prompt.enabled=true"
      "inference.prompt.source=${prompt_source}"
      "inference.prompt.strict_window_prompt=true"
      "inference.prompt.max_prompt_per_window=5"
    )
  else
    opts+=(
      "inference.prompt.enabled=false"
      "inference.prompt.source="
      "inference.prompt.strict_window_prompt=false"
      "inference.prompt.max_prompt_per_window=5"
    )
  fi

  if [[ "$DISTRIBUTED" == "0" ]]; then
    opts+=("distributed.enabled=false")
  fi

  echo "============================================================"
  echo "[xinxie] iter=${iter_idx}"
  echo "[xinxie] prompt_source=${prompt_source:-<none>}"
  echo "[xinxie] prompt_enabled=${enable_prompt}"
  echo "[xinxie] final_shp=${final_shp}"
  echo "[xinxie] prompt_dir=${prompt_dir}"
  echo "============================================================"

  if [[ "$DISTRIBUTED" == "1" && "$NPROC_PER_NODE" -gt 1 ]]; then
    "$TORCHRUN_BIN" --nproc_per_node="$NPROC_PER_NODE" \
      "$ROOT_DIR/cli/run_pipeline.py" --config "$CONFIG_PATH" "${opts[@]}"
  else
    "$PYTHON_BIN" "$ROOT_DIR/cli/run_pipeline.py" --config "$CONFIG_PATH" "${opts[@]}"
  fi
}

for ((i=0; i<ROUNDS; i++)); do
  if [[ "$i" -eq 0 ]]; then
    run_round "$i" "" "0"
  else
    prev_idx=$((i - 1))
    prev_prompt_dir="${OUTPUT_ROOT}/${RUN_NAME_PREFIX}_${prev_idx}/prompts"
    run_round "$i" "$prev_prompt_dir" "1"
  fi
done

echo "[xinxie] all rounds finished"
