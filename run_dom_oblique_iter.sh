#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

DOM_CONFIG="${ROOT_DIR}/refactor_v2/configs/dom_mode_template.yaml"
OBLIQUE_CONFIG="${ROOT_DIR}/refactor_v2/configs/oblique_mode_template.yaml"
ROUNDS=2
EXP_DIR="${ROOT_DIR}/outputs/refactor_v2/iter_exp"
PYTHON_BIN="python"
ENABLE_LEGACY_VIEW_LOOP=0
GT_SHP=""
INDEX_NAME=""
VIEW_NUM=5
POSE_CSV=""
DSM_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dom-config)
      DOM_CONFIG="$2"
      shift 2
      ;;
    --oblique-config)
      OBLIQUE_CONFIG="$2"
      shift 2
      ;;
    --rounds)
      ROUNDS="$2"
      shift 2
      ;;
    --exp-dir)
      EXP_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --enable-legacy-view-loop)
      ENABLE_LEGACY_VIEW_LOOP="$2"
      shift 2
      ;;
    --gt-shp)
      GT_SHP="$2"
      shift 2
      ;;
    --index-name)
      INDEX_NAME="$2"
      shift 2
      ;;
    --view-num)
      VIEW_NUM="$2"
      shift 2
      ;;
    --pose-csv)
      POSE_CSV="$2"
      shift 2
      ;;
    --dsm-path)
      DSM_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1"
      echo "Usage: $0 [--dom-config path] [--oblique-config path] [--rounds N] [--exp-dir path] [--python bin] [--enable-legacy-view-loop 0|1] [--gt-shp path] [--index-name name] [--view-num N] [--pose-csv path] [--dsm-path path]"
      exit 2
      ;;
  esac
done

if [[ "$ENABLE_LEGACY_VIEW_LOOP" == "1" ]]; then
  if [[ -z "$GT_SHP" || -z "$POSE_CSV" || -z "$DSM_PATH" ]]; then
    echo "When --enable-legacy-view-loop=1, --gt-shp, --pose-csv, and --dsm-path are required."
    exit 2
  fi
fi

# We need an image_glob for select_views and bbox prompts. Read from OBLIQUE_CONFIG if possible, or pass it.
# We'll just assume a default or pass it as an argument, but since we can't easily parse YAML in bash, we'll extract it using python or grep.
IMAGE_GLOB=$(grep -E '^[[:space:]]*image_glob:' "$OBLIQUE_CONFIG" | awk '{print $2}' | tr -d '"'\')
DOM_PATH=$(grep -A 1 'image_path:' "$DOM_CONFIG" | tail -n 1 | awk '{print $2}' | tr -d '"'\')

mkdir -p "$EXP_DIR"

echo "[Step 1/3] DOM baseline"
DOM_PER_IMAGE="${EXP_DIR}/round0_dom/per_image"
DOM_FINAL="${EXP_DIR}/round0_dom/dom.shp"
mkdir -p "$(dirname "$DOM_FINAL")"
"$PYTHON_BIN" "${ROOT_DIR}/main_refactor.py" \
  --config "$DOM_CONFIG" \
  output.per_image_shp_dir="$DOM_PER_IMAGE" \
  output.final_merged_shp="$DOM_FINAL" \
  postprocess.filter.enabled=true

echo "[Step 2/3] Oblique baseline (no prior merge)"
OBL_PER_IMAGE="${EXP_DIR}/round0_oblique/per_image"
OBL_FINAL="${EXP_DIR}/round0_oblique/oblique.shp"
mkdir -p "$(dirname "$OBL_FINAL")"
"$PYTHON_BIN" "${ROOT_DIR}/main_refactor.py" \
  --config "$OBLIQUE_CONFIG" \
  postprocess.dom_merge.enabled=false \
  output.per_image_shp_dir="$OBL_PER_IMAGE" \
  output.final_merged_shp="$OBL_FINAL" \
  postprocess.filter.enabled=true

PRIOR_SHP="$DOM_FINAL"

echo "[Step 3/3] Iterative fusion rounds: $ROUNDS"
for ((r=1; r<=ROUNDS; r++)); do
  ITER_DIR="${EXP_DIR}/round${r}_fused"
  ITER_PER_IMAGE="${ITER_DIR}/per_image"
  ITER_FINAL="${ITER_DIR}/fused_oblique.shp"
  mkdir -p "$ITER_DIR"

  PROMPT_ENABLED=0
  PROMPT_DIR=""

  if [[ "$ENABLE_LEGACY_VIEW_LOOP" == "1" && -n "$IMAGE_GLOB" ]]; then
    SELECTED_SHP="${ITER_DIR}/selected.shp"
    PROMPT_DIR="${ITER_DIR}/prompts"
    mkdir -p "$PROMPT_DIR"

    echo "[Round ${r}] [prep 1/2] select views from prior result"
    "$PYTHON_BIN" "${ROOT_DIR}/refactor_v2/cli/select_views.py" \
      --config "$OBLIQUE_CONFIG" \
      --merged_shp "$PRIOR_SHP" \
      --gt_shp "$GT_SHP" \
      --output_shp "$SELECTED_SHP" \
      --view_num "$VIEW_NUM"

    echo "[Round ${r}] [prep 2/2] generate bbox prompts"
    "$PYTHON_BIN" "${ROOT_DIR}/refactor_v2/cli/generate_bbox_prompt.py" \
      --config "$OBLIQUE_CONFIG" \
      --shp_file "$SELECTED_SHP" \
      --output_folder "$PROMPT_DIR"

    PROMPT_ENABLED=1
  fi

  echo "[Round ${r}] Oblique inference. prior=${PRIOR_SHP}"
  cmd=(
    "$PYTHON_BIN" "${ROOT_DIR}/main_refactor.py"
    --config "$OBLIQUE_CONFIG"
    "data.dom_shp=$PRIOR_SHP"
    postprocess.dom_merge.enabled=true
    postprocess.filter.enabled=true
    "output.per_image_shp_dir=$ITER_PER_IMAGE"
    "output.final_merged_shp=$ITER_FINAL"
  )

  if [[ "$PROMPT_ENABLED" == "1" ]]; then
    cmd+=(
      inference.prompt.enabled=true
      "inference.prompt.source=$PROMPT_DIR"
      inference.prompt.strict_window_prompt=true
    )
  fi

  "${cmd[@]}"

  # Now do DOM iteration
  DOM_ITER_PER_IMAGE="${ITER_DIR}/dom_per_image"
  DOM_ITER_FINAL="${ITER_DIR}/fused_final.shp"
  DOM_PROMPT_TXT="${ITER_DIR}/dom_prompts/bboxes.txt"
  
  if [[ -n "$DOM_PATH" ]]; then
      echo "[Round ${r}] Generate DOM prompts from ${ITER_FINAL}"
      "$PYTHON_BIN" "${ROOT_DIR}/refactor_v2/cli/generate_bbox_prompt_with_dom.py" \
        --config "$DOM_CONFIG" \
        --shp "$ITER_FINAL" \
        --output "$DOM_PROMPT_TXT"
      
      echo "[Round ${r}] DOM inference and final fusion."
      "$PYTHON_BIN" "${ROOT_DIR}/main_refactor.py" \
        --config "$DOM_CONFIG" \
        "data.dom_shp=$ITER_FINAL" \
        postprocess.dom_merge.enabled=true \
        postprocess.filter.enabled=true \
        "output.per_image_shp_dir=$DOM_ITER_PER_IMAGE" \
        "output.final_merged_shp=$DOM_ITER_FINAL" \
        inference.prompt.enabled=true \
        "inference.prompt.source=$DOM_PROMPT_TXT"
        
      PRIOR_SHP="$DOM_ITER_FINAL"
  else
      PRIOR_SHP="$ITER_FINAL"
  fi
done

echo "Done. Latest fused result: ${PRIOR_SHP}"
