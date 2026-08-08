#!/usr/bin/env bash
# Unified per-station experiment driver for erperiments_draft.md (tables 1-8).
# See experiments_plan.md for the protocol this implements.
#
#   ./scripts/run_station_exp.sh <STATION> <STAGE> [STAGE ...]
#
#   STATION : 001-BeiOu | 003-XinXie | 004-CangFang
#   STAGE   : S0  preflight
#             S1  shared perspective cache   S1R  repair that cache
#             S1P reproject that cache in place (no inference) -- use after a
#                 projection-only code change invalidates proj/ but not infer/
#             SD  TDOM branch WITH prior -- P's TDOM input (not a table row)
#             SM1 M1  SAM3-TDOM-Iter                  (table 1/2)
#             SM2 M2  SAM3-MV-PixelVote-Iter          (table 1/2)
#             SM3 M3  SAM3-TDOM+MV-LateFusion-Iter    (table 1/2)
#             SP  P   Ours, dual-source feedback      (table 1/2, table 3 Full)
#             S2  iter_0 postprocess | S3 iterate t=1..N -- perspective-only,
#                 supplies table 3 "MV feedback only" and table 5 "w/o TDOM"
#             S4  camera-direction sets at t=0          (table 7)
#             S5  module-geometry prior ablation, t=1   (table 4)
#             S6  TDOM-only feedback path, t=1          (table 3 row 2)
#             S7  camera-direction sets at t=1          (table 7, feedback round)
#
#   Dependencies:
#     S1 (+S1R) -> S2 -> S3 ;  S1 -> S4 -> S7 ;  S1 -> SM2/SM3 ;
#     SD -> SP -> S6 ;  SM1 -> SM3 ;  S1 -> S5
#
#   Prior: only SD/SP/S2/S3/S4 carry the module-geometry prior. SM1/SM2/SM3 are
#   baselines and run with NO_PRIOR_OPTS + GLOBAL_PROJ_OPTS per draft table 0.
#
# Every stage is idempotent: with SKIP_EXISTING=1 (default) an existing final
# shapefile makes the stage a no-op, so Ctrl-C and re-run resumes in place.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

STATION="${1:-}"
if [[ -z "$STATION" ]]; then
    echo "usage: $0 <STATION> <STAGE> [STAGE ...]" >&2
    exit 1
fi
shift
STAGES=("$@")
if [[ ${#STAGES[@]} -eq 0 ]]; then
    echo "usage: $0 <STATION> <STAGE> [STAGE ...]" >&2
    exit 1
fi

# --- station presets: GPU assignment per experiments_plan.md section 1 --------
case "$STATION" in
    # Two ranks share one 48GB card on the single-GPU stations: inference needs
    # ~10GB but projection is CPU-bound at ~23s/image, so one rank per card
    # leaves most of the 48 cores idle. Each rank also holds its own preloaded
    # DSM, so raising this further is bounded by RAM, not VRAM.
    001-BeiOu)
        DEF_GPUS="0"; DEF_NPROC=2; DEF_PORT=29500
        DEF_CONFIG="$ROOT_DIR/configs/BeiOu/oblique_views.yaml"
        DEF_DOM_CONFIG="$ROOT_DIR/configs/BeiOu/dom_only.yaml"
        ;;
    003-XinXie)
        DEF_GPUS="1"; DEF_NPROC=2; DEF_PORT=29501
        DEF_CONFIG="$ROOT_DIR/configs/XinXie/oblique_views.yaml"
        DEF_DOM_CONFIG="$ROOT_DIR/configs/XinXie/dom_only.yaml"
        ;;
    004-CangFang)
        DEF_GPUS="2,3,4,5"; DEF_NPROC=4; DEF_PORT=29502
        DEF_CONFIG="$ROOT_DIR/configs/CangFang/oblique_views.yaml"
        DEF_DOM_CONFIG="$ROOT_DIR/configs/CangFang/dom_only.yaml"
        ;;
    *)
        echo "unknown station: $STATION (expected 001-BeiOu|003-XinXie|004-CangFang)" >&2
        exit 1
        ;;
esac

PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
GPUS="${GPUS:-$DEF_GPUS}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$DEF_NPROC}"
MASTER_PORT="${MASTER_PORT:-$DEF_PORT}"
CONFIG_PATH="${CONFIG_PATH:-$DEF_CONFIG}"
DOM_CONFIG="${DOM_CONFIG:-$DEF_DOM_CONFIG}"
DATA_ROOT="${DATA_ROOT:-/data/dataset/PV/${STATION}/}"
OUT_ROOT="${OUT_ROOT:-/data/dataset/PV/ZS_PV/${STATION}-exp}"
IMAGE_GLOB="${IMAGE_GLOB:-images/*.JPG}"

ITER_MAX="${ITER_MAX:-2}"            # main iterations t=1..ITER_MAX
SKIP_EXISTING="${SKIP_EXISTING:-1}"
BATCH_SIZE="${BATCH_SIZE:-}"         # override for the 11GB 2080Ti group
OVERLAP="${OVERLAP:-}"               # slicing overlap ratio override
PIXEL_CELL_SIZE="${PIXEL_CELL_SIZE:-0.2}"   # M2/M3 vote grid resolution (m)
PIXEL_MIN_VOTES="${PIXEL_MIN_VOTES:-2}"     # cells needing >= this support
PIXEL_MIN_AREA="${PIXEL_MIN_AREA:-0}"       # drop vote components below this (m^2)
# Suffix for the M2/M3 output directory, so a re-tuned baseline lands beside the
# default-parameter one instead of destroying it. The projection cache is NOT
# suffixed: it depends only on the projection method, not on the fusion knobs,
# so every variant reuses the same shared_proj.
#
# min_votes counts *polygons* covering a cell, not views. Measured coverage is
# 27.0 polygons per module at BeiOu, 26.1 at XinXie, 34.2 at CangFang, so the
# useful threshold is an order of magnitude above the default of 2 and differs
# per station -- see scripts/sweep_pixel_vote.py.
PIXEL_SUFFIX="${PIXEL_SUFFIX:-}"
# Which table-4 ablations stage S5 runs; narrow it to split S5 across cards.
S5_CONFIGS="${S5_CONFIGS:-noprior no_area no_ratio no_shape}"   # + only_shape

LOG_DIR="${OUT_ROOT}/_logs"
# Several streams of the same stage can run at once (disjoint S5_CONFIGS split
# across cards). They all `tee -a` the same file, so without a per-stream
# suffix the stage log is an unreadable interleave of every stream.
STAGE_LOG_SUFFIX="${STAGE_LOG_SUFFIX:-}"
SHARED_DIR="${OUT_ROOT}/iter_0/shared"
SHARED_INFER="${SHARED_DIR}/infer"
SHARED_PROJ="${SHARED_DIR}/proj"
MAIN_DIR="${OUT_ROOT}/iter_0/full"   # t=0 result over the full direction set

mkdir -p "$LOG_DIR"

log() { echo "[$STATION][$(date +%H:%M:%S)] $*"; }

# Common overrides applied to every invocation.
common_opts() {
    local -a o=()
    [[ -n "$BATCH_SIZE" ]] && o+=("inference.dataloader.batch_size=${BATCH_SIZE}")
    if [[ -n "$OVERLAP" ]]; then
        o+=("inference.slicing.overlap_height_ratio=${OVERLAP}")
        o+=("inference.slicing.overlap_width_ratio=${OVERLAP}")
    fi
    printf '%s\n' "${o[@]:-}"
}

# Dispatch one pipeline run on this station's GPUs. Falls back to a plain
# single-process run when the caller disabled distribution (postprocess-only
# stages must not open an NCCL group).
run_pipeline() {
    local -a opts=("$@")
    local force_single=0
    for opt in "${opts[@]}"; do
        [[ "$opt" == "distributed.enabled=false" ]] && force_single=1 && break
    done

    local -a extra=()
    mapfile -t extra < <(common_opts)
    for e in "${extra[@]}"; do [[ -n "$e" ]] && opts+=("$e"); done

    if [[ "$force_single" == "0" && "$NPROC_PER_NODE" -gt 1 ]]; then
        CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN_BIN" \
            --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" \
            "$ROOT_DIR/cli/run_pipeline.py" --config "$CONFIG_PATH" "${opts[@]}"
    else
        # Pin to the first listed device so postprocess-only runs stay off the
        # cards the other stations are using.
        CUDA_VISIBLE_DEVICES="${GPUS%%,*}" "$PYTHON_BIN" \
            "$ROOT_DIR/cli/run_pipeline.py" --config "$CONFIG_PATH" "${opts[@]}"
    fi
}

# ---------------------------------------------------------------- S0 preflight
stage_S0() {
    log "S0 preflight"
    "$PYTHON_BIN" "$ROOT_DIR/scripts/exp_preflight.py" \
        --station "$STATION" --config "$CONFIG_PATH" \
        --data-root "$DATA_ROOT" --image-glob "$IMAGE_GLOB"

    local smoke_dir="${OUT_ROOT}/_preflight"
    if [[ "$SKIP_EXISTING" == "1" && -f "${smoke_dir}/smoke.shp" ]]; then
        log "S0 smoke test already done, skip"
        return
    fi
    mkdir -p "$smoke_dir"

    # One image through the full inference+projection path on the target GPUs.
    local -a all_imgs=()
    mapfile -t all_imgs < <(compgen -G "${DATA_ROOT}${IMAGE_GLOB}" | sort)
    local first_img="${all_imgs[0]:-}"
    if [[ -z "$first_img" ]]; then
        log "S0 FAILED: no image matched ${DATA_ROOT}${IMAGE_GLOB}"
        return 1
    fi
    echo "$first_img" > "${smoke_dir}/one_image.txt"
    log "S0 smoke image: $(basename "$first_img")"

    run_pipeline \
        "distributed.enabled=false" \
        "pipeline.run_inference=true" \
        "pipeline.run_projection=true" \
        "pipeline.run_postprocess=true" \
        "data.data_root=${DATA_ROOT}" \
        "data.image_glob=" \
        "data.image_list_file=${smoke_dir}/one_image.txt" \
        "postprocess.view_selection.enabled=false" \
        "postprocess.prompt_export.enabled=false" \
        "output.trace_exports.enabled=false" \
        "output.per_image_raw_shp_dir=${smoke_dir}/infer" \
        "output.per_image_shp_dir=${smoke_dir}/proj" \
        "output.final_merged_shp=${smoke_dir}/smoke.shp"

    log "S0 OK -> ${smoke_dir}/smoke.shp"
}

# ------------------------------------------------- S1 shared inference cache
# The single full-inference pass for the station. Everything downstream reuses
# it, so this is the only stage whose cost scales with the image count.
stage_S1() {
    if [[ "$SKIP_EXISTING" == "1" ]] && compgen -G "${SHARED_PROJ}/*.shp" > /dev/null; then
        log "S1 cache exists (${SHARED_PROJ}), skip"
        return
    fi
    log "S1 building shared iter_0 cache"
    mkdir -p "$SHARED_INFER" "$SHARED_PROJ"

    run_pipeline \
        "pipeline.run_inference=true" \
        "pipeline.run_projection=true" \
        "pipeline.run_postprocess=false" \
        "data.data_root=${DATA_ROOT}" \
        "data.image_glob=${IMAGE_GLOB}" \
        "inference.prompt.enabled=false" \
        "inference.prompt.source=" \
        "output.per_image_raw_shp_dir=${SHARED_INFER}" \
        "output.per_image_shp_dir=${SHARED_PROJ}" \
        "output.final_merged_shp=${SHARED_DIR}/_placeholder.shp"

    log "S1 done -> ${SHARED_PROJ}"
}

# --------------------------------------------------- S1R repair the S1 cache
# Projects any raw shapefile that has no counterpart in proj/. Runs are complete
# by construction since the barrier fix in pipeline.py, but caches built before
# it lost whichever images the slowest rank was still writing. Cheap no-op when
# nothing is missing.
stage_S1R() {
    local tmp_raw="${SHARED_DIR}/_repair_raw"
    rm -rf "$tmp_raw"
    mkdir -p "$tmp_raw"

    local n=0
    local base
    for raw in "${SHARED_INFER}"/*.shp; do
        base="$(basename "$raw" .shp)"
        if [[ ! -f "${SHARED_PROJ}/${base}.shp" ]]; then
            # Shapefiles are multi-file; carry every sidecar across.
            cp "${SHARED_INFER}/${base}".* "$tmp_raw"/ 2>/dev/null
            n=$((n + 1))
        fi
    done

    if [[ "$n" -eq 0 ]]; then
        log "S1R cache complete ($(ls -1 "${SHARED_PROJ}"/*.shp | wc -l) projected), nothing to repair"
        rm -rf "$tmp_raw"
        return
    fi

    log "S1R projecting ${n} image(s) missing from proj/"
    run_pipeline \
        "distributed.enabled=false" \
        "pipeline.run_inference=false" \
        "pipeline.run_projection=true" \
        "pipeline.run_postprocess=false" \
        "data.data_root=${DATA_ROOT}" \
        "data.image_glob=${IMAGE_GLOB}" \
        "output.per_image_raw_shp_dir=${tmp_raw}" \
        "output.per_image_shp_dir=${SHARED_PROJ}" \
        "output.final_merged_shp=${SHARED_DIR}/_repair_placeholder.shp"

    rm -rf "$tmp_raw"
    log "S1R done -> $(ls -1 "${SHARED_PROJ}"/*.shp | wc -l) projected"
}

# -------------------------------------------- S1P reproject the shared cache
# Rebuild proj/ from the (valid) infer/ raw shapefiles, skipping SAM3 inference
# entirely. The projection code changed (deterministic nearest-hit ray/DSM
# intersection) but inference did not, so the pixel-space infer/ cache is still
# exact -- only object-space proj/ is stale. This reuses the same
# run_inference=false + run_projection=true path _m_baseline uses, so it
# distributes across all of the station's ranks. Delete proj/ first (see the
# cleanup preceding the run) so this rebuilds every image rather than no-op'ing.
stage_S1P() {
    if ! compgen -G "${SHARED_INFER}/*.shp" > /dev/null; then
        log "S1P FAILED: no raw cache in ${SHARED_INFER}; run S1 first"
        return 1
    fi
    if [[ "$SKIP_EXISTING" == "1" ]] && compgen -G "${SHARED_PROJ}/*.shp" > /dev/null; then
        log "S1P proj cache present (${SHARED_PROJ}); delete it to force reprojection, skip"
        return
    fi
    log "S1P reprojecting shared cache from infer/ ($(ls -1 "${SHARED_INFER}"/*.shp | wc -l) images, no inference)"
    mkdir -p "$SHARED_PROJ"
    run_pipeline \
        "pipeline.run_inference=false" \
        "pipeline.run_projection=true" \
        "pipeline.run_postprocess=false" \
        "data.data_root=${DATA_ROOT}" \
        "data.image_glob=${IMAGE_GLOB}" \
        "output.per_image_raw_shp_dir=${SHARED_INFER}" \
        "output.per_image_shp_dir=${SHARED_PROJ}" \
        "output.final_merged_shp=${SHARED_DIR}/_placeholder.shp"
    log "S1P done -> $(ls -1 "${SHARED_PROJ}"/*.shp | wc -l) projected"
}

# --- baseline knobs (draft table 0) ------------------------------------------
# M1/M2/M3 all leave "Physical prior" blank: rank and threshold on raw SAM3
# confidence, with no area / rectangularity / aspect-ratio term. Only P uses
# the prior, which is why SD (P's TDOM branch) does NOT take these.
NO_PRIOR_OPTS=(
    "projection.score.w_sem=1.0"
    "projection.score.w_pv=0.0"
)
# "Standard / global" DSM projection, as opposed to P's per-module local fit.
GLOBAL_PROJ_OPTS=("projection.oblique.method=affine")

# --------------------------------------------------------- SM1 baseline M1
# SAM3-TDOM-Iter: DOM only, no prior, iterative bbox prompting.
stage_SM1() {
    local root="${OUT_ROOT}/m1"
    for ((t = 0; t <= ITER_MAX; t++)); do
        local tdir="${root}/iter_${t}"
        local final="${tdir}/final.shp"
        if [[ "$SKIP_EXISTING" == "1" && -f "$final" ]]; then
            log "SM1 t=${t} exists, skip"; continue
        fi
        mkdir -p "$tdir"

        local -a opts=(
            "distributed.enabled=false"
            "pipeline.run_inference=true"
            "pipeline.run_projection=true"
            "pipeline.run_postprocess=true"
            "postprocess.view_selection.enabled=false"
            "postprocess.dom_merge.enabled=false"
            "postprocess.prompt_export.enabled=true"
            "postprocess.prompt_export.mode=dom"
            "postprocess.prompt_export.output_txt=${tdir}/prompts.txt"
            "output.trace_exports.enabled=false"
            "output.per_image_raw_shp_dir=${tdir}/infer"
            "output.per_image_shp_dir=${tdir}/proj"
            "output.final_merged_shp=${final}"
            "${NO_PRIOR_OPTS[@]}"
        )
        if [[ "$t" -eq 0 ]]; then
            opts+=("inference.prompt.enabled=false" "inference.prompt.source=")
        else
            local prev="${root}/iter_$((t - 1))/prompts.txt"
            [[ -f "$prev" ]] || { log "SM1 FAILED: missing ${prev}"; return 1; }
            opts+=("inference.prompt.enabled=true" "inference.prompt.source=${prev}"
                   "inference.prompt.strict_window_prompt=false"
                   "inference.prompt.max_prompt_per_window=5")
        fi

        log "SM1 t=${t} (M1: TDOM, no prior)"
        CONFIG_PATH="$DOM_CONFIG" run_pipeline "${opts[@]}"
    done
    log "SM1 done -> ${root}"
}

# ----------------------------------------------------- SM2 / SM3 baselines
# M2 = perspective MV only, pixel voting.
# M3 = same grid plus the TDOM mask burned in (pixel-level late fusion); its
#      TDOM input is M1's result for the same round, per the agreed protocol.
#
# Both use global/affine projection and no prior, so they cannot reuse S1's
# proj/ (built with slope_correction + prior). They CAN reuse S1's infer/,
# since inference at t=0 is identical -- only the projection differs.
_m_baseline() {
    local base_name="$1" extra_shp_of_round="$2"
    local name="${base_name}${PIXEL_SUFFIX}"
    local root="${OUT_ROOT}/${name}"
    # Deliberately unsuffixed: the projection depends on the projection method,
    # not on the fusion knobs, so a re-tuned run reuses the existing cache
    # instead of spending hours reprojecting identical geometry.
    local base_proj="${OUT_ROOT}/${base_name}/shared_proj"

    # One-off reprojection of the shared raw cache under baseline settings.
    if ! compgen -G "${base_proj}/*.shp" > /dev/null; then
        mkdir -p "$base_proj"
        log "${name} reprojecting S1 raw cache (affine, no prior)"
        run_pipeline \
            "pipeline.run_inference=false" \
            "pipeline.run_projection=true" \
            "pipeline.run_postprocess=false" \
            "data.data_root=${DATA_ROOT}" \
            "data.image_glob=${IMAGE_GLOB}" \
            "output.per_image_raw_shp_dir=${SHARED_INFER}" \
            "output.per_image_shp_dir=${base_proj}" \
            "output.final_merged_shp=${root}/_placeholder.shp" \
            "${GLOBAL_PROJ_OPTS[@]}" "${NO_PRIOR_OPTS[@]}"
    fi

    for ((t = 0; t <= ITER_MAX; t++)); do
        local tdir="${root}/iter_${t}"
        local final="${tdir}/final.shp"
        if [[ "$SKIP_EXISTING" == "1" && -f "$final" ]]; then
            log "${name} t=${t} exists, skip"; continue
        fi
        mkdir -p "${tdir}/prompts"

        local -a opts=(
            "pipeline.run_postprocess=true"
            "postprocess.view_selection.enabled=false"
            "postprocess.dom_merge.enabled=false"
            "postprocess.multiview.enabled=true"
            "postprocess.multiview.strategy=pixel_vote"
            "postprocess.multiview.cell_size=${PIXEL_CELL_SIZE}"
            "postprocess.multiview.min_votes=${PIXEL_MIN_VOTES}"
            "postprocess.multiview.min_area=${PIXEL_MIN_AREA}"
            "postprocess.prompt_export.enabled=true"
            "postprocess.prompt_export.mode=oblique"
            "postprocess.prompt_export.output_dir=${tdir}/prompts"
            "output.trace_exports.enabled=false"
            "output.final_merged_shp=${final}"
            "${GLOBAL_PROJ_OPTS[@]}" "${NO_PRIOR_OPTS[@]}"
        )

        if [[ -n "$extra_shp_of_round" ]]; then
            local extra="${OUT_ROOT}/m1/iter_${t}/final.shp"
            [[ -f "$extra" ]] || { log "${name} FAILED: missing TDOM branch ${extra} (run SM1 first)"; return 1; }
            opts+=("postprocess.multiview.extra_features_shp=${extra}")
        fi

        if [[ "$t" -eq 0 ]]; then
            opts+=("distributed.enabled=false" "pipeline.run_inference=false"
                   "pipeline.run_projection=false"
                   "output.per_image_raw_shp_dir=${SHARED_INFER}"
                   "output.per_image_shp_dir=${base_proj}"
                   "inference.prompt.enabled=false" "inference.prompt.source=")
        else
            opts+=("pipeline.run_inference=true" "pipeline.run_projection=true"
                   "data.data_root=${DATA_ROOT}" "data.image_glob=${IMAGE_GLOB}"
                   "inference.prompt.enabled=true"
                   "inference.prompt.source=${root}/iter_$((t - 1))/prompts"
                   "inference.prompt.strict_window_prompt=false"
                   "inference.prompt.max_prompt_per_window=5"
                   "output.per_image_raw_shp_dir=${tdir}/infer"
                   "output.per_image_shp_dir=${tdir}/proj")
        fi

        log "${name} t=${t}"
        run_pipeline "${opts[@]}"
    done
    log "${name} done -> ${root}"
}

stage_SM2() { _m_baseline "m2" ""; }
stage_SM3() { _m_baseline "m3" "m1"; }

# ------------------------------------------------------- SD TDOM path (M1)
# DOM baseline t=0 plus its own feedback iterations. Produces two things the
# rest of the study depends on:
#   - table 1/2 row M1 (SAM3-TDOM-Iter)
#   - the TDOM input that Ours / M3 fuse with the perspective branch
#
# The DOM is a single huge raster and split_items_for_rank splits by image, so
# extra ranks buy nothing here -- always single process.
stage_SD() {
    local dom_cfg="${DOM_CONFIG}"
    local dom_root="${OUT_ROOT}/dom"

    for ((t = 0; t <= ITER_MAX; t++)); do
        local tdir="${dom_root}/iter_${t}"
        local final="${tdir}/final.shp"
        local prompts="${tdir}/prompts.txt"

        if [[ "$SKIP_EXISTING" == "1" && -f "$final" ]]; then
            log "SD t=${t} exists, skip"
            continue
        fi
        mkdir -p "$tdir"

        local -a opts=(
            "distributed.enabled=false"
            "pipeline.run_inference=true"
            "pipeline.run_projection=true"
            "pipeline.run_postprocess=true"
            "postprocess.view_selection.enabled=false"
            "postprocess.dom_merge.enabled=false"
            "postprocess.prompt_export.enabled=true"
            "postprocess.prompt_export.mode=dom"
            "postprocess.prompt_export.output_txt=${prompts}"
            "output.trace_exports.enabled=false"
            "output.per_image_raw_shp_dir=${tdir}/infer"
            "output.per_image_shp_dir=${tdir}/proj"
            "output.final_merged_shp=${final}"
        )

        if [[ "$t" -eq 0 ]]; then
            opts+=("inference.prompt.enabled=false" "inference.prompt.source=")
        else
            local prev_prompts="${dom_root}/iter_$((t - 1))/prompts.txt"
            if [[ ! -f "$prev_prompts" ]]; then
                log "SD FAILED: missing prompts from t=$((t - 1)) (${prev_prompts})"
                return 1
            fi
            opts+=(
                "inference.prompt.enabled=true"
                "inference.prompt.source=${prev_prompts}"
                "inference.prompt.strict_window_prompt=false"
                "inference.prompt.max_prompt_per_window=5"
            )
        fi

        log "SD t=${t} (DOM)"
        CONFIG_PATH="$dom_cfg" run_pipeline "${opts[@]}"
    done
    log "SD done -> ${dom_root}"
}

# --------------------------------------------- SP Ours / P, dual-source loop
# The headline method: TDOM + perspective MV, fused in object space, with BOTH
# sources re-prompted each round (draft table 3 "Full dual-source feedback").
#
# One round is two pipeline invocations:
#   (a) DOM re-inferred from the previous round's DOM prompts   -> dom/final.shp
#   (b) perspective re-inferred from the previous round's oblique prompts, then
#       fused and dom_merge'd against (a)                       -> final.shp
# Round t=0 skips (a) and reuses the SD baseline, so Ours does not inherit M1's
# iterative gain -- the TDOM prior it starts from is the unrefined t=0 DOM.
stage_SP() {
    local ours_root="${OUT_ROOT}/ours"
    local dom_image="${DATA_ROOT}dom/DOM.tif"
    local sd_baseline="${OUT_ROOT}/dom/iter_0/final.shp"

    if [[ ! -f "$sd_baseline" ]]; then
        log "SP FAILED: missing DOM baseline (${sd_baseline}). Run SD first."
        return 1
    fi
    if [[ ! -f "$dom_image" ]]; then
        log "SP FAILED: missing DOM raster ${dom_image}"
        return 1
    fi

    for ((t = 0; t <= ITER_MAX; t++)); do
        local tdir="${ours_root}/iter_${t}"
        local final="${tdir}/final.shp"
        local dom_branch="${tdir}/dom/final.shp"

        if [[ "$SKIP_EXISTING" == "1" && -f "$final" ]]; then
            log "SP t=${t} exists, skip"
            continue
        fi
        mkdir -p "${tdir}/prompts" "${tdir}/dom"

        # --- (a) the TDOM branch for this round ---------------------------
        if [[ "$t" -eq 0 ]]; then
            dom_branch="$sd_baseline"
            log "SP t=0 reusing SD baseline as TDOM prior"
        else
            local prev_dom_prompts="${ours_root}/iter_$((t - 1))/dom_prompts.txt"
            if [[ ! -f "$prev_dom_prompts" ]]; then
                log "SP FAILED: missing DOM prompts from t=$((t - 1))"
                return 1
            fi
            if [[ "$SKIP_EXISTING" == "1" && -f "$dom_branch" ]]; then
                log "SP t=${t} DOM branch exists, skip"
            else
                log "SP t=${t} (a) DOM re-inference"
                CONFIG_PATH="$DOM_CONFIG" run_pipeline \
                    "distributed.enabled=false" \
                    "pipeline.run_inference=true" \
                    "pipeline.run_projection=true" \
                    "pipeline.run_postprocess=true" \
                    "inference.prompt.enabled=true" \
                    "inference.prompt.source=${prev_dom_prompts}" \
                    "inference.prompt.strict_window_prompt=false" \
                    "inference.prompt.max_prompt_per_window=5" \
                    "postprocess.view_selection.enabled=false" \
                    "postprocess.dom_merge.enabled=false" \
                    "postprocess.prompt_export.enabled=false" \
                    "output.trace_exports.enabled=false" \
                    "output.per_image_raw_shp_dir=${tdir}/dom/infer" \
                    "output.per_image_shp_dir=${tdir}/dom/proj" \
                    "output.final_merged_shp=${dom_branch}"
            fi
        fi

        # --- (b) perspective branch + object-space fusion with TDOM -------
        local -a opts=(
            "pipeline.run_postprocess=true"
            "data.dom_shp=${dom_branch}"
            "postprocess.view_selection.enabled=false"
            "postprocess.dom_merge.enabled=true"
            "postprocess.dom_merge.strategy=confidence"
            "postprocess.prompt_export.enabled=true"
            "postprocess.prompt_export.mode=both"
            "postprocess.prompt_export.output_dir=${tdir}/prompts"
            "postprocess.prompt_export.output_txt=${tdir}/dom_prompts.txt"
            "postprocess.prompt_export.dom_image=${dom_image}"
            "output.trace_exports.enabled=false"
            "output.final_merged_shp=${final}"
        )

        if [[ "$t" -eq 0 ]]; then
            # Reuse the S1 cache: t=0 needs no perspective inference at all.
            opts+=(
                "distributed.enabled=false"
                "pipeline.run_inference=false"
                "pipeline.run_projection=false"
                "output.per_image_raw_shp_dir=${SHARED_INFER}"
                "output.per_image_shp_dir=${SHARED_PROJ}"
                "inference.prompt.enabled=false"
                "inference.prompt.source="
            )
        else
            local prev_prompts="${ours_root}/iter_$((t - 1))/prompts"
            opts+=(
                "pipeline.run_inference=true"
                "pipeline.run_projection=true"
                "data.data_root=${DATA_ROOT}"
                "data.image_glob=${IMAGE_GLOB}"
                "inference.prompt.enabled=true"
                "inference.prompt.source=${prev_prompts}"
                "inference.prompt.strict_window_prompt=false"
                "inference.prompt.max_prompt_per_window=5"
                "output.per_image_raw_shp_dir=${tdir}/infer"
                "output.per_image_shp_dir=${tdir}/proj"
            )
        fi

        log "SP t=${t} (b) perspective fusion + dom_merge"
        run_pipeline "${opts[@]}"
    done
    log "SP done -> ${ours_root}"
}

# ------------------------------------------------------ S2 iter_0 postprocess
# Pure postprocess over the S1 cache: cheap, CPU-only, no NCCL. Produces the
# t=0 result and the bbox prompts that S3 feeds back.
#
# view_selection is off throughout: the multi-view axis of this study is the
# camera-direction set (table 7), not a top-N view count, so every valid image
# of the selected directions contributes to fusion.
stage_S2() {
    local final="${MAIN_DIR}/final.shp"
    if [[ "$SKIP_EXISTING" == "1" && -f "$final" ]]; then
        log "S2 iter_0 postprocess exists, skip"
        return
    fi
    mkdir -p "${MAIN_DIR}/prompts"
    log "S2 iter_0 postprocess (all directions, view_selection off)"
    run_pipeline \
        "distributed.enabled=false" \
        "pipeline.run_inference=false" \
        "pipeline.run_projection=false" \
        "pipeline.run_postprocess=true" \
        "output.per_image_raw_shp_dir=${SHARED_INFER}" \
        "output.per_image_shp_dir=${SHARED_PROJ}" \
        "postprocess.view_selection.enabled=false" \
        "postprocess.prompt_export.enabled=true" \
        "postprocess.prompt_export.output_dir=${MAIN_DIR}/prompts" \
        "inference.prompt.enabled=false" \
        "inference.prompt.source=" \
        "output.trace_exports.enabled=false" \
        "output.final_merged_shp=${final}"
    log "S2 done -> ${final}"
}

# ------------------------------------------------- S3 main iterations t=1..N
# Each round re-infers with the previous round's prompts, so unlike S2 this
# cannot reuse the shared cache.
stage_S3() {
    log "S3 iterations t=1..${ITER_MAX} (all directions)"
    for ((t = 1; t <= ITER_MAX; t++)); do
        local prev=$((t - 1))
        local prev_prompts="${OUT_ROOT}/iter_${prev}/full/prompts"
        local vdir="${OUT_ROOT}/iter_${t}/full"
        local final="${vdir}/final.shp"

        if [[ "$SKIP_EXISTING" == "1" && -f "$final" ]]; then
            log "S3 t=${t} exists, skip"
            continue
        fi
        if ! compgen -G "${prev_prompts}/*" > /dev/null; then
            log "S3 FAILED: missing prompts from t=${prev} (${prev_prompts}). Run S2 first."
            return 1
        fi

        mkdir -p "${vdir}/prompts" "${vdir}/infer" "${vdir}/proj"
        log "S3 t=${t}"
        run_pipeline \
            "pipeline.run_inference=true" \
            "pipeline.run_projection=true" \
            "pipeline.run_postprocess=true" \
            "data.data_root=${DATA_ROOT}" \
            "data.image_glob=${IMAGE_GLOB}" \
            "inference.prompt.enabled=true" \
            "inference.prompt.source=${prev_prompts}" \
            "inference.prompt.strict_window_prompt=false" \
            "inference.prompt.max_prompt_per_window=5" \
            "postprocess.view_selection.enabled=false" \
            "postprocess.prompt_export.enabled=true" \
            "postprocess.prompt_export.output_dir=${vdir}/prompts" \
            "output.trace_exports.enabled=false" \
            "output.per_image_raw_shp_dir=${vdir}/infer" \
            "output.per_image_shp_dir=${vdir}/proj" \
            "output.final_merged_shp=${final}"
    done
    log "S3 done"
}

# -------------------------------------------- S4 table 7: camera-direction sets
# Six cumulative view sets (TDOM only / +Nadir / +O1..+O4). t=0 costs nothing --
# it just postprocesses a symlinked subset of the S1 cache.
stage_S4() {
    local dirs_root="${OUT_ROOT}/iter_0/dirs"
    log "S4 building direction subsets"
    "$PYTHON_BIN" "$ROOT_DIR/scripts/split_directions.py" \
        --station "$STATION" --data-root "$DATA_ROOT" --image-glob "$IMAGE_GLOB" \
        --emit links --proj-dir "$SHARED_PROJ" --out-root "$dirs_root"

    for setdir in "${dirs_root}"/*/; do
        local name final
        name="$(basename "$setdir")"
        final="${setdir}/final.shp"
        # d0_tdom carries no perspective images; it is the DOM-only row.
        if ! compgen -G "${setdir}/proj/*.shp" > /dev/null; then
            log "S4 ${name}: no perspective images, skip"
            continue
        fi
        if [[ "$SKIP_EXISTING" == "1" && -f "$final" ]]; then
            log "S4 ${name} exists, skip"
            continue
        fi
        mkdir -p "${setdir}/prompts"
        log "S4 ${name}"
        run_pipeline \
            "distributed.enabled=false" \
            "pipeline.run_inference=false" \
            "pipeline.run_projection=false" \
            "pipeline.run_postprocess=true" \
            "output.per_image_raw_shp_dir=${SHARED_INFER}" \
            "output.per_image_shp_dir=${setdir}/proj" \
            "postprocess.view_selection.enabled=false" \
            "postprocess.prompt_export.enabled=true" \
            "postprocess.prompt_export.output_dir=${setdir}/prompts" \
            "inference.prompt.enabled=false" \
            "inference.prompt.source=" \
            "output.trace_exports.enabled=false" \
            "output.final_merged_shp=${final}"
    done
    log "S4 done"
}

# ============================================================================
# Ablation stages. All three are pinned to t=1 per experiments_plan.md section 6
# -- running them to convergence would triple the total, and t=1 already shows
# the effect. Each writes <name>/iter_<t>/final.shp, the layout the evaluator's
# discovery and collect_run_stats.py both understand.
# ============================================================================

# ---------------------------------------------- S5 table 4: geometry-prior
# Drop one sub-score at a time; the remaining weights renormalise on their own
# (projection/scoring.py divides by their sum). Everything else stays at the
# Ours setting: local projection, instance fusion, dual-source feedback.
#
# The weights act during projection+scoring, so t=0 does not need inference --
# only a reprojection of the shared raw cache. t=1 does need a full pass,
# because the prompts differ once the t=0 ranking changes.
#
# The "No module-geometry prior" row instead ranks on raw SAM3 confidence, which
# is what NO_PRIOR_OPTS expresses.
_prior_ablation() {
    local name="$1"; shift
    local -a weight_opts=("$@")
    local root="${OUT_ROOT}/${name}"
    local proj_dir="${root}/shared_proj"

    # (1) t=0: reproject the shared cache under the ablated weights.
    if ! compgen -G "${proj_dir}/*.shp" > /dev/null; then
        mkdir -p "$proj_dir"
        log "${name} reprojecting shared cache under ablated weights"
        run_pipeline \
            "distributed.enabled=false" \
            "pipeline.run_inference=false" \
            "pipeline.run_projection=true" \
            "pipeline.run_postprocess=false" \
            "data.data_root=${DATA_ROOT}" \
            "data.image_glob=${IMAGE_GLOB}" \
            "output.per_image_raw_shp_dir=${SHARED_INFER}" \
            "output.per_image_shp_dir=${proj_dir}" \
            "output.final_merged_shp=${root}/_placeholder.shp" \
            "${weight_opts[@]}"
    fi

    for ((t = 0; t <= 1; t++)); do
        local tdir="${root}/iter_${t}"
        local final="${tdir}/final.shp"
        if [[ "$SKIP_EXISTING" == "1" && -f "$final" ]]; then
            log "${name} t=${t} exists, skip"; continue
        fi
        mkdir -p "${tdir}/prompts"

        local -a opts=(
            "pipeline.run_postprocess=true"
            "postprocess.view_selection.enabled=false"
            "postprocess.prompt_export.enabled=true"
            "postprocess.prompt_export.output_dir=${tdir}/prompts"
            "output.trace_exports.enabled=false"
            "output.final_merged_shp=${final}"
            "${weight_opts[@]}"
        )
        if [[ "$t" -eq 0 ]]; then
            opts+=("distributed.enabled=false" "pipeline.run_inference=false"
                   "pipeline.run_projection=false"
                   "output.per_image_raw_shp_dir=${SHARED_INFER}"
                   "output.per_image_shp_dir=${proj_dir}"
                   "inference.prompt.enabled=false" "inference.prompt.source=")
        else
            opts+=("pipeline.run_inference=true" "pipeline.run_projection=true"
                   "data.data_root=${DATA_ROOT}" "data.image_glob=${IMAGE_GLOB}"
                   "inference.prompt.enabled=true"
                   "inference.prompt.source=${root}/iter_0/prompts"
                   "inference.prompt.strict_window_prompt=false"
                   "inference.prompt.max_prompt_per_window=5"
                   "output.per_image_raw_shp_dir=${tdir}/infer"
                   "output.per_image_shp_dir=${tdir}/proj")
        fi

        log "${name} t=${t}"
        run_pipeline "${opts[@]}"
    done
    log "${name} done -> ${root}"
}

# The four configurations are independent, so S5 can be split across cards by
# running two invocations with disjoint S5_CONFIGS. That parallelises the whole
# stage, not just inference: each configuration spends ~40 min in a
# single-process CPU postprocess (t=0) that extra ranks do nothing for.
#
#   GPUS=0 MASTER_PORT=29501 S5_CONFIGS="noprior no_area"  ... S5
#   GPUS=1 MASTER_PORT=29503 S5_CONFIGS="no_ratio no_shape" ... S5
stage_S5() {
    # Deliberately unquoted: S5_CONFIGS is a space-separated selection.
    for name in $S5_CONFIGS; do
        case "$name" in
            noprior)  _prior_ablation "abl_noprior"  "${NO_PRIOR_OPTS[@]}" ;;
            no_area)  _prior_ablation "abl_no_area"  "projection.score.w_area=0.0" ;;
            no_ratio) _prior_ablation "abl_no_ratio" "projection.score.w_ratio=0.0" ;;
            no_shape) _prior_ablation "abl_no_shape" "projection.score.w_shape=0.0" ;;
            # Cumulative view of table 4: keep only rectangularity. The next rung
            # up ("area + rectangularity") needs no run of its own -- it is
            # w_ratio=0, i.e. the existing abl_no_ratio.
            only_shape)
                _prior_ablation "abl_only_shape" \
                    "projection.score.w_area=0.0" "projection.score.w_ratio=0.0" ;;
            # The reverse ordering of the cumulative curve. Adding rectangularity
            # first makes it look inert, because RQ saturates the moment area
            # joins; area-first shows what rectangularity is still worth on top.
            only_area)
                _prior_ablation "abl_only_area" \
                    "projection.score.w_ratio=0.0" "projection.score.w_shape=0.0" ;;
            *)
                log "S5 FAILED: unknown ablation '${name}' (expected one of: noprior no_area no_ratio no_shape only_shape only_area)"
                return 1
                ;;
        esac
    done
    log "S5 done for '${S5_CONFIGS}' (table 4; Full-prior row is ours/iter_1)"
}

# ------------------------------------- S6 table 3: TDOM-only feedback path
# Row 2 of table 3, the only one the suite never ran. The t=0 candidate set is
# dual-source (identical to Ours t=0, which is why the SD baseline and the
# shared cache are reused verbatim); the difference is what gets re-prompted
# afterwards: TDOM is re-inferred from its own prompts, the perspective side is
# NOT, so the fusion at t=1 sees refreshed TDOM against unchanged MV evidence.
stage_S6() {
    local root="${OUT_ROOT}/fb_tdom_only"
    local dom_image="${DATA_ROOT}dom/DOM.tif"
    local ours_t0="${OUT_ROOT}/ours/iter_0"

    if [[ ! -f "${ours_t0}/final.shp" ]]; then
        log "S6 FAILED: missing Ours t=0 (${ours_t0}/final.shp). Run SP first."
        return 1
    fi
    if [[ ! -f "${ours_t0}/dom_prompts.txt" ]]; then
        log "S6 FAILED: missing DOM prompts from Ours t=0"
        return 1
    fi

    local tdir="${root}/iter_1"
    local final="${tdir}/final.shp"
    if [[ "$SKIP_EXISTING" == "1" && -f "$final" ]]; then
        log "S6 t=1 exists, skip"
        return 0
    fi
    mkdir -p "${tdir}/prompts" "${tdir}/dom"

    # (a) TDOM branch re-inferred from Ours' t=0 DOM prompts.
    local dom_branch="${tdir}/dom/final.shp"
    if [[ "$SKIP_EXISTING" != "1" || ! -f "$dom_branch" ]]; then
        log "S6 (a) DOM re-inference"
        CONFIG_PATH="$DOM_CONFIG" run_pipeline \
            "distributed.enabled=false" \
            "pipeline.run_inference=true" \
            "pipeline.run_projection=true" \
            "pipeline.run_postprocess=true" \
            "inference.prompt.enabled=true" \
            "inference.prompt.source=${ours_t0}/dom_prompts.txt" \
            "inference.prompt.strict_window_prompt=false" \
            "inference.prompt.max_prompt_per_window=5" \
            "postprocess.view_selection.enabled=false" \
            "postprocess.dom_merge.enabled=false" \
            "postprocess.prompt_export.enabled=false" \
            "output.trace_exports.enabled=false" \
            "output.per_image_raw_shp_dir=${tdir}/dom/infer" \
            "output.per_image_shp_dir=${tdir}/dom/proj" \
            "output.final_merged_shp=${dom_branch}"
    fi

    # (b) Perspective side deliberately NOT re-inferred: reuse the t=0 cache.
    log "S6 (b) fuse refreshed TDOM against unchanged perspective evidence"
    run_pipeline \
        "distributed.enabled=false" \
        "pipeline.run_inference=false" \
        "pipeline.run_projection=false" \
        "pipeline.run_postprocess=true" \
        "data.dom_shp=${dom_branch}" \
        "output.per_image_raw_shp_dir=${SHARED_INFER}" \
        "output.per_image_shp_dir=${SHARED_PROJ}" \
        "inference.prompt.enabled=false" \
        "inference.prompt.source=" \
        "postprocess.view_selection.enabled=false" \
        "postprocess.dom_merge.enabled=true" \
        "postprocess.dom_merge.strategy=confidence" \
        "postprocess.prompt_export.enabled=true" \
        "postprocess.prompt_export.mode=both" \
        "postprocess.prompt_export.output_dir=${tdir}/prompts" \
        "postprocess.prompt_export.output_txt=${tdir}/dom_prompts.txt" \
        "postprocess.prompt_export.dom_image=${dom_image}" \
        "output.trace_exports.enabled=false" \
        "output.final_merged_shp=${final}"
    log "S6 done -> ${root}"
}

# --------------------------------------------- S7 table 7 at t=1 (feedback)
# S4 produced the six direction sets at t=0 only, but the protocol
# (experiments_plan.md S4) calls for one feedback round each. Inference is
# restricted to the subset's own images via the per-set image list, so the cost
# is the sum of the subsets, not six full passes.
#
# data.image_glob must be cleared alongside: io_flow/input_resolver.py takes the
# *union* of image_path, image_glob and image_list_file rather than the first
# one set, so leaving the station config's glob in place silently restores the
# full image set -- the run looks healthy and reports total_images=328 for a
# 62-image subset.
stage_S7() {
    local dirs_root="${OUT_ROOT}/iter_0/dirs"
    local lists_root="${OUT_ROOT}/_dir_lists"

    mkdir -p "$lists_root"
    "$PYTHON_BIN" "$ROOT_DIR/scripts/split_directions.py" \
        --station "$STATION" --data-root "$DATA_ROOT" --image-glob "$IMAGE_GLOB" \
        --emit lists --out-root "$lists_root"

    for setdir in "${dirs_root}"/*/; do
        local name t0_prompts list_file tdir final
        name="$(basename "$setdir")"
        t0_prompts="${setdir}/prompts"
        list_file="${lists_root}/${name}.txt"
        tdir="${OUT_ROOT}/iter_1/dirs/${name}"
        final="${tdir}/final.shp"

        if [[ ! -f "${setdir}/final.shp" ]]; then
            log "S7 ${name}: no t=0 result, skip"; continue
        fi
        if [[ ! -f "$list_file" ]]; then
            log "S7 ${name}: no image list (${list_file}), skip"; continue
        fi
        if [[ "$SKIP_EXISTING" == "1" && -f "$final" ]]; then
            log "S7 ${name} exists, skip"; continue
        fi

        mkdir -p "${tdir}/prompts" "${tdir}/infer" "${tdir}/proj"
        log "S7 ${name} t=1 ($(wc -l < "$list_file") images)"
        run_pipeline \
            "pipeline.run_inference=true" \
            "pipeline.run_projection=true" \
            "pipeline.run_postprocess=true" \
            "data.data_root=${DATA_ROOT}" \
            "data.image_list_file=${list_file}" \
            "data.image_glob=" \
            "inference.prompt.enabled=true" \
            "inference.prompt.source=${t0_prompts}" \
            "inference.prompt.strict_window_prompt=false" \
            "inference.prompt.max_prompt_per_window=5" \
            "postprocess.view_selection.enabled=false" \
            "postprocess.prompt_export.enabled=true" \
            "postprocess.prompt_export.output_dir=${tdir}/prompts" \
            "output.trace_exports.enabled=false" \
            "output.per_image_raw_shp_dir=${tdir}/infer" \
            "output.per_image_shp_dir=${tdir}/proj" \
            "output.final_merged_shp=${final}"
    done
    log "S7 done"
}

for stage in "${STAGES[@]}"; do
    case "$stage" in
        S0|S1|S1R|S1P|SD|SP|SM1|SM2|SM3|S2|S3|S4|S5|S6|S7)
            "stage_${stage}" 2>&1 | tee -a "${LOG_DIR}/${stage}${STAGE_LOG_SUFFIX}.log"
            ;;
        *)
            echo "unknown stage: $stage (expected S0|S1|S1R|S1P|SD|SP|SM1|SM2|SM3|S2|S3|S4|S5|S6|S7)" >&2
            exit 1
            ;;
    esac
done

log "stages finished: ${STAGES[*]}"
