# Refactor V2 Pipeline

This folder contains a refactored pipeline with three explicit stages:

1. Inference
2. Projection and PV scoring
3. Postprocess (multi-view fusion, optional DOM merge)

## Core design

- Inference adapters are model-driven and registry-based.
- `sam3` is implemented.
- `rex_omni` has a placeholder adapter to be filled in.
- DataLoader is used for sliding-window batch inference.
- Multi-GPU inference is supported by rank-level image split (torchrun style env).

## Self-contained refactor code

Key components are copied into `refactor_v2` so the new pipeline does not directly import legacy task code from `utils/` and `models/`:

- config loader: `refactor_v2/config_loader.py`
- mask polygon conversion: `refactor_v2/inference/mask_utils.py`
- SAM3 segmenter: `refactor_v2/inference/models/sam3_segmenter.py`

## Main outputs

Each feature writes:

- `con_sem`: semantic confidence from detector score
- `con_pv`: PV geometry confidence from area/ratio/shape scoring
- `con_weight`: weighted sum of `con_sem` and `con_pv`

Compatibility field:

- `con`: same value as `con_weight`

## Config usage

Base template:

- `refactor_v2/configs/base_refactor.yaml`

Station template:

- `refactor_v2/configs/station_template.yaml`
- `refactor_v2/configs/legacy_compat_template.yaml` (approximate legacy behavior)

## Run commands

Single GPU:

```bash
python main_refactor.py --config refactor_v2/configs/station_template.yaml
```

Override values from CLI:

```bash
python main_refactor.py \
  --config refactor_v2/configs/station_template.yaml \
  inference.dataloader.batch_size=8 \
  output.final_merged_shp=outputs/refactor_v2/tmp/final.shp
```

Multi GPU:

```bash
torchrun --nproc_per_node=2 main_refactor.py \
  --config refactor_v2/configs/station_template.yaml
```

## Important modules

- `refactor_v2/inference/runner.py`: DataLoader sliding-window inference (no NMS before projection)
- `refactor_v2/projection/projector.py`: dual projection entry and `con_pv`/`con_weight`
- `refactor_v2/projection/oblique_projector.py`: oblique projection using DSM + POS (collinearity + affine)
- `refactor_v2/projection/collinearity.py`: copied collinearity and affine utilities
- `refactor_v2/postprocess/merge.py`: multi-view and DOM merge strategies
- `refactor_v2/io/shp_io.py`: shapefile read/write with new confidence fields

## NMS policy

- Before projection: no NMS for both DOM and oblique pipelines.
- After projection: NMS can be used inside postprocess fusion strategy when configured.

## Legacy compatibility knobs

- `inference.prompt.strict_window_prompt`
  - If true and prompt is enabled, only windows with valid prompt boxes are inferred.
  - Useful to mimic prompt-driven window behavior from legacy DOM expansion scripts.

- `projection.score.mode`
  - `standard` (default): length/width-derived target scoring.
  - `legacy_gaussian`: area/ratio/shape scoring closer to legacy `process.py` and `process_dom.py` style.

- `refactor_v2/configs/legacy_compat_template.yaml`
  - Provides a ready-to-use legacy-like parameter set:
    - strict prompt windows
    - legacy geometry score parameters
    - multiview/dom merge IoU set to `0.25`

## Projection modes

- `projection.mode=dom`
  - For DOM imagery with geotransform.
  - Reuses the raster georeference directly.

- `projection.mode=oblique`
  - For oblique imagery.
  - Uses DSM + POS to build control points through collinearity equations.
  - Fits affine transform per feature when enough control points exist.
  - Falls back to direct collinearity projection if affine fitting is under-constrained.
  - Provides a placeholder switch for future slope correction (`enable_slope_correction`).

## DOM vs Oblique split

Use different configs, not different code branches in your scripts:

- DOM config: `refactor_v2/configs/dom_mode_template.yaml`
  - Set `projection.mode=dom`
  - Input is DOM image(s) with geotransform

- Oblique config: `refactor_v2/configs/oblique_mode_template.yaml`
  - Set `projection.mode=oblique`
  - Must provide `projection.oblique.pose_csv` and `projection.oblique.dsm_path`

## Iteration workflow (recommended)

Round 0 (baseline):

1. Run DOM baseline:

   python main_refactor.py --config refactor_v2/configs/dom_mode_template.yaml

2. Run oblique baseline:

   python main_refactor.py --config refactor_v2/configs/oblique_mode_template.yaml

Round 1..N (fusion iteration):

1. Take previous round trusted result as merge prior (DOM result or fused result).
2. Set it to `data.dom_shp` and enable `postprocess.dom_merge.enabled=true`.
3. Run oblique again.

You can use `refactor_v2/configs/oblique_iterative_merge_template.yaml` as the starting point.

Example with CLI override:

python main_refactor.py \
  --config refactor_v2/configs/oblique_mode_template.yaml \
  data.dom_shp=/path/to/round0_dom_or_fused.shp \
  postprocess.dom_merge.enabled=true \
  output.final_merged_shp=outputs/refactor_v2/iter1/final/fused_iter1.shp

Then repeat by replacing `data.dom_shp` with the latest fused result.

## One-command iterative run

You can also run the full loop with script:

./refactor_v2/run_dom_oblique_iter.sh \
  --dom-config refactor_v2/configs/dom_mode_template.yaml \
  --oblique-config refactor_v2/configs/oblique_mode_template.yaml \
  --rounds 3 \
  --exp-dir outputs/refactor_v2/iter_xxx \
  --python /home/haozi/anaconda3/bin/python

The script runs:

1. DOM baseline
2. Oblique baseline
3. Oblique fusion rounds, where each round uses previous fused result as `data.dom_shp`

### Optional legacy view/prompt loop in iterative script

`run_dom_oblique_iter.sh` now supports running legacy view selection and bbox prompt generation before each fusion round.

Example:

```bash
./refactor_v2/run_dom_oblique_iter.sh \
  --dom-config refactor_v2/configs/dom_mode_template.yaml \
  --oblique-config refactor_v2/configs/legacy_compat_template.yaml \
  --rounds 3 \
  --exp-dir outputs/refactor_v2/iter_legacy_loop \
  --enable-legacy-view-loop 1 \
  --gt-shp /path/to/gt/pv.shp \
  --index-name mapping/BeiOu_pv_index \
  --view-num 5 \
  --pose-csv /path/to/CAM/pose.csv \
  --dsm-path /path/to/dsm/DSM.tif
```

When enabled, each round performs:

1. `postprocess/select_views.py` on previous fused result
2. `mapping/generate_bbox_prompt.py` to build per-image prompt txt files
3. oblique run with `inference.prompt.enabled=true`, `inference.prompt.source=<round_prompt_dir>`, and strict prompt windows
