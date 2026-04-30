# 🛰️ MultiView-PV: Multi-View Photovoltaic Panel Detection Pipeline

A refactored, production-ready pipeline for detecting and segmenting **photovoltaic (PV) panels** from aerial imagery. The pipeline supports both Digital Orthomosaic (DOM) and oblique imagery, with end-to-end geospatial projection, multi-view confidence scoring, and iterative fusion workflows.

---

## ✨ Features

- **Three-stage pipeline**: Inference → Geospatial Projection → Multi-view Postprocessing
- **Dual imagery support**: DOM (geotransform-based) and oblique (collinearity + affine, DSM + POS)
- **Model-registry architecture**: Clean adapter pattern for swapping segmentation models
- **Multi-GPU inference**: Rank-level image splitting via `torchrun`
- **Sliding-window batch inference**: Handles large-format aerial tiles efficiently
- **Iterative multi-view fusion**: Progressively refine detections across views with prompt-loop support
- **Config-first design**: YAML inheritance with CLI override via dotlist syntax

---

## 🏗️ Architecture

```
Input Images (DOM / Oblique)
         │
         ▼
┌─────────────────────┐
│   1. Inference       │  Sliding-window batch inference (SAM3)
│   runner.py          │  Text prompt + geometry prompt support
└────────┬────────────┘
         │  pixel-space masks / polygons
         ▼
┌─────────────────────┐
│  2. Projection &     │  DOM: geotransform
│     PV Scoring       │  Oblique: collinearity + affine (DSM + POS)
│   projector.py       │  Outputs: con_sem, con_pv, con_weight
└────────┬────────────┘
         │  geo-space features (shapefile)
         ▼
┌─────────────────────┐
│  3. Postprocess      │  Per-image NMS → View selection →
│   merge.py           │  Multi-view fusion → DOM merge
└────────┬────────────┘
         │
         ▼
   Final Shapefile (.shp)
```

### Confidence Fields

Each output feature carries three confidence scores:

| Field | Description |
|-------|-------------|
| `con_sem` | Semantic confidence from detector score |
| `con_pv` | PV geometry confidence (area / ratio / shape scoring) |
| `con_weight` | Weighted sum of `con_sem` and `con_pv` |
| `con` | Alias for `con_weight` (backward compatibility) |

---

## 📦 Requirements

- Python ≥ 3.8
- PyTorch (with CUDA recommended)
- GDAL / OGR
- OpenCV
- Pillow
- OmegaConf
- tqdm
- SAM3 (submodule: `third_party/sam3`)

### Install

```bash
# Clone with submodules
git clone --recurse-submodules <repo-url>
cd MultiView-PV

# Install dependencies (conda recommended)
conda install gdal
pip install torch torchvision opencv-python pillow omegaconf tqdm

# Install SAM3
cd third_party/sam3
pip install -e .
cd ../..
```

---

## 🚀 Quick Start

### Single GPU

```bash
python cli/run_pipeline.py --config configs/station_template.yaml
```

### Multi-GPU (torchrun)

```bash
torchrun --nproc_per_node=2 cli/run_pipeline.py \
  --config configs/station_template.yaml
```

### CLI Config Overrides

Any config field can be overridden at runtime using dotlist syntax:

```bash
python cli/run_pipeline.py \
  --config configs/station_template.yaml \
  inference.dataloader.batch_size=8 \
  output.final_merged_shp=outputs/experiment1/final.shp
```

---

## ⚙️ Configuration

Configs follow a three-level inheritance: `_base.yaml` → template → station instance.

### Config Templates

| Template | Use Case |
|----------|----------|
| `configs/_base.yaml` | Base defaults (do not edit directly) |
| `configs/station_template.yaml` | Generic starting point for any station |
| `configs/dom_mode_template.yaml` | DOM imagery with geotransform |
| `configs/oblique_mode_template.yaml` | Oblique imagery (requires pose CSV + DSM) |
| `configs/legacy_compat_template.yaml` | Approximate legacy pipeline behavior |
| `configs/oblique_iterative_merge_template.yaml` | Iterative multi-view fusion workflow |

### Projection Modes

**DOM mode** (`projection.mode=dom`):
- Input: DOM image with embedded geotransform
- Projection: direct raster georeference

**Oblique mode** (`projection.mode=oblique`):
- Input: Oblique imagery + POS CSV + DSM (GeoTIFF)
- Projection: collinearity equations → affine transform fitting per feature
- Fallback: direct collinearity if affine is under-constrained
- Optional: slope correction (`enable_slope_correction`)

### NMS Policy

| Stage | DOM | Oblique |
|-------|-----|---------|
| Before projection | ✗ No NMS | ✗ No NMS |
| After projection (postprocess) | Configurable | Configurable |

---

## 🔁 Iterative Multi-View Workflow

The pipeline supports iterative fusion to progressively improve recall and precision across multiple views.

### Manual Iteration

**Round 0 — Baseline:**

```bash
# Step 1: DOM baseline
python cli/run_pipeline.py --config configs/dom_mode_template.yaml

# Step 2: Oblique baseline
python cli/run_pipeline.py --config configs/oblique_mode_template.yaml
```

**Round 1..N — Fusion:**

```bash
python cli/run_pipeline.py \
  --config configs/oblique_mode_template.yaml \
  data.dom_shp=/path/to/round0_fused.shp \
  postprocess.dom_merge.enabled=true \
  output.final_merged_shp=outputs/iter1/fused.shp
```

Repeat, replacing `data.dom_shp` with the latest fused result each round.

### Automated Iterative Script

```bash
./scripts/run_dom_oblique_iter.sh \
  --dom-config configs/dom_mode_template.yaml \
  --oblique-config configs/oblique_mode_template.yaml \
  --rounds 3 \
  --exp-dir outputs/iter_experiment \
  --python /path/to/python
```

The script executes:
1. DOM baseline
2. Oblique baseline
3. N rounds of oblique fusion, each using the previous round's output as `data.dom_shp`

### Iterative Script with Legacy View/Prompt Loop

```bash
./scripts/run_dom_oblique_iter.sh \
  --dom-config configs/dom_mode_template.yaml \
  --oblique-config configs/legacy_compat_template.yaml \
  --rounds 3 \
  --exp-dir outputs/iter_legacy_loop \
  --enable-legacy-view-loop 1 \
  --gt-shp /path/to/gt/pv.shp \
  --index-name mapping/station_pv_index \
  --view-num 5 \
  --pose-csv /path/to/CAM/pose.csv \
  --dsm-path /path/to/dsm/DSM.tif
```

When `--enable-legacy-view-loop 1` is set, each round additionally runs:
1. `postprocess/select_views.py` — view selection on previous fused result
2. `mapping/generate_bbox_prompt.py` — generate per-image prompt bounding boxes
3. Oblique inference with `inference.prompt.enabled=true` and strict prompt windows

---

## 🗂️ Project Structure

```
MultiView-PV/
├── cli/                        # Entry points
│   └── run_pipeline.py
├── configs/                    # YAML config templates
│   ├── _base.yaml
│   ├── station_template.yaml
│   ├── dom_mode_template.yaml
│   ├── oblique_mode_template.yaml
│   ├── legacy_compat_template.yaml
│   └── oblique_iterative_merge_template.yaml
├── inference/                  # Stage 1: model inference
│   ├── runner.py               # Sliding-window batch inference
│   ├── models/                 # Model adapters (SAM3, etc.)
│   └── mask_utils.py
├── projection/                 # Stage 2: geospatial projection & scoring
│   ├── projector.py            # Dual projection entry + confidence scoring
│   ├── oblique_projector.py    # Oblique: DSM + POS collinearity
│   └── collinearity.py         # Collinearity and affine utilities
├── postprocess/                # Stage 3: fusion and merge
│   ├── merge.py                # Multi-view and DOM merge strategies
│   └── select_views.py         # Best-view selection
├── io_flow/                    # I/O utilities
│   └── shp_io.py               # Shapefile read/write with confidence fields
├── utils/                      # Shared utilities
├── scripts/                    # Bash automation scripts
├── tests/                      # Test suite
├── third_party/                # External dependencies
│   └── sam3/                   # SAM3 segmentation model (git submodule)
└── pipeline.py                 # Pipeline orchestration
```

---

## 🔧 Legacy Compatibility

For users migrating from the legacy pipeline:

- **`configs/legacy_compat_template.yaml`**: Ready-to-use config approximating legacy behavior:
  - Strict prompt windows (`inference.prompt.strict_window_prompt=true`)
  - Legacy geometry score mode (`projection.score.mode=legacy_gaussian`)
  - Multi-view / DOM merge IoU set to `0.25`

- **Score mode**: `projection.score.mode`
  - `standard` (default): length/width-derived target scoring
  - `legacy_gaussian`: area/ratio/shape scoring matching legacy `process.py`

---

## 📄 License

This project is intended for research purposes. Please cite appropriately if used in publications.

---

## 🙏 Acknowledgements

- [SAM3](https://github.com/facebookresearch/sam3) — Segment Anything Model v3 (Facebook Research)
- GDAL / OGR — Geospatial Data Abstraction Library
