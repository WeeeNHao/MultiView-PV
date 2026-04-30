---
name: multiview-pv-ops
description: Operations skill for the MultiView-PV photovoltaic panel detection project. Use this skill for ANY of the following: (1) setting up or deploying the environment on a new machine — PyTorch GPU, GDAL, FlashAttention, SAM3 weights; (2) creating a new power station config (新电站); (3) running or debugging the detection pipeline — DOM mode, Oblique mode, iterative fusion, or any combination. Trigger whenever the user mentions deploying, installing, setting up the environment, creating a station, running the pipeline, or anything related to operating the MultiView-PV project.
---

# MultiView-PV Operations Skill

This skill covers three workflows. Read the relevant reference file for the full procedure.

| Task | Reference File | When to use |
|------|---------------|-------------|
| 🛠️ Environment setup | `references/setup.md` | New machine, reinstall, dependency issues |
| 🏗️ New station config | `references/new-station.md` | First time deploying a new power station |
| 🚀 Run pipeline | `references/run.md` | Any pipeline execution — single run, iterative, module-by-module |

---

## Project Layout (quick reference)

```
MultiView-PV/
├── cli/run_pipeline.py          ← single entry point
├── configs/
│   ├── _base.yaml               ← base defaults (do not edit)
│   └── {StationName}/
│       ├── base.yaml            ← station base config
│       ├── dom.yaml             ← DOM-only overrides
│       ├── oblique.yaml         ← oblique-only overrides
│       └── oblique_iter.yaml    ← iterative fusion overrides
├── weights/
│   └── sam3/                    ← SAM3 checkpoint + vocab (after download)
├── third_party/
│   └── sam3/                    ← SAM3 source (git submodule)
└── outputs/
    └── {StationName}/           ← all run outputs land here
```

---

## Key Config Fields (cheat sheet)

| Config key | Values | Effect |
|-----------|--------|--------|
| `projection.mode` | `dom` / `oblique` / `auto` | Which projection path |
| `pipeline.run_inference` | `true` / `false` | Skip inference, reuse cached per-image shapefiles |
| `postprocess.view_selection.enabled` | `true` / `false` | Filter to best N views |
| `postprocess.multiview.enabled` | `true` / `false` | Fuse detections across views |
| `postprocess.dom_merge.enabled` | `true` / `false` | Merge oblique result with existing DOM shapefile |
| `model.checkpoint_path` | path | SAM3 checkpoint `.pt` file |
| `model.bpe_path` | path | BPE vocab `.txt.gz` file |
| `data.image_glob` | glob | e.g. `/data/station/images/*.JPG` |
| `data.dom_shp` | path | existing DOM shapefile for fusion |
| `output.final_merged_shp` | path | where final result is written |

---

## CLI Override Syntax

Any config key can be overridden at runtime with dotlist notation:

```bash
python cli/run_pipeline.py \
  --config configs/StationName/base.yaml \
  inference.dataloader.batch_size=8 \
  output.final_merged_shp=outputs/StationName/exp1/final.shp
```

For full procedures, read the reference files above.
