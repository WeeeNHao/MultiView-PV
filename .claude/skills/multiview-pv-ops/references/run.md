# Running the Pipeline

Entry point: `cli/run_pipeline.py`

---

## Module Control 开关总览

以下是所有可独立控制的模块开关，可在 config 中预设，也可在命令行覆盖：

| 开关 | 默认 | 说明 |
|-----|------|------|
| `pipeline.run_inference` | `true` | `false` = 跳过推理，直接用已有的 per-image shapefiles 做后处理 |
| `postprocess.per_image_nms.enabled` | `false` | 单图 NMS，投影后、视角选择前 |
| `postprocess.view_selection.enabled` | `true` | 多视角筛选（倾斜专用） |
| `postprocess.multiview.enabled` | `true` | 多视角融合 |
| `postprocess.dom_merge.enabled` | `false` | 与 DOM 结果融合（迭代专用） |
| `postprocess.prompt_export.enabled` | `false` | 导出 bbox prompt 供下一轮使用 |
| `output.trace_exports.enabled` | `false` | 输出中间结果 shapefile（调试用） |
| `inference.prompt.enabled` | `false` | 使用 bbox prompt 引导推理 |
| `distributed.enabled` | `true` | 多进程分布式（单 GPU 时会自动降级） |

**任何开关都可以在命令行临时覆盖，无需修改 config 文件。**

---

## 1. DOM 模式

### 基础运行

```bash
python cli/run_pipeline.py --config configs/{StationName}/dom.yaml
```

### 多 GPU

```bash
torchrun --nproc_per_node=2 cli/run_pipeline.py \
  --config configs/{StationName}/dom.yaml
```

### 只做后处理（跳过推理，reuse 已有 per-image shapefiles）

适用场景：调整后处理参数时不想重跑推理。

```bash
python cli/run_pipeline.py \
  --config configs/{StationName}/dom.yaml \
  pipeline.run_inference=false \
  output.per_image_shp_dir=/path/to/existing/per_image/
```

### 调整关键参数示例

```bash
python cli/run_pipeline.py \
  --config configs/{StationName}/dom.yaml \
  inference.dataloader.batch_size=8 \
  projection.score.score_threshold=0.6 \
  output.final_merged_shp=outputs/{StationName}/dom/exp2.shp
```

---

## 2. 倾斜摄影模式（Oblique）

### 基础运行

```bash
python cli/run_pipeline.py --config configs/{StationName}/oblique.yaml
```

### 多 GPU

```bash
torchrun --nproc_per_node=4 cli/run_pipeline.py \
  --config configs/{StationName}/oblique.yaml
```

### 调整视角数

```bash
python cli/run_pipeline.py \
  --config configs/{StationName}/oblique.yaml \
  postprocess.view_selection.view_num=8
```

### 关闭视角选择（收集全部视角后直接融合）

```bash
python cli/run_pipeline.py \
  --config configs/{StationName}/oblique.yaml \
  postprocess.view_selection.enabled=false
```

### 开启 per-image NMS（高重叠场景）

```bash
python cli/run_pipeline.py \
  --config configs/{StationName}/oblique.yaml \
  postprocess.per_image_nms.enabled=true \
  postprocess.per_image_nms.iou_threshold=0.25
```

### 开启中间结果输出（调试）

```bash
python cli/run_pipeline.py \
  --config configs/{StationName}/oblique.yaml \
  output.trace_exports.enabled=true \
  output.trace_exports.collected_shp=outputs/{StationName}/debug/collected.shp \
  output.trace_exports.selected_shp=outputs/{StationName}/debug/selected.shp
```

### 使用 prompt 引导推理（需要上一轮导出的 prompt 目录）

```bash
python cli/run_pipeline.py \
  --config configs/{StationName}/oblique.yaml \
  inference.prompt.enabled=true \
  inference.prompt.source=/path/to/prompts/ \
  inference.prompt.strict_window_prompt=true
```

---

## 3. DOM + 倾斜联合迭代融合

迭代流程分两阶段：
1. **Round 0**：独立跑 DOM 基线 + 倾斜基线，获得初始结果
2. **Round 1..N**：用上一轮融合结果作为 DOM prior，逐步提升召回率

### Round 0 — 基线

```bash
# DOM 基线
python cli/run_pipeline.py --config configs/{StationName}/dom.yaml

# 倾斜基线（同时导出 prompt 供下一轮使用）
python cli/run_pipeline.py \
  --config configs/{StationName}/oblique.yaml \
  postprocess.prompt_export.enabled=true \
  postprocess.prompt_export.output_dir=outputs/{StationName}/iter_0/prompts/
```

### Round 1 — 第一轮融合

```bash
python cli/run_pipeline.py \
  --config configs/{StationName}/oblique_iter.yaml \
  data.dom_shp=outputs/{StationName}/dom/dom_merged.shp \
  inference.prompt.enabled=true \
  inference.prompt.source=outputs/{StationName}/iter_0/prompts/ \
  inference.prompt.strict_window_prompt=true \
  postprocess.prompt_export.output_dir=outputs/{StationName}/iter_1/prompts/ \
  output.per_image_shp_dir=outputs/{StationName}/iter_1/per_image/ \
  output.final_merged_shp=outputs/{StationName}/iter_1/fused.shp
```

### Round N — 继续迭代

将 `data.dom_shp` 换成上一轮输出的 `fused.shp`，依此类推。

---

## 4. 自动化迭代脚本

对于需要多轮迭代的电站，可以参考 `scripts/run_xinxie_5view_3iter.sh` 创建专属脚本。

脚本支持的环境变量：

| 变量 | 默认值 | 说明 |
|-----|--------|------|
| `PYTHON_BIN` | `python` | Python 可执行文件路径 |
| `TORCHRUN_BIN` | `torchrun` | torchrun 路径 |
| `CONFIG_PATH` | station 默认 config | 基础 config 路径 |
| `OUTPUT_ROOT` | 电站输出目录 | 所有迭代结果的根目录 |
| `DATA_ROOT` | 电站数据目录 | 图像数据根目录 |
| `IMAGE_GLOB` | `images/*.JPG` | 图像匹配 glob |
| `ROUNDS` | `4` | 迭代轮数 |
| `VIEW_NUM` | `5` | 视角选择数量 |
| `NPROC_PER_NODE` | `1` | GPU 数量 |
| `DISTRIBUTED` | `1` | `0` = 关闭分布式 |
| `MASTER_PORT` | `29500` | DDP master port |

运行示例：

```bash
ROUNDS=3 VIEW_NUM=5 NPROC_PER_NODE=2 \
  DATA_ROOT=/data/station/ \
  OUTPUT_ROOT=outputs/NewStation/ \
  CONFIG_PATH=configs/NewStation/oblique.yaml \
  ./scripts/run_xinxie_5view_3iter.sh
```

---

## 5. 仅后处理模式

当推理已经完成，只想重新跑后处理（例如调整 NMS 阈值、重新融合）：

```bash
python cli/run_pipeline.py \
  --config configs/{StationName}/oblique.yaml \
  pipeline.run_inference=false \
  output.per_image_shp_dir=/path/to/existing/per_image/ \
  postprocess.multiview.iou_threshold=0.15 \
  output.final_merged_shp=outputs/{StationName}/reprocess/fused_v2.shp
```

---

## 6. 常见问题

### 显存不足（OOM）

```bash
# 减小 batch size
inference.dataloader.batch_size=4

# 减小 slice 重叠
inference.slicing.overlap_height_ratio=0.3
inference.slicing.overlap_width_ratio=0.3
```

### 检测率低 / 漏检严重

```bash
# 降低分数阈值
model.detection_threshold=0.4
projection.score.score_threshold=0.5
inference.early_filter.min_score=0.35
```

### 误检多 / 精度低

```bash
# 提高阈值
model.detection_threshold=0.6
projection.score.score_threshold=0.75
```

### 倾斜投影精度差

检查并调整：
- `projection.oblique.avg_alt`（地面高程是否准确）
- `projection.oblique.focal` / `cx` / `cy`（相机内参是否与实际匹配）
- `projection.oblique.min_control_points`（控制点不足时降低此值）

### 多 GPU 时 port 冲突

```bash
torchrun --nproc_per_node=2 --master_port=29501 cli/run_pipeline.py \
  --config configs/{StationName}/oblique.yaml \
  distributed.enabled=true
```
