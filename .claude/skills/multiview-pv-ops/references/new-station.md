# New Station Configuration

Every station gets its own directory under `configs/`. The structure is:

```
configs/{StationName}/
├── base.yaml           ← 电站基础信息（必须）
├── dom.yaml            ← DOM 模式专用（可选，继承 base）
├── oblique.yaml        ← 倾斜摄影模式专用（可选，继承 base）
└── oblique_iter.yaml   ← 迭代融合模式（可选，继承 base）
```

---

## Step 1 — Create base.yaml

`base.yaml` 定义电站固有的参数：相机内参、数据路径、PV 几何评分目标。
**不要**在这里写 `data.image_glob` 或 `output` 路径 —— 那些放到各模式 yaml 里。

```yaml
# configs/{StationName}/base.yaml
_base_:
  - ../_base.yaml

# ── 模型权重路径（全局共用，可覆盖 _base.yaml 中的默认值）──────────────────
model:
  checkpoint_path: weights/sam3_cache/facebook/sam3/sam3.pt
  bpe_path: third_party/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz
  device: cuda         # 无 GPU 时改为 cpu

# ── 投影参数（倾斜摄影必填，DOM 模式可省略）─────────────────────────────────
projection:
  oblique:
    pose_csv:  "/path/to/station/CAM/pose.csv"    # 相机位姿文件
    dsm_path:  "/path/to/station/dsm/DSM.tif"     # 数字地面模型
    avg_alt:   30.0       # 电站地面高程（米），影响投影精度
    preload_dsm: true     # 大 DSM 建议开启，首次加载后缓存

    # 相机内参（默认值适用于 5280x3956 标准镜头，需根据实际相机调整）
    focal: 3713.29
    cx: 2647.02
    cy: 1969.28
    image_width: 5280
    image_height: 3956

  # ── PV 几何评分目标（单块组件的实际尺寸，单位：米）──────────────────────
  score:
    mode: standard
    target_length: 2.2    # 组件长度
    target_width:  1.1    # 组件宽度
    # 评分权重（根据电站实测情况微调，以下为推荐起点）
    area_beta:  0.25
    ratio_beta: 0.2
    shape_beta: 0.1
    w_area: 10.0
    w_ratio: 5.0
    w_shape: 5.0
    w_sem:  0.2
    w_pv:   0.8
    score_threshold: 0.6  # 低于此分数的检测结果直接丢弃
```

**base.yaml 不需要填写** `data`、`output`、`postprocess`，这些在各模式 yaml 中按需设置。

---

## Step 2 — DOM 模式配置

适用于正射影像（DOM）输入，图像自带 geotransform，不需要 pose/DSM。

```yaml
# configs/{StationName}/dom.yaml
_base_:
  - ./base.yaml

projection:
  mode: dom             # 强制使用 DOM 投影，不走倾斜逻辑

data:
  data_root: "/path/to/station/"
  image_path:
    - "dom/DOM.tif"     # 相对于 data_root，或填绝对路径

postprocess:
  view_selection:
    enabled: false      # DOM 通常只有一张图，不需要 view selection
  multiview:
    enabled: true
    strategy: nms_keep_max
    score_field: con_weight
    iou_threshold: 0.2
  dom_merge:
    enabled: false      # 纯 DOM 不需要合并

output:
  per_image_shp_dir: outputs/{StationName}/dom/per_image/
  final_merged_shp:  outputs/{StationName}/dom/dom_merged.shp
```

---

## Step 3 — 倾斜摄影模式配置

适用于多视角倾斜影像（JPG/TIFF），需要 pose CSV + DSM。

```yaml
# configs/{StationName}/oblique.yaml
_base_:
  - ./base.yaml

projection:
  mode: oblique

data:
  data_root: "/path/to/station/"
  image_glob: "images/*.JPG"   # 或 image_list_file 指向文件列表

postprocess:
  per_image_nms:
    enabled: false             # 投影前不做 NMS（推荐关闭）

  view_selection:
    enabled: true
    view_num: 5                # 每个地物保留最佳 N 个视角
    iou_threshold: 0.1
    use_geometry_iou: true

  multiview:
    enabled: true
    strategy: nms_keep_max
    score_field: con_weight
    iou_threshold: 0.2

  dom_merge:
    enabled: false

  # 导出 bbox prompt（用于下一轮迭代，不需要时可删除此段）
  prompt_export:
    enabled: false
    mode: oblique
    output_dir: outputs/{StationName}/oblique/prompts/
    min_size: 50
    include_intersections: true
    use_spatial_index: true

output:
  per_image_shp_dir: outputs/{StationName}/oblique/per_image/
  final_merged_shp:  outputs/{StationName}/oblique/oblique_merged.shp
  trace_exports:
    enabled: false             # 调试时开启可查看中间结果
    collected_shp: outputs/{StationName}/oblique/trace_collected.shp
    selected_shp:  outputs/{StationName}/oblique/trace_selected.shp
```

---

## Step 4 — DOM + 倾斜联合迭代融合配置

迭代流程：先跑 DOM 基线，再用 DOM 结果辅助倾斜摄影的后处理融合，逐轮提升。

```yaml
# configs/{StationName}/oblique_iter.yaml
_base_:
  - ./base.yaml

projection:
  mode: oblique

data:
  data_root: "/path/to/station/"
  image_glob: "images/*.JPG"
  dom_shp: ""   # 每轮迭代时从 CLI 覆盖：data.dom_shp=/path/to/prev_round.shp

postprocess:
  view_selection:
    enabled: true
    view_num: 5
    iou_threshold: 0.1
    use_geometry_iou: true

  multiview:
    enabled: true
    strategy: nms_keep_max
    score_field: con_weight
    iou_threshold: 0.2

  dom_merge:
    enabled: true             # ← 开启 DOM 融合
    strategy: confidence
    score_field: con_weight
    iou_threshold: 0.2
    dom_score_field: con
    dom_label_field: label
    dom_src_field: src

  prompt_export:
    enabled: true
    mode: oblique
    output_dir: outputs/{StationName}/iter/prompts/
    min_size: 50
    include_intersections: true
    use_spatial_index: true

output:
  per_image_shp_dir: outputs/{StationName}/iter/per_image/
  final_merged_shp:  outputs/{StationName}/iter/iter_fused.shp
```

---

## Config Field Reference

### 必须填写（所有模式）

| 字段 | 说明 |
|-----|------|
| `model.checkpoint_path` | SAM3 权重文件路径 |
| `model.bpe_path` | BPE 词表文件路径 |
| `data.image_path` / `data.image_glob` / `data.image_list_file` | 图像输入（三选一） |
| `output.final_merged_shp` | 最终结果输出路径 |

### 倾斜模式额外必填

| 字段 | 说明 |
|-----|------|
| `projection.oblique.pose_csv` | 相机位姿 CSV 文件 |
| `projection.oblique.dsm_path` | 数字地面模型 GeoTIFF |
| `projection.oblique.avg_alt` | 电站地面平均高程（米） |

### 常见调整项

| 字段 | 默认值 | 说明 |
|-----|--------|------|
| `projection.score.target_length` | 2.2 | PV 组件长度（米） |
| `projection.score.target_width` | 1.1 | PV 组件宽度（米） |
| `projection.score.score_threshold` | 0.7 | 低分过滤阈值 |
| `postprocess.view_selection.view_num` | 5 | 每个地物保留的视角数 |
| `inference.dataloader.batch_size` | 16 | GPU 显存不足时调小 |
| `model.detection_threshold` | 0.5 | 分割置信度阈值 |

---

## 目录命名建议

```
configs/
├── 001-BeiOu/      ← 编号-名称
├── 002-XinXie/
└── 003-NewStation/
```

输出目录建议与 config 命名保持一致：
```
outputs/
├── 001-BeiOu/
├── 002-XinXie/
└── 003-NewStation/
```
