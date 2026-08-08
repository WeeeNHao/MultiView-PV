# 设计文档：去 DOM 的实验重组

> 编写日期：2026-08-08
> 对应改动：`erperiments_draft.md` 表 0–8 → 新表 0–8；`experiments_plan.md` 协议；`scripts/make_tables.py`；`postprocess/prompt_export.py`；`scripts/run_station_exp.sh`
> 结果来源：`/data/dataset/PV/ZS_PV/eval_exp2/all_stations_summary.csv` + `run_stats.csv`

---

## 1. 背景与动机

原设计把 TDOM（数字正射）作为方法的一等输入，它出现在草案表 0 的 **M1 / M3 / P 三行**。核对 `eval_exp2` 的实际结果后，TDOM 在本文方法里不承载任何增益。四条证据，从弱到强：

**证据一 —— 只对弱融合有用（原表 1）。** 把 TDOM 并入多视像素投票，t=2 的 F1 从 0.4940（M2 tuned）升到 0.5729（M3 tuned），+0.079。这是**像素级弱融合**下的收益。

**证据二 —— 反馈路径上无增益（原表 3）。** `Perspective-MV feedback only` = 0.9941 vs `Full dual-source feedback` = 0.9918，MV 单源反而略高。

**证据三 —— 输入源上无增益（原表 5）。** `Ours w/o TDOM` = 0.9941 ≥ `Full dual-source input` = 0.9918。

**证据四（最强，本次新核对）—— 跑到 t=2 后 TDOM 是净负收益。** 无 TDOM 的 `full` 变体三站 t=0/1/2 均已跑完，与双源 `ours` 对比（三站宏平均）：

| 指标 (t=2) | `full`（无 TDOM） | `ours`（双源） | 差值 |
| --- | ---: | ---: | ---: |
| RQ (=F1) | **0.9999** | 0.9969 | +0.0030 |
| PQ | **0.9753** | 0.9730 | +0.0023 |
| AJI | **0.9752** | 0.9702 | +0.0050 |
| AP95 | 0.9589 | **0.9675** | −0.0086 |
| SQ | 0.9754 | **0.9761** | −0.0007 |
| Area IoU | **0.9752** | 0.9702 | +0.0050 |

逐站看，差异全部来自 BeiOu：

| 站点 | 变体 | #Pred | Prec | Rec | F1 | FP | FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BeiOu | `full` | 1772 | 1.0000 | 1.0000 | **1.0000** | 0 | 0 |
| BeiOu | `ours` | 1804 | 0.9823 | 1.0000 | 0.9911 | **32** | 0 |
| XinXie | `full` | 2306 | 0.9996 | 1.0000 | 0.9998 | 1 | 0 |
| XinXie | `ours` | 2306 | 0.9996 | 1.0000 | 0.9998 | 1 | 0 |
| CangFang | `full` | 3633 | 1.0000 | 0.9997 | 0.9999 | 0 | 1 |
| CangFang | `ours` | 3633 | 1.0000 | 0.9997 | 0.9999 | 0 | 1 |

**把 TDOM 并进来，在最干净的那个站上凭空引入 32 个假阳。** 另外两站 TDOM 一个目标都没多贡献（预测数逐位相同）。

**唯一代价**：AP95 掉 0.0086，且几乎全部来自 XinXie（`full` 0.9421 vs `ours` 0.9636）；BeiOu 上 `full` 反而更高（0.9901 vs 0.9896）。按既定口径 AP95 是本数据上唯一有区分度的指标，这一项必须在正文里如实报告，不能只报 F1。

**结论**：TDOM 只救得了弱融合，救不了强融合。方法侧删除 TDOM，只保留一条 DOM baseline 作为**动机对照**（说明单张正射有天花板），不再与多视绑定。

---

## 2. 已锁定的决定

1. **Ours 重新定义为纯透视多视**，即现有的 `full` 变体（三站 `oblique_views.yaml` 本来就是 `dom_merge.enabled: false`）。三站 t=0/1/2 全部已跑完 —— **主表零新增跑批**。
2. **DOM baseline 保留 `dom`**（本文管线只喂 TDOM，含几何先验 + 迭代），t=2 F1 = 0.7365。**放弃 `m1`**（无几何先验，t=2 F1 = 0.0562，且随迭代持续变差 0.1108 → 0.0796 → 0.0562，作为 baseline 会被当成稻草人）。
3. **删除 M3**（`SAM3-TDOM+MV-LateFusion-Iter`）。
4. **删除原表 5（输入数据源消融）与原表 6（融合单位消融）。**
5. **原表 3 重设计**为反馈路由消融，四臂累积链（见 §4）。
6. **新增投影方式消融**（共线方程 / 仿射变换 / 斜面纠正），见 §5。
7. **迭代次数协议变更：除迭代收敛分析（表 7）外，所有表一律报 t=2。**（原协议是表 3/4/5/7 固定 t=1。）**t=1 不再作为任何正式表格的渲染层**，也不作为补跑未完成时的占位值 —— 缺失一律渲染 `--`。t=0/1 的历史行保留在 CSV 与逐站明细里，但不进任何正式表格。
8. 原表 3 的数据不浪费，改造成"为什么不用正射"的分析表（新表 8），**全部复用现成结果**。
9. **TDOM 与多视完全隔离**：跑批计划里不再有任何配置同时用到 TDOM 与透视多视。
   原表 8 的 `fb_tdom_only` 行（TDOM 只作反馈源灌进透视管线）因此删除 ——
   它只跑到 t=1，补完它是整个计划里唯一剩下的 TDOM×多视混合配置，
   等于花 5 小时 GPU 去重新混合本文正要分开的两条支路。
10. **`proj_affine` 不单独调参**：跑出什么记录什么。为它单独调阈值会破坏与另外两种投影方式的可比性。

### 2.1 关于删除表 6 的说明

草案里表 6 的设计意图是"保留相同候选、局部投影、模块几何先验和双源反馈，仅把物方融合输出从栅格投票/连通域替换为候选实例关联/NMS"。**但实际生成时用的是 `m3`/`m3_tuned`/`ours` 三行，与表 1 完全重复**（0.0000 / 0.5729 / 0.9969 逐位相同），草案里"保留相同先验只换融合单位"的单变量设计从未被实现。因此删除表 6 **不丢失任何已测量的信息**。

**必须诚实标注的一点**：新表 1 里 M2 vs Ours 是"相同输入源、不同融合范式"的**端到端对照**——融合单位、几何先验、逐模块局部投影三者是一起变的，**不是单变量消融**，表注里要写清楚。真正的单变量证据在表 3（反馈路由）、表 4（几何先验）、表 5（投影方式）。

---

## 3. 新表结构

**协议：表 1–6、表 8 一律报 t=2；表 7 是迭代收敛分析，报 t=0/1/2 三行。**

| 新编号 | 内容 | 变体来源 | 需要新跑 |
| --- | --- | --- | --- |
| 表 0 | 主实验方法配置 | — | 否（改表定义） |
| 表 1 | 主实验总体结果（三站宏平均） | `dom` / `m2`(+`m2_tuned`) / `full` | 否 |
| 表 2 | 主实验逐电站结果 | 同上 | 否 |
| 表 3 | **反馈路由消融**（重设计） | `full` t0 / `fb_selfimg` / `fb_srcview` / `full` | **是，4 轮** |
| 表 4 | 模块几何先验累积消融 | `abl_*` ×6 + `full` | **是，6 轮**（t=1 → t=2） |
| 表 5 | **投影方式消融**（新增） | `proj_collin` / `proj_affine` / `full` | **是，4 轮 + 2 次 CPU 投影** |
| 表 6 | 镜头方向增量分析（原表 7） | `dom` + `d1_nadir`..`d5_o4` | **是，5 轮**（子集，t=1 → t=2） |
| 表 7 | 迭代收敛分析（原表 8） | `full` t=0..2 | 否 |
| 表 8 | 分析：正射路径为何是死路 | `dom` / `ours` / `full` | 否 |

### 3.1 表 0 —— 方法配置

删掉 M3 行；Ours 行去掉 TDOM 勾选；M1 改为 `dom`（带本文先验的 TDOM 管线）。

| ID | Method | TDOM | Perspective MV | DSM projection | Fusion unit | Physical prior | Iterative bbox prompting | Geometry-constrained instance fusion |
| -- | --- | :--: | :--: | --- | --- | :--: | :--: | :--: |
| M1 | SAM3-TDOM-Iter (w/ prior) | ✓ | | DOM raster | TDOM mask refinement | ✓ | ✓ | |
| M2 | SAM3-MV-PixelVote-Iter | | ✓ | Standard / global | Pixel voting | | ✓ | |
| P | **Ours** | | ✓ | Per-module local projection | Instance | ✓ | ✓ | ✓ |

> M1 的命名要改：`SAM3-TDOM-Iter` 现在指的是带本文几何先验的 TDOM 管线（`dom` 变体），不是原来那个无先验的 `m1`。建议叫 `Ours-TDOM-only` 或 `SAM3-TDOM-Iter (w/ prior)`，正文说明它是"本文方法只喂单张正射"的上界。

### 3.2 表 6（原表 7）—— 行标签修正

现有行标签全部写着 `TDOM + Nadir + …`，**这是错的**。核对 `d5_o4` 与 `full` 在 t=0 的逐站结果完全一致（BeiOu 均为 1772 个预测 / Area IoU 0.9711 / F1 1.0000），说明 `dirs/` 下的跑批**从来就不含 TDOM**。新标签：

| 旧标签 | 新标签 | 变体 |
| --- | --- | --- |
| TDOM only | TDOM only (no perspective views) | `dom` |
| TDOM + Nadir | Nadir only | `d1_nadir` |
| TDOM + Nadir + O1 | Nadir + O1 | `d2_o1` |
| TDOM + Nadir + O1 + O2 | Nadir + O1 + O2 | `d3_o2` |
| TDOM + Nadir + O1 + O2 + O3 | Nadir + O1 + O2 + O3 | `d4_o3` |
| TDOM + Nadir + O1 + O2 + O3 + O4 | Nadir + O1 + O2 + O3 + O4 | `d5_o4` |

第一行保留为参考行（0 个透视视角的锚点），但要标注它走的是另一条管线（DOM 栅格），不是同一条曲线上的点。

### 3.3 表 8 —— 分析：正射路径为何是死路

按 TDOM 在管线中出现的位置排列，统一 t=2：

| TDOM 的角色 | 变体 | 数据状态 |
| --- | --- | --- |
| 唯一输入（本文管线只喂 TDOM） | `dom` | 现成（t=2 F1 0.7365） |
| 输入 + 反馈双源 | `ours` | 现成（t=2 F1 0.9969） |
| **完全不用（本文最终方法）** | `full` | 现成（t=2 F1 0.9999） |

这张表顺带修掉了原表 3 的一处口径不一致：`experiments_plan.md` 写"所有配置均使用 TDOM 与 Perspective MV 完成第 0 轮候选生成"，但 `Perspective-MV feedback only` 那行实际用的是 `full`，它从 t=0 起就没有 TDOM。新框架的自变量明确是"TDOM 在管线里的位置"，`full` = "哪里都不出现"，口径自洽。

---

## 4. 表 3 重设计：反馈路由消融

### 4.1 要支撑的论点

去掉 DOM 后，原表 3 的自变量（"反馈走哪条源"）只剩一个取值，表塌了。但它底下要支撑的论点没塌，只是需要换一种表达：

> 迭代反馈之所以有效，是因为它**绕了物方一圈**：某个组件在视角 A 被看清、融合进物方、再重投影成视角 B 的 bbox prompt，于是视角 B 里漏掉的那个组件被"喊"了出来。

原来这个论点用"跨数据源"表达，现在改用"跨视角"表达 —— 后者才是去掉 DOM 之后论文真正的卖点。

### 4.2 四臂机制矩阵

每一步只增加一个机制，构成严格的累积链：

| # | Arm | 变体 | 过物方？ | prompt 经先验筛选？ | 跨视角广播？ |
| --- | --- | --- | :--: | :--: | :--: |
| 1 | No feedback (t=0) | `full` t=0 | — | — | — |
| 2 | Image-space self re-prompt | `fb_selfimg` | ✗ | ✗ | ✗ |
| 3 | Object-space, source view only | `fb_srcview` | ✓ | ✓ | ✗ |
| 4 | **Object-space, all covering views (Ours)** | `full` | ✓ | ✓ | ✓ |

- **1 → 2**：加"迭代 re-prompt"本身。
- **2 → 3**：加"物方融合 + 几何先验筛选"（几何先验只有在物方才定义得出来，所以 arm 2 必然没有它 —— 这是设计的性质，不是混淆）。
- **3 → 4**：加"跨视角广播"。

| Configuration (t=2) | RQ (=F1) | SQ | PQ | AJI | AP95 | 数据状态 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| No feedback (t=0) | 0.9105 | 0.9659 | 0.8798 | 0.8312 | 0.7058 | 现成 |
| Image-space self re-prompt | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | `fb_selfimg` |
| Object-space, source view only | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | `fb_srcview` |
| **Object-space, all covering views (Ours)** | **0.9999** | 0.9754 | 0.9753 | 0.9752 | 0.9589 | 现成 |

（论文最终版另加 Runtime 列，由 `make_tables.py` 从 `run_stats.csv` 重算，此处不预填。）

### 4.3 为什么这个消融是可发表的（两种结果都能写）

- 若 arm 3 召回持平 t=0、SQ/AP95 上升 → "自视角反馈只改善掩膜质量，**检出增益全部来自跨视角广播**"。
- 若 arm 3 整体接近 arm 4 → "物方融合本身就够，广播是锦上添花"，需要相应调整正文论述。
- 若 arm 2 低于 t=0 → "朴素自迭代会退化"，这与 `m1` 已观察到的现象一致（无先验迭代 0.1108 → 0.0796 → 0.0562），互相印证。

### 4.4 无混淆的关键前提（已核实）

`configs/_base.yaml:47` 与 `scripts/run_station_exp.sh:346` 均设 `inference.prompt.strict_window_prompt: false`。因此**没收到 prompt 的影像照样全窗口跑推理，只是退回 text prompt**，行为与 t=0 一致。三臂之间唯一的差别就是 prompt 的来源与路由，不存在"某些影像被跳过"的混淆。

若将来有人把 `strict_window_prompt` 改成 `true`，这个消融立刻失效 —— 需要在表注和 `experiments_plan.md` 里写死这条前提。

### 4.5 两个新变体的定义

#### `fb_srcview` —— 物方反馈，只回灌源视角

**零代码改动。** 现成开关 `postprocess.prompt_export.include_intersections`：

- `true`（现状 = Ours）：每个物方融合实例重投影进**所有几何上覆盖它的影像**（`_ObliqueFootprintIndex` 查 footprint）。
- `false`：只回灌给 `src` 字段记录的那个**产生它的视角**（`postprocess/prompt_export.py:303-306`）。融合、几何先验、NMS、DSM 重投影全部保留，唯独砍掉广播。

三站配置均为 `postprocess.multiview.strategy: nms_keep_max`，因此 `src` 是**单个文件名**（`cluster_weighted` 才会 `;` 拼接；实测 0 条含 `;`），不触发 `_normalize_image_name` 对拼接串的失效，也不触发 shapefile 字符串字段 80 字符截断。

实测覆盖面（基于现有 `iter_0/full/final.shp`）：

| 电站 | 全部影像 | 收到 prompt 的影像 | 占比 | 中位 prompt 数/图 | 最大 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 001-BeiOu | 328 | 133 | 41% | 8 | 65 |
| 003-XinXie | 383 | 196 | 51% | 6 | 70 |
| 004-CangFang | 1454 | 379 | 26% | 3 | 31 |

即 **60–74% 的影像永远听不到别的视角的发现** —— 这正是被消融掉的机制。

#### `fb_selfimg` —— 图像空间自迭代

**需要新增一个 prompt 导出模式，约 40 行。**

定义：每张影像用**自己上一轮掩膜的像素空间 bbox** 再次 prompt，完全不过物方 —— 无投影、无跨视融合、无几何先验。

数据源：`${OUT}/iter_{t-1}/shared/infer/`（已核实为像素坐标，`srs=None`，带 `con` 字段，BeiOu 328 个 shp 对 328 张影像，1:1）。

**实现要点（已核实的坑）**：

- 文件命名是 `images_DJI_20251013154527_0001_V__r0.shp`，带 `images_` 前缀和 `__r<N>` rank 后缀。而 `inference/window_dataset.py:100` 找的是 `Path(image_path).stem + ".txt"`，即 `DJI_20251013154527_0001_V.txt`。**导出时必须剥掉前后缀**，否则 prompt 一条都匹配不上，而且不会报错 —— 会静默退化成 text-prompt-only，看起来"跑成功了"但等于什么都没做。
- 为公平对照，沿用与物方路径**相同**的 `min_size: 50` 与 `min_confidence: 0.5` 阈值，使两臂之间唯一的差别是路由而非筛选强度。
- 输出格式与现有一致：每行 `x1,y1,x2,y2`，全图像素坐标。下游 `_load_prompt_boxes` / `_window_prompt_boxes` 无需改动（它们本来就吃全图像素坐标再切窗）。
- `_load_prompt_boxes` 另有一道 `area < 500.0` 的硬过滤，两臂同样适用，无需特殊处理。
- t=2 那一轮的 prompt 来源是 **t=1 自己的** `infer/`，不是 t=0 的 —— 自迭代必须逐轮推进，不能固定在 t=0 的掩膜上。

**导出后必须核对计数**：`prompts/` 目录的 txt 数应等于影像数（328 / 383 / 1454），且总 box 数与 `infer/` 里过阈值的多边形数一致。这条核对是强制的 —— 本项目已有两次静默数据丢失的先例（`experiments_plan.md` §0.2b、§0.2d），都是靠"输入数 vs 输出数"才发现的。

---

## 5. 表 5 新增：投影方式消融

### 5.1 动机

逐模块局部平面拟合（`slope_correction`）是本文投影侧的主要贡献，且有独立的开发与修 bug 历史（commit `23ed553` 斜面纠正与仿射模块修复、`6bfabe7` / `42836bd` 射线-DSM 最近命中）。目前没有任何表格量化它相对更朴素的两种投影方式的增益，这是消融矩阵里的一个空洞。

### 5.2 三个方法（代码已就绪，零改动）

`projection/oblique_projector.py:65` 已支持 `auto / affine / collinearity / slope_correction` 四个取值，`project_feature()` 在 854–888 行按 `method` 分派。**只需改一个 config key `projection.oblique.method`。**

| Projection method | 机制 | 代码入口 | 变体 |
| --- | --- | --- | --- |
| Collinearity (direct) | 共线方程逐点直接投影，地面高程由射线-DSM 求交给出，不做局部平面拟合 | `_project_points_direct_collinearity` | `proj_collin` |
| Affine (control-point) | 由控制点对拟合仿射变换，整体映射像方多边形 | `_build_affine_pairs` + `compute_affine_transform` | `proj_affine` |
| **Slope correction (ours)** | 逐模块采样 → RANSAC 拟合局部平面 → 射线-平面求交 | `_project_feature_slope_correction` | `full` |

> `auto`（先试 affine，控制点不足退回 collinearity）是历史默认值，不进表 —— 它是两种方法的混合，作为消融行没有意义。

### 5.3 表 5 形态

投影方式直接决定物方定位精度，因此除核心指标外**必须报 Centroid RMSE**，它是最能区分三种方法的指标；同时报 Area IoU 与 AJI（面积贴合与过/欠分割）。

| Projection method | RQ (=F1) ↑ | SQ ↑ | PQ ↑ | AJI ↑ | AP95 ↑ | Area IoU ↑ | Centroid RMSE (m) ↓ | Runtime ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Collinearity (direct) | 待跑 | | | | | | | |
| Affine (control-point) | 待跑 | | | | | | | |
| **Slope correction (ours)** | **0.9999** | 0.9754 | 0.9753 | 0.9752 | 0.9589 | 0.9752 | 0.0144 | 现成 |

### 5.4 成本细节

换投影方法**只影响投影 + 打分阶段**，不影响像方推理。因此：

- **t=0 免费（纯 CPU）**：复用 `iter_0/shared/infer/` 的像素结果重跑 projection + postprocess，不需要 GPU 推理。
- **t=1 / t=2 需要 GPU 推理**：投影变了 → 融合实例变了 → prompt 变了 → 必须重推理。

每个方法 2 轮全量推理 + 1 次 CPU 投影，两个方法共 **4 轮 + 2 次 CPU 投影**。

### 5.5 已知风险：`proj_affine` 会成批丢要素

强制 `method: affine` 时，控制点少于 3 个的要素**无法确定仿射解**，`project_feature` 给它打上 `affine_failed` 并原样返回（仍是像方坐标），随后被 `project_and_score_features` 丢弃。这条路径已由 `tests/test_projection.py::test_forced_affine_without_enough_control_points_is_tagged_failed` 固化。

后果：`proj_affine` 那一行的召回损失里，有一部分**不是投影不准，而是要素根本没进入输出**。这是该方法的真实性质，必须报告，但必须和"投影不准导致的漏检"区分开。

因此跑 `proj_affine` 时**强制统计** `projection_method == "affine_failed"` 的要素数，并在表 5 的表注里给出丢弃比例。否则那一行会看起来莫名其妙地差，而真实原因（控制点不足）读者看不到。

另外 `analysis/` 里记录过仿射路径产出过大 footprint 的现象（与 M2/M3 像素投票崩塌相关）。若 `proj_affine` 跑出异常大的多边形导致 NMS 或评估阶段耗时失控，**先记录现象，不调参** —— 为了让它"好看"而单独调阈值会破坏三种投影方式的可比性。

---

## 6. 跑批清单与成本

### 6.1 缺口盘点

`eval_exp2` 现有 21 个变体。按"除收敛分析外一律 t=2"的新协议，缺口如下（`6` = t∈{0,1}，`9` = t∈{0,1,2}，`3` = 仅 t=1）：

| 变体 | 现有迭代 | 需要 | 新增全量轮次 | 备注 |
| --- | --- | --- | ---: | --- |
| `dom` `full` `m1` `m2*` `m3*` `ours` | 0,1,2 | ✓ | 0 | 主表零跑批 |
| `abl_*` ×6 | 0,1 | t=2 | 6 | 表 4 |
| `d1_nadir`..`d5_o4` ×5 | 0,1 | t=2 | 5（**子集**） | 表 6 |
| `fb_selfimg`（新） | — | 1,2 | 2 | 表 3，t=0 等同 `full` |
| `fb_srcview`（新） | — | 1,2 | 2 | 表 3，t=0 等同 `full` |
| `proj_collin`（新） | — | 0,1,2 | 2 + 1 CPU | 表 5 |
| `proj_affine`（新） | — | 0,1,2 | 2 + 1 CPU | 表 5 |
| **合计** | | | **14 全量 + 5 子集 + 2 CPU** | |

### 6.2 墙钟估算

按 `experiments_plan.md` §1.3 的实测单图均摊：

| 电站 | 影像 | rank | s/img | 14 全量轮 | 5 子集轮 | 小计 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 001-BeiOu | 328 | 2 | 13.7（实测） | 17.5 h | 3.4 h | **20.9 h** |
| 003-XinXie | 383 | 2 | ~14（外推） | 20.9 h | 4.6 h | **25.5 h** |
| 004-CangFang | 1454 | 4 | 12.0（实测） | 67.9 h | 15.1 h | **83.0 h** |

子集轮次的影像量是五个累积子集之和：BeiOu 903 / XinXie 1180 / CangFang 4519。

三站并行 → **关键路径 CangFang ≈ 3.5 天**。BeiOu / XinXie 约 1 天后腾出 GPU0 / GPU1，而 `abl_*` 与 `proj_*` 各轮彼此独立、可迁移过去分担，实际预期 **2.5–3 天**。

### 6.3 执行前必须做的一件事

当前工作树里 `projection/oblique_projector.py` 与 `projection/projector.py` **都有未提交的修改**。投影方式消融直接依赖这两个文件的行为，而这是一次 2.5–3 天的跑批 —— **必须先把代码提交并打 tag，从一个确定的状态启动**，否则结果不可复现，也无法判断表 5 的差异来自方法还是来自中途改动。

---

## 7. 代码改动

### 7.1 `postprocess/prompt_export.py` —— 新增 `self_image` 模式

- `_resolve_modes()` 增加 `self_image` 分支。
- 新增 `_export_self_image_prompts(cfg, prompt_cfg)`：不吃 `shp_path`（融合结果），改为遍历 `prompt_cfg.per_image_raw_shp_dir` 下的 per-image 像素坐标 shp。
- `maybe_export_bbox_prompts()` 里为该模式走不同的调用签名。
- 新增配置项 `postprocess.prompt_export.per_image_raw_shp_dir`。
- 单元测试：给定一个像素坐标 shp（含 `con` 低于阈值的要素、含小于 `min_size` 的要素、文件名带 `images_` 前缀和 `__r0` 后缀），断言输出 txt 文件名正确、被过滤的要素不出现、坐标逐位正确。

### 7.2 `scripts/make_tables.py`

- `table1` / `table2`：方法行改为 `[("SAM3-TDOM-Iter (w/ prior)", "dom"), *baseline_rows(r, "m2", "SAM3-MV-PixelVote-Iter"), ("**Ours**", "full")]`。
- `table3`：改为四行反馈路由消融，变体 `full`(t=0) / `fb_selfimg` / `fb_srcview` / `full`，后三者取 t=2。
- `table4`：base 变体不变，**迭代从 t=1 改为 t=2**。
- 新增 `table5`（投影方式消融）：`proj_collin` / `proj_affine` / `full`，t=2，含 Centroid RMSE 列。
- 删除原 `table5()`（输入源）与 `table6()`（融合单位）及其调用。
- 原 `table7` → 新表 6：行标签去掉 `TDOM +`，迭代 t=1 → t=2。
- 原 `table8` → 新表 7：`"ours"` 全部改为 `"full"`（本来就报 t=0..2，协议不变）。
- 新增新表 8（正射路径分析），变体 `dom` / `ours` / `full`，t=2。
- 更新模块 docstring 的变体 → 表格映射表。
- 顶部协议注释改写为："表 1–6、表 8 报 t=2；表 7 报 t=0/1/2。"

### 7.3 `scripts/run_station_exp.sh`

新增 stage：

| Stage | 内容 | 产出变体 |
| --- | --- | --- |
| `SFB1` | 导出 `include_intersections=false` 的 prompts → 跑 t=1、t=2 | `fb_srcview` |
| `SFB2` | 导出 `mode=self_image` 的 prompts → 跑 t=1、t=2（每轮重新从上一轮 `infer/` 导出） | `fb_selfimg` |
| `SPJ` | 对 `collinearity` / `affine` 各跑 t=0（复用 `infer/`，纯 CPU 投影）+ t=1、t=2 | `proj_collin`、`proj_affine` |

表 4 与表 6 的 t=2 **不新增 stage**：`_reprojection_variant`（S5 用）与 `S7` 的循环上界原先硬编码为 `t<=1`，改为 `ITER_MAX` 后重跑 `S5` / `S7` 即靠 `SKIP_EXISTING` 续上 t=2。平行的 `S5X`/`S4X` 会复制一份 prompt 链逻辑，迟早漂移。

> 顺带修掉一个只在 t≥2 才暴露的缺陷：`_reprojection_variant` 的 prompt 源原先硬编码为 `iter_0/prompts`。循环停在 t=1 时无害，跑到 t=2 就会每轮重放 t=0 的 prompt，呈现出一个从未发生过的“收敛”。现已按 `iter_$((t-1))/prompts` 逐轮链接。

全部遵守既有约定：产物落固定路径、`SKIP_EXISTING=1` 可断点续跑、`tee` 到 `${OUT}/_logs/{stage}.log`、通过 `scripts/tmux_launch.sh` 起 session。

### 7.4 `scripts/eval_0518_batch.py` / `collect_run_stats.py`

发现式扫描，新变体目录只要落在既有布局下就会被自动发现。**需要确认**新变体名不被 `make_tables.py:45` 的 `_DIRSET_RE = ^d\d_` 误判为方向集 —— `fb_` 与 `proj_` 前缀均安全（`fb_tdom_only` 已有先例）。

---

## 8. 文档改动

| 文件 | 改动 |
| --- | --- |
| `erperiments_draft.md` | 表 0 删 M3、Ours 去 TDOM 列；表 3 换成四臂反馈路由消融；新增表 5 投影方式消融；删原表 5、表 6；原表 7/8 重编号为 6/7；新增表 8 正射路径分析；各表"执行要点"重写；全表迭代协议改为 t=2 |
| `experiments_plan.md` | §0.2 表格状态更新；§0.2c 关于 `SP` 双源闭环的段落改为历史说明；§5 的 S5 消融描述重写；§6 协议决定改写：迭代深度统一 t=2（原"表 3/4/5/7 固定 t=1"作废），新增"`strict_window_prompt` 必须为 false"一条 |
| `experiments_results.md` | `python scripts/make_tables.py > experiments_results.md` 重新生成 |
| `experiments_results_{beiou,xinxie,cangfang}.md` | 同上，逐站重新生成 |

---

## 9. 已知问题与风险

1. **AP95 的代价必须如实报告。** 去掉 TDOM 后 AP95 从 0.9675 降到 0.9589（−0.0086），几乎全部来自 XinXie。既定口径是"F1 在 IoU 0.50–0.90 完全饱和，AP95 是唯一有区分度的指标"，所以不能只报 F1 就宣布去掉 TDOM 没有代价。建议正文写成：以 −0.009 的 AP95 换取 +0.003 F1 / +0.005 AJI、一整条数据支路的移除和更低的运行成本。
2. **M2 vs Ours 不是单变量对照。** 见 §2.1，表注必须写明。
3. **`fb_selfimg` 的文件名映射是静默失败点。** 见 §4.5，导出后强制核对计数。
4. **表 4 / 表 6 在补跑完成前是空的。** 协议锁死 t=2，**不允许**拿 t=1 的旧数字占位或当"临时结果"渲染 —— 缺什么就是 `--`。这是刻意接受的代价：宁可表格空着，也不让两套迭代深度的数字混进同一张表。
5. **`proj_affine` 可能产出异常 footprint。** 见 §5.5，不为它单独调参。
6. **多天跑批必须从提交后的代码状态启动。** 见 §6.4。
7. **原表 7 的标签错误已存在于当前 `experiments_results.md` 中。** 若这份结果已经对外发过，需要主动更正。
8. **`m1` / `m3` 变体的结果不删除，只是不进表。** 保留在逐站明细里，万一审稿人要求"无先验的朴素 baseline"或"朴素多源融合"可以随时补回。
9. **新表 3 / 表 5 在补跑完成前会有 `--` 行。** `make_tables.py` 的缺失渲染机制支持这种部分状态，可以先出表看其余部分。

---

## 10. 验收标准

- [ ] `python scripts/make_tables.py` 无报错产出新表 0–8。
- [ ] 表 1 的 Ours 行与 `full` t=2 逐位一致（F1 0.9999 / PQ 0.9753 / AJI 0.9752 / AP95 0.9589）。
- [ ] **回归校验（不涉及 t=1 渲染）**：表 1–6、表 8 的每个数值单元格，须与一段**独立校验脚本**直接从 `all_stations_summary.csv` 手算的三站宏平均逐位一致。校验脚本不得复用 `make_tables.py` 的聚合函数，否则校验不了聚合本身。
- [ ] **回归校验**：表 6 六行的变体映射须与原表 7 一一对应（`dom` / `d1_nadir` / `d2_o1` / `d3_o2` / `d4_o3` / `d5_o4`），行标签虽改但不得错位。
- [ ] `prompt_export.py` 的 `self_image` 模式有单元测试覆盖文件名剥离、置信度过滤、尺寸过滤三条路径。
- [ ] `fb_srcview` / `fb_selfimg` / `proj_collin` / `proj_affine` 四个变体三站跑完 t=2，`prompts/` 目录 txt 计数 = 影像数（328 / 383 / 1454）。
- [ ] `abl_*` ×6、`d1..d5` ×5 三站均补齐 t=2。
- [ ] 全仓搜 `fb_tdom_only`：除历史说明外不应出现在任何表定义或 stage 里。
- [ ] 表 5 三行的 Centroid RMSE 均有值（该列是这张表的关键区分指标，缺失即不合格）。
- [ ] `erperiments_draft.md` 中不再出现 M3、原表 5、原表 6，且 Ours 行不含 TDOM。
- [ ] 全仓 `grep -n "dual-source"` 的命中要么被删除，要么改写为历史/分析语境。
- [ ] 跑批启动前 `git status` 干净，且有对应 tag。
