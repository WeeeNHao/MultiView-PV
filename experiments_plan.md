实验执行规划（对应 `erperiments_draft.md` 表 0–8）

> 本文只规划**执行流程**，不改变草案里的方法定义。
> 编写日期：2026-07-18。基于当前 `main` 分支代码与 `/data/dataset/PV` 实际数据核查得出。

---

## 0. 现状核查结论（先看这一节）

### 0.1 机器与数据

单机 6 卡，48 核 CPU：

| GPU  | 型号            | 显存       |
| ---- | --------------- | ---------- |
| 0    | RTX A6000       | 48 GB      |
| 1    | RTX 5880 Ada    | 48 GB      |
| 2–5 | RTX 2080 Ti ×4 | 11 GB each |

三电站实际影像与镜头方向分布（由 `CAM/pose.csv` 的 Phi/Omega/Kappa 聚类得到，已实测）：

| 电站         | 影像数 | 下视 (tilt<10°) | 倾斜 | 4 个倾斜方位 (Kappa 聚类) | GT               |
| ------------ | -----: | ---------------: | ---: | ------------------------- | ---------------- |
| 001-BeiOu    |    328 |               62 |  266 | 85 / 86 / 30 / 65         | `gt/pv.shp` ✓ |
| 003-XinXie   |    383 |               94 |  289 | 67 / 81 / 59 / 82         | `gt/pv.shp` ✓ |
| 004-CangFang |   1454 |              330 | 1124 | 279 / 252 / 324 / 269     | `gt/pv.shp` ✓ |

**表 6（镜头方向增量）完全可做** —— 三站都有干净的下视簇 + 4 个倾斜方位簇。

> 注意：BeiOu / XinXie 的 `images/` 只是 `pose.csv` 的子集（328/1784、383/1338），
> 是已经裁过的 ROI。三站保持"用 `images/` 里的全部影像"这一口径即可，不要再改。

### 0.2 哪些表现在就能跑，哪些缺东西

> **2026-08-08 起本节按重组后的表号（表 0–8）重写，见 §0.4。**
> 旧表号（原表 5 输入源、原表 6 融合单位）已删除。

| 表   | 内容                 | 状态              | 缺口                                                             |
| ---- | -------------------- | ----------------- | ---------------------------------------------------------------- |
| 表 1/2 | 主实验 **Ours**（`full`） | ✅ 已跑完 t=0/1/2 | —                                                               |
| 表 1/2 | M1`dom`（TDOM + 本文先验） | ✅ 已跑完 t=0/1/2 | stage`SD`                                                       |
| 表 1/2 | M2`m2` / `m2_tuned`  | ✅ 已跑完 t=0/1/2 | `postprocess/pixel_fusion.py`，stage `SM2`                       |
| 表 3 | 迭代反馈路由           | ⏳ 缺 2 个变体    | `fb_selfimg`（`mode: self_image`，已实现）、`fb_srcview`（`include_intersections: false`，零改动），各需 t=1+t=2 |
| 表 4 | 模块几何先验           | ⏳ 缺 t=2         | `abl_*` ×6 已有 t=0/1，需各补一轮                              |
| 表 5 | 投影方式               | ⏳ 缺 2 个变体    | 改`projection.oblique.method` 即可，代码已就绪；t=0 可复用 `infer/` 纯 CPU 重投影，t=1/t=2 需推理 |
| 表 6 | 镜头方向增量           | ⏳ 缺 t=2         | `d1..d5` 已有 t=0/1，需各补一轮（子集，非全量）                |
| 表 7 | 迭代收敛               | ✅ 已跑完         | —                                                               |
| 表 8 | 正射路径分析           | ✅ 已跑完         | 三行（`dom` / `ours` / `full`）全部现成，零新增跑批            |

**指标侧缺口**（`scripts/eval_0518_batch.py` 的 `SUMMARY_FIELDS` 里没有）：
`Centroid RMSE`、`#SAM3 calls`、`Runtime`、`Over-seg.`、`Under-seg.`。
前三个可从 `logs/pipeline_summary_*.json` 补；后两个要在 `evaluation/metrics.py` 里加
（一个 GT 对多个 pred = 过分割，多个 GT 对一个 pred = 欠分割）。

> 2026-07-21 更新：像素级融合已实现（`postprocess/pixel_fusion.py`，含单元测试），
> 表 1/2 四行与表 6 全部可跑。**主实验不再有代码阻塞。**
>
> 2026-08-08 更新：上述**指标侧缺口已全部补齐** —— `Centroid RMSE` / `Over-seg.` /
> `Under-seg.` 在 `evaluation/metrics.py`，`#SAM3 calls` / `Runtime` 由
> `scripts/collect_run_stats.py` 从 `logs/pipeline_summary_*.json` 汇总到
> `run_stats.csv`。另外补了 `PQ`/`SQ`/`RQ`/`AJI`/`AP95` 等实例级指标
> （F1 在 IoU 0.50–0.90 完全饱和，AP50/75/90 三者同值，没有区分度）。

### 0.2b 已修复：多卡下投影阶段静默丢图（`pipeline.py`）

**症状**：BeiOu 的 S1 缓存 `infer/` 有 328 个 shp，`proj/` 只有 327 个，且日志无任何报错。

**根因**：`pipeline.py` 的推理阶段和投影阶段之间**没有 barrier**。投影阶段用
`_collect_rank_outputs()` glob 整个 `infer/` 目录再 `split_items_for_rank()` 重新分配。
先跑完推理的 rank 会在其他 rank 还在写文件时就对目录拍快照 —— 那些还没落盘的影像
**不会进入任何一个 rank 的投影任务列表，也不会报错**。

BeiOu 丢的正是 rank 0 最后写的那张（09:55）。rank 数越多、rank 间快慢差越大，丢得越多：
XinXie 2 rank 侥幸没丢，CangFang 是 4 rank，风险最高。

**修复**：在推理与投影之间加 barrier（`pipeline.py`，`barrier_after_inference` 阶段）。

**存量缓存修复**：新增 `S1R` 阶段，只补投影 `proj/` 里缺失的那些，缺失为 0 时是空操作。

| 电站     | rank 数 | 修复前 |         修复后 |         丢失 |
| -------- | ------: | -----: | -------------: | -----------: |
| BeiOu    |       2 |    327 |  **328** |            1 |
| XinXie   |       2 |    383 |            383 |            0 |
| CangFang |       4 |   1431 | **1454** | **23** |

丢失量随 rank 数显著放大，和预测一致：CangFang 4 rank 丢了 1.6% 的影像，
且全程无任何报错。**这类静默数据丢失是最危险的**——不做 `infer/` vs `proj/`
的计数核对就永远发现不了，而下游所有表格都建立在这份缓存上。

> 教训：跨 rank 的目录 glob 必须有 barrier 保护。`S1R` 保留在流程里作为常驻校验。

### 0.2d 已修复：`per_image_nms` 关闭时后处理静默产出 0 个目标

**症状**：BeiOu 的 DOM 跑完，`proj/` 里有 602 个目标，最终却报
`RuntimeError: dom/iter_0/final.shp: No such file or directory`。

**根因**：`pipeline.py` 里读取 per-image 结果的整段循环被**嵌套在
`if per_image_nms_enabled:` 内部**。NMS 只是对这些目标的可选精化，
却成了读取它们的前提 —— 一旦 `per_image_nms.enabled=false`，
`all_features` 永远是空的，后处理等于什么都没做。

三站的 `oblique_views.yaml` 都写了 `per_image_nms.enabled: true`，所以倾斜路径
侥幸没暴露；而 `dom_only.yaml` 没有这一项，**整条 TDOM 路径必然产出 0 个目标**。

次级缺陷：结果为空时不写 final.shp，但紧接着仍去做 prompt 导出，
于是真正的失败被伪装成 OGR 的 "No such file or directory"。

**修复**（`pipeline.py`）：

- 把收集循环提到 `if` 外面，NMS 退回为循环内的可选步骤；
- 结果为 0 时直接抛出带 `per_image_shp_dir` 和阈值提示的 `RuntimeError`，不再让它伪装成文件缺失。

**验证**：BeiOu DOM 复跑后处理 → 读入 602 → NMS 后 371 → 写出 final.shp → 导出 371 个 prompt。

> 教训：这两个 bug（本节与 0.2b）都是**静默数据丢失**，都不报错、都只有靠
> "输入数 vs 输出数" 的核对才能发现。跑任何新路径都要先看计数是否合理。

### 0.2c 已补：TDOM 路径（原先整条缺失）

**问题**：表 0 里 TDOM 出现在 **M1、M3、P 三行**，只有 M2 是纯 MV。
但驱动脚本 S0–S4 全是倾斜路径，`grep dom_merge|dom_shp|dom_only` 零命中，
三站 `oblique_views.yaml` 都是 `dom_merge.enabled: false`。
**DOM 基线一次都没跑过**，`full/` 主线其实是纯倾斜多视 —— 它不是 P。

**已补两个 stage**：

| Stage  | 内容                               | 对应表格                                                 |
| ------ | ---------------------------------- | -------------------------------------------------------- |
| `SD` | DOM 基线 t=0 + 迭代 t=1..2，单进程 | 表 1/2 的**M1**，并提供 Ours 的 TDOM 先验          |
| `SP` | 真 P：双源回灌闭环                 | 表 1/2 的**P**、表 3 的 **Full dual-source** |

**`SP` 一轮 = 两次 pipeline 调用**：
(a) 用上轮 DOM prompt 重推理 DOM → (b) 用上轮倾斜 prompt 重推理透视影像，
融合后与 (a) 做 `dom_merge`。t=0 跳过 (a)，复用 SD 基线 —— 这样 Ours
**不会白拿 M1 的迭代增益**，起点是未精化的 t=0 DOM，对比才干净。

**为此改了 `postprocess/prompt_export.py`**：

- 新增 `mode: both` —— 一次融合结果同时导出倾斜侧（per-image 目录）和 DOM 侧（单 txt）。
  原先 `_resolve_mode` 只返回一种 mode，双源闭环根本没法表达。
- 新增 `prompt_export.dom_image` —— 关键：`_export_dom_prompts` 原本取
  `resolve_image_paths()[0]` 当 DOM，但在倾斜运行里那是一张 **JPG**，会静默出错。

**现有 `full/` 结果不作废**，重新归属为：
表 3「Perspective-MV feedback only」+ 表 5「Ours w/o TDOM」。

> **2026-08-08 追记：本小节整体成为历史记录。** 上面那句"重新归属"后来又被推翻了
> 一次 —— `full/` 现在就是 **Ours 本身**。`SP` 双源闭环产出的 `ours` 变体不再是
> 方法，只作为表 8 的一行证据保留。详见 §0.4。

### 0.3 现有脚本的一个调度问题

`scripts/run_cangfang_3iter_1_15_views.sh` 在 `iter>=1` 时对 **view=1..15 每个都重跑一次完整推理**
（因为 prompt 随 view_num 变）。CangFang 就是 1454 × 15 × 3 次推理 —— 不可行。

**更根本的是：view_num 已经不是本研究的自变量了。**
草案表 6 的自变量是**镜头方向数**（下视 + 4 个倾斜方位的累积），不是 top-N 视角数。

因此（2026-07-21 决定）：

- `postprocess.view_selection` **全程关闭**，所选方向的全部有效影像都参与融合；
- 表 1/2 的 Ours 用**全部方向**（= 表 6 最后一行 = `images/` 里的全部影像）；
- 旧的 `views/view_{N}/` 目录结构废弃，改为 `full/`（主线）与 `dirs/{set}/`（表 6）。
  `scripts/eval_0518_batch.py` 三种布局都能发现，历史结果树仍可评估。

---

### 0.4 已弃用 TDOM：Ours 重新定义为纯透视多视（2026-08-08）

跑到 t=2 之后，无 TDOM 的 `full` 在几乎所有指标上都优于双源的 `ours`：

| 指标 (t=2, 三站宏平均) | `full`（无 TDOM） | `ours`（双源） |
| ---------------------- | ------------------: | ---------------: |
| RQ (=F1)               |    **0.9999** |           0.9969 |
| PQ                     |    **0.9753** |           0.9730 |
| AJI                    |    **0.9752** |           0.9702 |
| AP95                   |              0.9589 | **0.9675** |

差异全部来自 BeiOu：并入 TDOM 引入 **32 个假阳**（1804 vs 1772 个预测，
Prec 1.0000 → 0.9823）；另外两站 TDOM 一个目标都没多贡献，预测数逐位相同。

**唯一代价**是 AP95 −0.0086，几乎全部来自 XinXie，必须在论文里如实报告 ——
本文口径是"AP95 是唯一有区分度的指标"，不能只报 F1。

因此：

1. **Ours = `full`**（纯透视多视）。主表零新增跑批。
2. **DOM baseline 只保留 `dom`** 一行（本文管线只喂 TDOM，含先验与迭代），
   放弃无先验的 `m1`（t=2 F1 = 0.0562 且随迭代变差，作 baseline 像稻草人）。
3. **删除 M3、原表 5（输入源）、原表 6（融合单位）。** 原表 6 实际生成时用的就是
   `m3`/`ours` 两行，与表 1 逐位重复，草案里"保留相同先验只换融合单位"的
   单变量设计从未实现，删除不丢失任何已测量的信息。
4. **原表 3 重设计**为四臂反馈路由消融（见草案表 3）。
5. **新增表 5 投影方式消融**（共线方程 / 仿射 / 斜面纠正）。
6. **新增表 8**"正射路径为何是死路"，复用现成跑批。

完整依据与设计见
`docs/superpowers/specs/2026-08-08-dom-free-experiment-restructure-design.md`。
基线代码状态打在 tag `exp2-baseline` 上（产出 `eval_exp2` 全部数字的那一版）。

## 1. GPU 分配

按你的要求，一站一组卡，三站并行：

| 电站         | 影像 | GPU                                          | 启动方式                        | MASTER_PORT |
| ------------ | ---: | -------------------------------------------- | ------------------------------- | ----------- |
| 001-BeiOu    |  328 | `CUDA_VISIBLE_DEVICES=0` (A6000)           | `torchrun --nproc_per_node=2` | 29500       |
| 003-XinXie   |  383 | `CUDA_VISIBLE_DEVICES=1` (5880 Ada)        | `torchrun --nproc_per_node=2` | 29501       |
| 004-CangFang | 1454 | `CUDA_VISIBLE_DEVICES=2,3,4,5` (4×2080Ti) | `torchrun --nproc_per_node=4` | 29502       |

### 1.1 S0 实测结果（2026-07-21，已验证）

规划阶段标为高风险的两点**均已排除**：

- **2080 Ti 跑得动 SAM3** —— 没有 FlashAttention-2 崩溃。
- **11 GB 显存够用** —— 默认 `batch_size=2` + slice 1024 未 OOM，无需降参。

实测单图耗时：

| 电站         | GPU      | 推理 s/img | 投影 s/img |
| ------------ | -------- | ---------: | ---------: |
| 001-BeiOu    | A6000    |       16.7 |     ~8–14 |
| 003-XinXie   | 5880 Ada |       16.7 |         — |
| 004-CangFang | 2080 Ti  |       35.7 |         — |

历史生产日志（XinXie iter_1，world_size=4，96 img/rank）给出稳态值：
推理 39.4 s/img、**投影 22.7 s/img**。2080 Ti ≈ A6000 的 47%，与预估相符。

> 冒烟测试里单图投影一度显示 683 s，那几乎全是 **DSM 预载**的一次性开销
> （`ObliqueProjector` 走 `projection/projector.py:14` 的模块级缓存，每进程只载一次）。
> 稳态投影约 23 s/img，不构成瓶颈。

### 1.2 单卡多进程（已验证）

投影是 CPU 密集且每 rank 单线程，单进程会让 48 核大部分闲置。
48 GB 的 A6000 / 5880 上放 **2 个 rank** 后推理仅从 16.7s 降到 15.2/17.1s，GPU 仍有余量。

为此改了 `inference/distributed.py`：

- `torch.cuda.set_device(local_rank % device_count)` —— 否则单卡 2 rank 时
  rank1 会 `set_device(1)` 报 invalid device ordinal。
- rank 数超过可见卡数时**自动切 gloo** —— NCCL 不支持同卡多 rank（会挂）。
  本管线唯一的集合通信是 `barrier()`，gloo 完全够用。

**RAM 约束**：每个 rank 各自预载一份 DSM（CangFang ~14 GB / BeiOu ~9 GB / XinXie ~5 GB）。
按 4+2+2 rank 计约 84 GB / 125 GB，可行但需监控。若吃紧，
`projection.oblique.preload_dsm=false` 是首选降压手段（实测速度损失很小）。

### 1.3 墙钟预估（按实测重算）

单轮 = 影像数 / rank 数 × (推理 + 投影)：

| 电站         | rank |   每轮 | S1 + 2 轮迭代 |
| ------------ | ---: | -----: | ------------: |
| 001-BeiOu    |    2 | ~1.8 h |          ~7 h |
| 003-XinXie   |    2 | ~2.1 h |          ~8 h |
| 004-CangFang |    4 | ~5.9 h |         ~24 h |

CangFang 仍是关键路径。CPU 侧 `num_workers=4` × 8 rank = 32 worker，48 核够用。

**S1 实际墙钟（2026-07-21，三站并行）** —— 都比预估快：

| 电站         |   预估 |                          实际 S1 | 单图均摊 |
| ------------ | -----: | -------------------------------: | -------: |
| 001-BeiOu    | ~1.8 h | **1.25 h**（09:07→10:22） |   13.7 s |
| 004-CangFang | ~5.9 h |       **4.8 h**（290 min） |   12.0 s |

三站并行时 CangFang 的单图均摊反而略优于 BeiOu，说明瓶颈确实在 CPU 侧投影
而非 GPU 算力——4 rank 拿到的 CPU 份额比 2 rank 多。

---

## 2. 统一的单站实验步骤

**三个电站执行完全相同的 S0–S7**，只有 `STATION` / `GPU` / `DATA_ROOT` 三个变量不同。
每一步都满足：产物落在固定路径、可断点续跑（`SKIP_EXISTING=1`）、有独立日志。

```
STATION=004-CangFang
OUT=/data/dataset/PV/ZS_PV/${STATION}-exp
DATA=/data/dataset/PV/${STATION}/
CFG=configs/CangFang/oblique_views.yaml
```

### S0 — 预检（~10 分钟，必须先做完三站再进 S1）

1. GT、pose.csv、DSM、DOM 存在性检查
2. **单图冒烟测试**：在目标 GPU 上跑 1 张图，确认模型能加载（FA2 问题）、显存够、投影出的多边形坐标落在合理范围
3. 实测单图耗时 → 反推 S1/S3 墙钟，决定迭代轮数上限
4. 固化 batch_size / overlap（尤其 CangFang 组）

**S0 不通过就不要往下走。**

### S1 — iter_0 共享推理缓存（最贵的一步，全站唯一一次全量推理）

```
pipeline.run_inference=true  run_projection=true  run_postprocess=false
→ ${OUT}/iter_0/shared/infer/   (像素坐标 raw shp)
→ ${OUT}/iter_0/shared/proj/    (物方坐标 shp)
```

这份缓存被 S2（iter_0 后处理）、S4（表 6 方向集）、S5（表 4 的权重消融）全部复用。

### S2 — iter_0 后处理（纯后处理，CPU，`distributed.enabled=false`）

复用 S1 缓存跑一次全量后处理，产出 t=0 结果和喂给 S3 的 bbox prompts。
产物 `${OUT}/iter_0/full/final.shp` + `prompts/`。

> **`view_selection` 全程关闭（`enabled=false`）。**
> 本研究的多视角自变量是**镜头方向集合**（表 6），不是 top-N 视角数，
> 所选方向的全部有效影像都参与融合。管线在后处理只剩
> per-image NMS + multiview 融合。e

### S3 — 主迭代 t=1..2（全部镜头方向）→ 表 7、表 1/2 的 Ours

每轮：用上一轮的 `prompts/` 做 `inference.prompt.enabled=true`，全量重推理。
产物 `${OUT}/iter_{t}/full/final.shp`。

**迭代深度：最多 2 轮**（`ITER_MAX=2`，2026-07-21 决定）。
表 7 因此报 t=0 / 1 / 2 三行，末行即 "Converged"。省下 CangFang 约 4.8 h。

**收敛判据**（写进日志，用于表 7 的 "Converged" 行）：
`|ΔF1| < 0.005` **且** `|Δ#pred| / #pred < 0.01` 即判定收敛。
若 t=2 仍未满足该判据，表 7 末行须如实标注为"未收敛（截断于 t=2）"，
不得写成 Converged。

### S4 — 表 6 镜头方向增量（复用 S1 缓存，**零额外全量推理**）

关键点：推理是逐影像的，所以 6 个 view-set 只是**从 S1 的 `proj/` 里挑不同的影像子集**做后处理。
由 `scripts/split_directions.py` 完成分组与软链（`--emit links`）。

1. 用 pose.csv 把影像分为 `nadir` / `O1..O4`（tilt<10° 为 nadir；其余按 Kappa 分 4 个方位，
   固定 seed=42 决定 O1..O4 顺序）
2. 6 个累积子集：`d0_tdom` / `d1_nadir` / `d2_o1` / `d3_o2` / `d4_o3` / `d5_o4`
3. 每个子集：软链 S1 的对应 per-image shp → 跑后处理 → 跑 **1 轮反馈 (t=1)**

实测分组（累积后最后一档 = 全部影像，已验证）：

| 电站         | nadir | +O1 | +O2 |  +O3 |  +O4 |
| ------------ | ----: | --: | --: | ---: | ---: |
| 001-BeiOu    |    62 |  92 | 178 |  243 |  328 |
| 003-XinXie   |    94 | 153 | 234 |  316 |  383 |
| 004-CangFang |   330 | 654 | 906 | 1175 | 1454 |

`d0_tdom` 不含任何透视影像（DOM-only 行），S4 会跳过它，由 DOM 管线单独产出。

> **协议决定（2026-07-21，已于 2026-08-08 作废）**：方向增量表原先固定在 `t=1`，
> 理由是跑到收敛要 6 子集 × 4 轮全量推理，CangFang 单站 ~35k 张次不可行。
> 新协议统一 t=2（见 §6），ITER_MAX=2 下 6 个子集再补一轮约 4.5k 张次，可接受。

### S5 — 消融（表 3 / 4 / 5，**统一 t=2**）

- **表 4（几何先验）**：改 `projection.score.w_*`，只影响**投影+打分**阶段。
  t=0 那一层可以直接复用 S1 的 `infer/`（像素结果）重跑 projection，省掉推理。
  `abl_*` ×6 已有 t=0/1，各补一轮到 t=2。
- **表 5（投影方式）**：改 `projection.oblique.method`，同样只影响投影+打分。
  `collinearity` / `affine` 的 t=0 复用 `infer/` 纯 CPU 重投影；t=1/t=2 因为
  融合实例变了、prompt 随之变，必须重推理。
- **表 3（反馈路由）**：`No feedback` = S2 的 t=0（免费）；
  `Ours` = S3 的 t=2（免费）；`fb_selfimg` 与 `fb_srcview` 各需 t=1+t=2 两轮。

### S5 之后新增的 stage（2026-08-08）

| Stage | 内容                                                                 | 产出变体                        |
| ----- | -------------------------------------------------------------------- | ------------------------------- |
| `SFB1` | 导出 `include_intersections=false` 的 prompts → 跑 t=1、t=2         | `fb_srcview`                  |
| `SFB2` | 导出 `mode=self_image` 的 prompts → 跑 t=1、t=2（每轮重新从上一轮 `infer/` 导出） | `fb_selfimg`     |
| `SPJ`  | `collinearity` / `affine` 各跑 t=0（复用 `infer/`，纯 CPU）+ t=1、t=2 | `proj_collin`、`proj_affine` |

表 4 与表 6 的 t=2 **不新增 stage**：`_reprojection_variant`（S5 用）与 `S7` 的循环
上界原先是硬编码的 `t<=1`，已改为 `ITER_MAX`，重跑 `S5` / `S7` 即靠 `SKIP_EXISTING`
续上 t=2。写平行的 `S5X`/`S4X` 会复制一份 prompt 链逻辑，迟早与本体漂移。

> 顺带修了一个只在 t≥2 才会暴露的缺陷：`_reprojection_variant` 的 prompt 源
> 原先硬编码为 `iter_0/prompts`。循环停在 t=1 时无害，但一旦跑到 t=2，
> 每一轮都会重放 t=0 的 prompt —— 结果会呈现出一个从未发生过的"收敛"。
> 现在按 `iter_$((t-1))/prompts` 逐轮链接。

> **没有 `fb_tdom_only` 的续跑 stage。** 补完它是整个计划里唯一剩下的
> TDOM×多视混合配置，与"把 TDOM 完全从多视中隔离出去"直接冲突。表 8 因此
> 收敛为三行，全部现成。**跑批计划里不再有任何配置同时用到 TDOM 与透视多视。**

**跑批量**：14 全量轮 + 5 子集轮 + 2 次纯 CPU 投影。按 §1.3 实测单图均摊：

| 电站         | 影像 | rank | s/img       | 14 全量轮 | 5 子集轮 |     小计 |
| ------------ | ---: | ---: | ----------- | --------: | -------: | -------: |
| 001-BeiOu    |  328 |    2 | 13.7（实测） |    17.5 h |    3.4 h | **20.9 h** |
| 003-XinXie   |  383 |    2 | ~14（外推）  |    20.9 h |    4.6 h | **25.5 h** |
| 004-CangFang | 1454 |    4 | 12.0（实测） |    67.9 h |   15.1 h | **83.0 h** |

三站并行 → 关键路径 CangFang ≈ 3.5 天。BeiOu/XinXie 约 1 天后腾出 GPU0/1，
`abl_*` 与 `proj_*` 各轮彼此独立、可迁移过去分担，实际预期 **2.5–3 天**。

**`fb_selfimg` 的强制核对**：`infer/` 的文件名是 `images_<IMG>__r0.shp`，
而 prompt 加载器找的是 `<IMG>.txt`。前后缀没剥对就一条都匹配不上，**且不报错** ——
会静默退化成 text-prompt-only，看起来跑成功了但等于什么都没做。
导出后必须核对 `prompts/` 的 txt 数 = 影像数（328 / 383 / 1454）；
`_export_self_image_prompts` 的返回值里带 `images` 与 `files` 两个计数就是为此。

**`proj_affine` 的强制统计**：强制 `method: affine` 时，控制点少于 3 个的要素
会被打上 `affine_failed` 并在下游丢弃。该行的召回损失里有一部分不是投影不准，
而是要素根本没进输出 —— 必须统计丢弃比例写进表注，且**不为它单独调参**。

### S6 — 评估

`scripts/eval_0518_batch.py` 是**发现式**的：扫 `iter_*/full/`、`iter_*/dirs/*/`、
`<method>/iter_*/` 及历史的 `views/view_*/`，缺的自动跳过。所以 **任何时候都能跑，
跑完多少算多少**；新变体（`fb_*` / `proj_*`）只要落在 `<method>/iter_{t}/final.shp`
就会被自动发现，不用改评估脚本。

> **必须显式给 `--pattern` 和 `--out-root`。** 默认值是 `--pattern '*-exp'` +
> `--out-root <zs-root>/eval_0518`，**匹配不到当前的 `-exp2` 树** —— 用默认参数跑
> 会去评旧的 `-exp` 实验并写进另一个目录，全程不报错，看起来成功但评的是另一批数据。
> 当前这一轮的正确调用是：
>
> ```bash
> python scripts/eval_0518_batch.py \
>     --pattern '*-exp2' --out-root /data/dataset/PV/ZS_PV/eval_exp2
> python scripts/collect_run_stats.py    # 刷新 run_stats.csv 的成本列
> ```

### S7 — 汇总出表（出完必须校验）

从 `all_stations_summary.csv` + `run_stats.csv` 生成表 1–8 的 Markdown：

```bash
python scripts/make_tables.py                        > experiments_results.md
python scripts/make_tables.py --stations 001-BeiOu   > experiments_results_beiou.md
python scripts/make_tables.py --stations 003-XinXie  > experiments_results_xinxie.md
python scripts/make_tables.py --stations 004-CangFang > experiments_results_cangfang.md

# 逐字符校验：从 CSV 独立重算每个单元格，不复用 make_tables 的聚合函数
python scripts/check_tables.py experiments_results.md
python scripts/check_tables.py experiments_results_beiou.md    --stations 001-BeiOu
python scripts/check_tables.py experiments_results_xinxie.md   --stations 003-XinXie
python scripts/check_tables.py experiments_results_cangfang.md --stations 004-CangFang
```

**校验不通过就不要用那份表。** 校验器会同时抓到数值错和行数/变体映射错
（负向测试验证过：改一个数值、删一行都会被报出来）。它唯一抓不到的是
"`make_tables` 与 `check_tables` 两处声明犯同一个错"，所以两边的行→变体映射
是刻意分别写的，不是互相 import。

---

## 3. 执行顺序（三站并行的甘特）

```
        BeiOu(GPU0)     XinXie(GPU1)    CangFang(GPU2-5)
S0      预检 ────────── 预检 ────────── 预检              ← 三站都过了才继续
S1      推理缓存        推理缓存        推理缓存 ──────────────┐
S2      iter_0后处理    iter_0后处理    iter_0后处理(CPU)     │ CangFang
S3      t=1,2,3         t=1,2,3         t=1,2,3 ──────────────┤ 是关键路径
S4      表7方向集(CPU)  表7方向集(CPU)  表7方向集(CPU)        │ 约 2.7× BeiOu
S5      表3/4/5         表3/4/5         表3/4/5 ──────────────┘
S6      ← 随时可跑，不必等全部完成 →
```

BeiOu / XinXie 做完后 GPU0/1 会空出来。届时可以把 CangFang 的
**S5 消融**（互相独立、且部分不需要推理）挪到 GPU0/1 上跑，压缩尾部时间。

---

## 4. "随时可查看结果" 的机制

四层，从粗到细：

1. **进度总览** —— `scripts/exp_status.sh`
   扫三站的产物树，打印一张表：每站每个 stage 完成/进行中/未开始、已产出多少 shp、
   当前 GPU 占用、各 stage 墙钟。一条命令看全局。
2. **后台执行一律走 tmux** —— `scripts/tmux_launch.sh <STATION> <STAGE...>`
   一站一个 session（`BeiOu` / `XinXie` / `CangFang`）。**不用 `nohup`，不用 PID 文件**：
   session 本身就是句柄，`tmux attach -t BeiOu` 随时进去看实时 tqdm 进度条，
   `Ctrl-b d` 脱离不影响运行。

   ```bash
   ./scripts/tmux_launch.sh 004-CangFang S1R S2 S4 S3   # 起或续
   tmux ls                                              # 看谁在跑
   tmux attach -t CangFang                              # 进去看
   ```

   **日志照常保留**：`run_station_exp.sh` 每个 stage 都 `tee` 到
   `${OUT}/_logs/{stage}.log`，脱离 tmux 后 `tail -f` 依然可用。
   pipeline 自带的 `PipelineRunLogger` 另外写 `logs/pipeline_summary_*.json`
   （含各阶段耗时、feature 数量变化），是 Runtime / #SAM3 calls 的数据来源。

   > 两个踩过的坑，已在 `tmux_launch.sh` 里处理：
   >
   > - session 结束 stage 后 `exec bash` 会**重新 source profile 退回 `(base)`**，
   >   直接往 pane 里发命令会 `ModuleNotFoundError: omegaconf`。必须先 `conda activate`。
   > - 旧版本遇到同名 session 直接报错退出，导致只能手工 send-keys。
   >   现在改为：pane 空闲则复用，pane 忙则拒绝并提示。
   >
3. **随时评估** —— `python scripts/eval_0518_batch.py --pattern '*-exp'`
   发现式扫描，不阻塞正在跑的实验，产出 `summary.csv`。
4. **随时中断/续跑** —— 所有 stage 遵守 `SKIP_EXISTING=1`：
   已存在最终 shp 就跳过。Ctrl-C 后重新执行同一条命令即可续上，不会重算。

---

## 5. 建议的实施顺序（2026-07-18 制定，**已全部完成**）

> 保留为记录。表号是**重组前**的旧编号（当时的表 5 = 输入源、表 6 = 融合单位，
> 两者已于 2026-08-08 删除）。新的待办见 §0.2 的状态表与 §2 的新增 stage。

| 优先级       | 事项                                                              | 说明                            |
| ------------ | ----------------------------------------------------------------- | ------------------------------- |
| **P0** | 补`configs/CangFang/dom_only.yaml`                              | 表 1/2 的 M1、表 5 都要         |
| **P0** | 统一的`scripts/run_station_exp.sh`（S0–S5）                    | 三站共用，只换环境变量          |
| **P0** | `scripts/exp_status.sh`                                         | 监控入口                        |
| **P0** | 先跑 S0 预检                                                      | 尤其是 2080Ti 的 FA2 / 显存问题 |
| **P1** | 按方位分影像的小工具（表 7）                                      | 逻辑已验证，几十行              |
| **P1** | `evaluation/metrics.py` 加 Over-seg / Under-seg / Centroid RMSE | 表 4/6 需要                     |
| **P2** | **像素级融合分支**（M2 / M3 / 表 6）                        | 唯一的实质性算法实现工作        |

P0 做完就能开跑并持续出结果；P2 可以在 GPU 跑主实验的同时并行开发。

---

## 6. 已锁定的协议决定（2026-08-08 修订）

1. **迭代深度统一 t=2。** 表 1–6、表 8 一律报 t=2；表 7 是迭代收敛分析，
   报 t=0/1/2。原"表 3/4/5/7 固定在 t=1"作废。
2. **t=1 不是渲染层。** 它是执行的必经步骤（每轮 prompt 来自上一轮，t=0 跳不到 t=2），
   但不作为任何正式表格的渲染层，也**不作为补跑未完成时的占位值** ——
   缺什么就渲染 `--`。宁可表格空着，也不让两套迭代深度的数字混进同一张表。
   `make_tables.py` 因此**移除了 per-table 的迭代旋钮**，从代码上堵死这条路。
3. **`strict_window_prompt` 必须为 `false`。** 表 3 的反馈路由消融依赖这一点：
   没收到 prompt 的影像照样全窗口跑推理、只是退回 text prompt，行为与 t=0 一致，
   三臂之间才只差 prompt 的来源与路由。改成 `true` 会让该消融立刻失效。
4. **`proj_affine` 不单独调参。** 跑出什么记录什么；为它单独调阈值会破坏
   三种投影方式的可比性。
5. **多天跑批必须从提交并打过 tag 的代码状态启动。** 表 5 的结论直接依赖
   `projection/oblique_projector.py` 的行为，从未提交状态启动会让结果不可复现，
   也分不清差异来自方法还是来自中途改动。
6. **出表后必须跑校验。** `python scripts/check_tables.py <md>` 会从 CSV 独立
   重算每个单元格逐字符比对，它**不复用** `make_tables.py` 的聚合函数（否则
   校验退化成自我认同）。逐站文件要带 `--stations` 一起校验。

### 历史决定（已被上面取代，保留备查）

- 2026-07-21：主实验与收敛表跑到 t=2（ITER_MAX=2），表 3/4/5/7 固定 t=1。
- 2026-07-21：像素级融合（M2/M3/原表 6）不阻塞开跑，在 GPU 跑实验期间并行开发。
  已完成；M3 与原表 6 随 2026-08-08 重组一并删除。
