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

**表 7（镜头方向增量）完全可做** —— 三站都有干净的下视簇 + 4 个倾斜方位簇。

> 注意：BeiOu / XinXie 的 `images/` 只是 `pose.csv` 的子集（328/1784、383/1338），
> 是已经裁过的 ROI。三站保持"用 `images/` 里的全部影像"这一口径即可，不要再改。

### 0.2 哪些表现在就能跑，哪些缺东西

| 表     | 内容                          | 状态        | 缺口                                                                                        |
| ------ | ----------------------------- | ----------- | ------------------------------------------------------------------------------------------- |
| 表 1/2 | 主实验，**Ours** 那一行 | ✅ 可跑     | —                                                                                          |
| 表 1/2 | M1`SAM3-TDOM-Iter`          | ✅ 已实现   | stage`SM1`（DOM 迭代，**无几何先验**）                                              |
| 表 1/2 | M2`SAM3-MV-PixelVote-Iter`  | ✅ 已实现   | `postprocess/pixel_fusion.py` + `multiview.st`，stage `SM2`                           |
| 表 1/2 | M3`TDOM+MV-LateFusion-Iter` | ✅ 已实现   | 同上 +`multiview.extra_features_shp` 把 TDOM 掩膜并入同一投票网格，stage `SM3`          |
| 表 3   | 迭代反馈路径                  | ✅ 可跑     | `prompt_export` 已支持 `mode: dom` 与 `oblique`，两条回灌路径都在                     |
| 表 4   | 模块几何先验                  | ✅ 配置即可 | 置`projection.score.w_area/w_ratio/w_shape=0`；"无先验"行需要走 SAM3 原始 confidence 排序 |
| 表 5   | 输入数据源                    | ✅ 可跑     | `w/o MV`=`SM1`/`SD`；`w/o TDOM`=`S3` 的 `full/`                                 |
| 表 6   | 融合单位                      | ✅ 已实现   | 复用`pixel_vote` 分支与 P 的实例融合对比                                                  |
| 表 7   | 镜头方向增量                  | ✅ 可跑     | 已建`scripts/split_directions.py`（按 pose 分方位 + 软链 per-image shp）                  |
| 表 8   | 迭代收敛                      | ✅ 可跑     | —                                                                                          |

**指标侧缺口**（`scripts/eval_0518_batch.py` 的 `SUMMARY_FIELDS` 里没有）：
`Centroid RMSE`、`#SAM3 calls`、`Runtime`、`Over-seg.`、`Under-seg.`。
前三个可从 `logs/pipeline_summary_*.json` 补；后两个要在 `evaluation/metrics.py` 里加
（一个 GT 对多个 pred = 过分割，多个 GT 对一个 pred = 欠分割）。

> 2026-07-21 更新：像素级融合已实现（`postprocess/pixel_fusion.py`，含单元测试），
> 表 1/2 四行与表 6 全部可跑。**主实验不再有代码阻塞。**

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

### 0.3 现有脚本的一个调度问题

`scripts/run_cangfang_3iter_1_15_views.sh` 在 `iter>=1` 时对 **view=1..15 每个都重跑一次完整推理**
（因为 prompt 随 view_num 变）。CangFang 就是 1454 × 15 × 3 次推理 —— 不可行。

**更根本的是：view_num 已经不是本研究的自变量了。**
草案表 7 的自变量是**镜头方向数**（下视 + 4 个倾斜方位的累积），不是 top-N 视角数。

因此（2026-07-21 决定）：

- `postprocess.view_selection` **全程关闭**，所选方向的全部有效影像都参与融合；
- 表 1/2 的 Ours 用**全部方向**（= 表 7 最后一行 = `images/` 里的全部影像）；
- 旧的 `views/view_{N}/` 目录结构废弃，改为 `full/`（主线）与 `dirs/{set}/`（表 7）。
  `scripts/eval_0518_batch.py` 三种布局都能发现，历史结果树仍可评估。

---

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

这份缓存被 S2（iter_0 后处理）、S4（表 7 方向集）、S5（表 4 的权重消融）全部复用。

### S2 — iter_0 后处理（纯后处理，CPU，`distributed.enabled=false`）

复用 S1 缓存跑一次全量后处理，产出 t=0 结果和喂给 S3 的 bbox prompts。
产物 `${OUT}/iter_0/full/final.shp` + `prompts/`。

> **`view_selection` 全程关闭（`enabled=false`）。**
> 本研究的多视角自变量是**镜头方向集合**（表 7），不是 top-N 视角数，
> 所选方向的全部有效影像都参与融合。管线在后处理只剩
> per-image NMS + multiview 融合。e

### S3 — 主迭代 t=1..2（全部镜头方向）→ 表 8、表 1/2 的 Ours

每轮：用上一轮的 `prompts/` 做 `inference.prompt.enabled=true`，全量重推理。
产物 `${OUT}/iter_{t}/full/final.shp`。

**迭代深度：最多 2 轮**（`ITER_MAX=2`，2026-07-21 决定）。
表 8 因此报 t=0 / 1 / 2 三行，末行即 "Converged"。省下 CangFang 约 4.8 h。

**收敛判据**（写进日志，用于表 8 的 "Converged" 行）：
`|ΔF1| < 0.005` **且** `|Δ#pred| / #pred < 0.01` 即判定收敛。
若 t=2 仍未满足该判据，表 8 末行须如实标注为"未收敛（截断于 t=2）"，
不得写成 Converged。

### S4 — 表 7 镜头方向增量（复用 S1 缓存，**零额外全量推理**）

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

> **协议决定**：表 7 固定在 `t=1` 而不是跑到收敛。跑到收敛需要
> 6 子集 × 4 轮全量推理，CangFang 单站就 ~35k 张次，不可行。
> `t=1` 下 6 个子集合计 ≈ 4.4k 张次，可接受，且能反映方法增益。

### S5 — 消融（表 3 / 4 / 5，**统一固定在 t=1**）

同样理由：消融表跑到收敛会让总量 ×3。固定 `t=1`，在表注里写明。

- **表 4（几何先验）**：改 `projection.score.w_*`，只影响**投影+打分**阶段。
  t=0 那一层可以直接复用 S1 的 `infer/`（像素结果）重跑 projection，省掉推理。
- **表 5（输入源）**：`w/o MV` = DOM-only + 本文先验；`w/o TDOM` = 已有的倾斜主线，直接复用 S3。
- **表 3（反馈路径）**：`no feedback` = S2 的 t=0 结果（免费）；
  `full` = S3 的 t=1（免费）；只有 `TDOM only` 和 `MV only` 需要各跑 1 轮。

### S6 — 评估

`scripts/eval_0518_batch.py` 是**发现式**的：扫 `iter_*/full/`、`iter_*/dirs/*/` 及历史的 `views/view_*/`，
缺的自动跳过。所以 **任何时候都能跑，跑完多少算多少**。

### S7 — 汇总出表

从 `all_stations_summary.csv` + `logs/pipeline_summary_*.json` 生成表 1–8 的 Markdown。

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

## 5. 建议的实施顺序

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

## 6. 已锁定的协议决定（2026-07-21）

1. **迭代深度**：主实验（表 1/2）与表 8 最多跑到 t=2（2026-07-21 改为 ITER_MAX=2）；
   **表 3 / 4 / 5 / 7 全部固定在 t=1**，论文表注需写明该协议。
2. **像素级融合（M2 / M3 / 表 6）**：不阻塞开跑。先执行 P0/P1 产出
   Ours 主线与表 3/4/5/7/8；像素级融合分支在 GPU 跑实验期间并行开发，
   完成后单独补跑 M2 / M3 / 表 6（这三者共用同一分支，可一次跑完）。
