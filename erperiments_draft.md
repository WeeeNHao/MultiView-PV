# 实验结果表格草案

> 状态：讨论稿。`—` 表示待填实验结果。

## 表 0：主实验方法配置

`Iterative bbox prompting`：基于上一轮结果自动生成 bbox prompt，再次调用 SAM~3。
`Geometry-constrained instance fusion`：使用物方实例、组件物理先验和几何一致性进行融合与反馈。

执行要点：先按本表锁定每条主实验管线；后续实现和统计不得在表外共享/新增模块。

| ID | Method                       | TDOM | Perspective MV | DSM projection              | Fusion unit             | Physical prior | Iterative bbox prompting | Geometry-constrained instance fusion |
| -- | ---------------------------- | :--: | :------------: | --------------------------- | ----------------------- | :------------: | :----------------------: | :----------------------------------: |
| M1 | SAM3-TDOM-Iter               |  ✓  |                |                             | TDOM mask refinement    |                |            ✓            |                                      |
| M2 | SAM3-MV-PixelVote-Iter       |      |       ✓       | Standard / global           | Pixel voting            |                |            ✓            |                                      |
| M3 | SAM3-TDOM+MV-LateFusion-Iter |  ✓  |       ✓       | Standard / global           | Pixel-level late fusion |                |            ✓            |                                      |
| P  | **Ours**               |  ✓  |       ✓       | Per-module local projection | Instance                |       ✓       |            ✓            |                  ✓                  |

## 表 1：主实验总体结果

执行要点：四条完整管线均运行至各自停止条件；所有可调参数按同一 protocol 固定。结果报三个站点宏平均，逐站点结果见表 2。

| Method                       | Task-specific training / FT | Input                                 | Fusion paradigm                                          | Obj. Prec. ↑ | Obj. Rec. ↑ |   Obj. F1 ↑ |      AP75 ↑ | Obj. mIoU ↑ | Centroid RMSE (m) ↓ | #SAM3 calls ↓ |   Runtime ↓ |
| ---------------------------- | :-------------------------: | ------------------------------------- | -------------------------------------------------------- | ------------: | -----------: | -----------: | -----------: | -----------: | -------------------: | -------------: | -----------: |
| SAM3-TDOM-Iter               |            None            | TDOM                                  | Iterative single-product mask refinement                 |            — |     <br />— |           — |           — |           — |                   — |             — |           — |
| SAM3-MV-PixelVote-Iter       |            None            | Perspective MV + DSM                  | Iterative pixel voting                                   |            — |     <br />— |           — |           — |           — |                   — |             — |           — |
| SAM3-TDOM+MV-LateFusion-Iter |            None            | TDOM + Perspective MV + DSM           | Iterative pixel-level late fusion                        |            — |           — |           — |           — |           — |                   — |             — |           — |
| **Ours**               |       **None**       | **TDOM + Perspective MV + DSM** | **Iterative geometry-constrained instance fusion** |  **—** | **—** | **—** | **—** | **—** |         **—** |   **—** | **—** |

## 表 2：主实验逐电站结果

执行要点：直接复用表 1 的配置与固定参数；按站点分别统计，禁止为某一站点单独改 prompt、阈值或权重。

| Method                       | BeiOu Obj. F1 ↑ | XinXie Obj. F1 ↑ | CangFang Obj. F1 ↑ | Macro Avg. ↑ | Worst-site ↑ | Across-site Std. ↓ |
| ---------------------------- | ---------------: | ----------------: | ------------------: | ------------: | ------------: | ------------------: |
| SAM3-TDOM-Iter               |               — |                — |                  — |            — |            — |                  — |
| SAM3-MV-PixelVote-Iter       |               — |                — |                  — |            — |            — |                  — |
| SAM3-TDOM+MV-LateFusion-Iter |               — |                — |                  — |            — |            — |                  — |
| **Ours**               |     **—** |      **—** |        **—** |  **—** |  **—** |        **—** |

## 消融实验

### 表 3：迭代反馈路径消融

执行要点：所有配置均使用 TDOM 与 Perspective MV 完成第 0 轮候选生成；仅改变后续迭代中的 re-prompt 路径。比较双源反馈相对单源反馈和不反馈的增益。

| Configuration                       | Re-prompt TDOM | Re-prompt Perspective MV | Obj. Prec. ↑ | Obj. Rec. ↑ |   Obj. F1 ↑ |      AP75 ↑ | Obj. mIoU ↑ |   Runtime ↓ |
| ----------------------------------- | :------------: | :----------------------: | ------------: | -----------: | -----------: | -----------: | -----------: | -----------: |
| No iterative feedback ($t=0$)     |                |                          |            — |           — |           — |           — |           — |           — |
| TDOM feedback only                  |       ✓       |                          |            — |           — |           — |           — |           — |           — |
| Perspective-MV feedback only        |                |            ✓            |            — |           — |           — |           — |           — |           — |
| **Full dual-source feedback** |       ✓       |            ✓            |  **—** | **—** | **—** | **—** | **—** | **—** |

### 表 4：模块几何先验消融

执行要点：保留数据源、局部投影、实例级融合和双源反馈。删除某个子评分后，对剩余评分权重重新归一化；无先验行改用 SAM~3 原始 confidence 排序和 NMS。

| Configuration                        | Area score | Rectangularity score | Aspect-ratio score | Obj. Prec. ↑ | Obj. Rec. ↑ |   Obj. F1 ↑ | Obj. mIoU ↑ | Over-seg. ↓ | Under-seg. ↓ |
| ------------------------------------ | :--------: | :------------------: | :----------------: | ------------: | -----------: | -----------: | -----------: | -----------: | ------------: |
| No module-geometry prior             |            |                      |                    |            — |           — |           — |           — |           — |            — |
| w/o area score                       |            |          ✓          |         ✓         |            — |           — |           — |           — |           — |            — |
| w/o rectangularity score             |     ✓     |                      |         ✓         |            — |           — |           — |           — |           — |            — |
| w/o aspect-ratio score               |     ✓     |          ✓          |                    |            — |           — |           — |           — |           — |            — |
| **Full module-geometry prior** |     ✓     |          ✓          |         ✓         |  **—** | **—** | **—** | **—** | **—** |  **—** |

### 表 5：输入数据源消融

执行要点：只删除对应输入流，其余可适用模块保持 Full 设置。`w/o perspective MV` 与主实验 TDOM baseline 不同：它仍保留本文的模块几何先验和反馈逻辑。

| Configuration                    | TDOM | Perspective MV | Obj. Prec. ↑ | Obj. Rec. ↑ |   Obj. F1 ↑ |      AP75 ↑ | Obj. mIoU ↑ | Centroid RMSE (m) ↓ |   Runtime ↓ |
| -------------------------------- | :--: | :------------: | ------------: | -----------: | -----------: | -----------: | -----------: | -------------------: | -----------: |
| Ours w/o perspective MV          |  ✓  |                |            — |           — |           — |           — |           — |                   — |           — |
| Ours w/o TDOM                    |      |       ✓       |            — |           — |           — |           — |           — |                   — |           — |
| **Full dual-source input** |  ✓  |       ✓       |  **—** | **—** | **—** | **—** | **—** |         **—** | **—** |

### 表 6：融合单位消融

执行要点：保留相同候选、局部投影、模块几何先验和双源反馈；仅把物方融合输出从栅格投票/连通域替换为候选实例关联/NMS。

| Fusion representation                    | Candidate filtering / score          | Object-space output                         | Obj. Prec. ↑ | Obj. Rec. ↑ |   Obj. F1 ↑ |      AP75 ↑ | Obj. mIoU ↑ | Over-seg. ↓ | Under-seg. ↓ |   Runtime ↓ |
| ---------------------------------------- | ------------------------------------ | ------------------------------------------- | ------------: | -----------: | -----------: | -----------: | -----------: | -----------: | ------------: | -----------: |
| Pixel-level score-weighted fusion        | Full module-geometry prior           | Raster vote + connected components          |            — |           — |           — |           — |           — |           — |            — |           — |
| **Instance-level geometry fusion** | **Full module-geometry prior** | **Candidate-level association / NMS** |  **—** | **—** | **—** | **—** | **—** | **—** |  **—** | **—** |

## 多视分析

### 表 7：镜头方向增量分析

执行要点：TDOM 始终保留；下视镜头固定先加入；四个倾斜镜头按固定随机种子生成的顺序逐一加入。每个镜头方向对应的全部有效影像共同参与推理，其他设置固定。

| Raw-view set                     | Nadir | # Oblique directions | Obj. Prec. ↑ | Obj. Rec. ↑ | Obj. F1 ↑ | AP75 ↑ | Obj. mIoU ↑ | #SAM3 calls | Runtime ↓ |
| -------------------------------- | :---: | -------------------: | ------------: | -----------: | ---------: | ------: | -----------: | ----------: | ---------: |
| TDOM only                        |      |                    0 |            — |           — |         — |      — |           — |          — |         — |
| TDOM + Nadir                     |  ✓  |                    0 |            — |           — |         — |      — |           — |          — |         — |
| TDOM + Nadir + O1                |  ✓  |                    1 |            — |           — |         — |      — |           — |          — |         — |
| TDOM + Nadir + O1 + O2           |  ✓  |                    2 |            — |           — |         — |      — |           — |          — |         — |
| TDOM + Nadir + O1 + O2 + O3      |  ✓  |                    3 |            — |           — |         — |      — |           — |          — |         — |
| TDOM + Nadir + O1 + O2 + O3 + O4 |  ✓  |                    4 |            — |           — |         — |      — |           — |          — |         — |

## 迭代分析

### 表 8：迭代收敛分析

执行要点：仅运行 Full method，记录每轮完整闭环后的输出。提前收敛的站点在后续轮次沿用最终输出，以计算跨站点宏平均；同时累计 SAM~3 调用量和耗时。

|                                            Iteration$t$ | #Pred. | TP | FP | FN | Obj. Prec. ↑ | Obj. Rec. ↑ | Obj. F1 ↑ | AP75 ↑ | Obj. mIoU ↑ | Cumulative#SAM3 calls | Cumulative runtime ↓ |
| --------------------------------------------------------: | -----: | -: | -: | -: | ------------: | -----------: | ---------: | ------: | -----------: | --------------------: | --------------------: |
| 0: text-prompt initialization + first object-space fusion |     — | — | — | — |            — |           — |         — |      — |           — |                    — |                    — |
|                                                         1 |     — | — | — | — |            — |           — |         — |      — |           — |                    — |                    — |
|                                                         2 |     — | — | — | — |            — |           — |         — |      — |           — |                    — |                    — |
|                                                         3 |     — | — | — | — |            — |           — |         — |      — |           — |                    — |                    — |
|                                                $\cdots$ |     — | — | — | — |            — |           — |         — |      — |           — |                    — |                    — |
|                                                 Converged |     — | — | — | — |            — |           — |         — |      — |           — |                    — |                    — |
