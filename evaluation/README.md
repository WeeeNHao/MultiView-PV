# Vector Evaluation (矢量结果评估)

Instance-level metrics between two polygon shapefiles (prediction vs ground truth),
matching the MultiView-PV feature/IO conventions.

## Metrics

| Metric | Meaning |
| --- | --- |
| **IoU** | `mean_matched_IoU` (宏平均：匹配对 IoU 的均值) and `micro_IoU` (微平均：Σ交集/Σ并集) |
| **F1** | per-IoU-threshold precision / recall / F1 (greedy COCO matching) |
| **mAP** | COCO-style mean AP over IoU sweep `0.50:0.05:0.95`, plus `AP50` / `AP75` |
| **MSE** | 面积误差：`area_MSE_per_instance` (匹配对 `area_pred - area_gt` 的均方误差), plus `total_area_error` |

Each shapefile `feature` is one PV panel (逐块多边形). Predictions are ranked by the
confidence field (`con_weight` by default).

## CLI

```bash
python cli/run_eval.py \
    --pred outputs/final_merged.shp \
    --gt   data/ground_truth.shp \
    --primary-iou 0.5 \
    --json reports/eval.json \
    --csv  reports/eval.csv
```

Options:

- `--score-field`     confidence field used to rank predictions (default `con_weight`)
- `--primary-iou`     IoU threshold for headline F1 / IoU / area metrics (default `0.5`)
- `--iou-thresholds`  comma-separated mAP sweep (default `0.50,...,0.95`)
- `--label`           restrict evaluation to one class label
- `--json` / `--csv`  optional report outputs

## Programmatic use

```python
from evaluation import evaluate_shapefiles

result = evaluate_shapefiles("pred.shp", "gt.shp", primary_iou=0.5)
print(result.summary_dict())
for tm in result.per_threshold:
    print(tm.iou_threshold, tm.f1, tm.average_precision)
```

Or with in-memory feature lists (e.g. inside the pipeline):

```python
from evaluation import evaluate_feature_lists
from io_flow.shp_io import read_features_from_shapefile

preds = read_features_from_shapefile("pred.shp")
gts   = read_features_from_shapefile("gt.shp")
result = evaluate_feature_lists(preds, gts, primary_iou=0.5)
```

## Notes

- Geometry math reuses `postprocess.nms` helpers; invalid self-intersecting
  rings are repaired via `Buffer(0)` before intersection.
- Empty GT + empty preds → AP = 1.0; empty preds with GT present → AP = 0.0.
- `area`/IoU are in map units of the shapefile CRS; if the layer is in a
  projected CRS (米), areas are in m² and `area_MSE` in m⁴.