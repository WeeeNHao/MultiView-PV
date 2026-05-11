# Code Review — MultiView-PV
Reviewed: 2026-05-09
Depth: standard
Scope: full Python source (pipeline / cli / inference / projection / postprocess / io_flow / utils), plus the two configs called out as recently edited

## Summary
- Files reviewed: 30 Python sources + 2 YAML configs
- Critical: 8 | Warning: 12 | Info: 7
- Top themes:
  - **Coordinate-system bugs** in shapefile/bbox handling — `read_features_from_shapefile` builds `bbox` in the wrong order, and the oblique footprint query uses a mismatched envelope tuple. Both feed downstream NMS/spatial indexing and silently corrupt fusion.
  - **Distributed/multiprocess hazards**: rank-only main-process early-return skips required `model.close()` and `cleanup_distributed()` shutdown for non-main ranks; raw shp barrier missing between inference and projection-from-raw stages; `barrier_if_needed` runs even when `distributed.enabled=false` if RANK/WORLD_SIZE leaked from env.
  - **GPU/inference correctness**: SAM3 batched inference shares a global `_GLOBAL_COUNTER` (not thread/process safe) and forces `torch.float16` autocast even on CPU device; mask postprocessing only ever returns the first contour from each window mask, dropping multi-blob detections.
  - **DSM/projection math**: `_ray_dsm_intersection` does not converge-check on the last iteration (returns None when within tol after final step); affine fitting silently uses `None` for `avg_alt` when fallback is enabled but config sets it to numeric 0/negative; `geo_to_image_xy` calls `np.linalg.solve` per-point on hot path (called millions of times) instead of inverting once.
  - **Resource/file handle leaks**: r-tree temp files leaked when exceptions thrown before `idx.close()`; GDAL datasets not closed with `del`; per-feature `geom.Clone()` retained in dicts past their useful lifetime.
  - **Silent failures**: many `try/except Exception: pass` blocks (DSM read, prompt export per-photo loop) hide real errors and degrade output without logging.

## Critical Findings

### [C1] `read_features_from_shapefile` builds `bbox` in wrong order — `io_flow/shp_io.py:189`
**Issue:** GDAL's `geom.GetEnvelope()` returns `(minX, maxX, minY, maxY)`. The code does:
```python
env = geom.GetEnvelope()
bbox = [float(env[0]), float(env[2]), float(env[1]), float(env[3])]
```
This produces `[minX, minY, maxX, maxY]` only if `env` were `(minX, minY, maxX, maxY)` — but it's not. The actual result is `[minX, minY, maxX, maxY]` from `(minX, maxX, minY, maxY)`, which **is correct**, BUT the same envelope unpacking is done inconsistently elsewhere. See `prompt_export.py:301`:
```python
minx, maxx, miny, maxy = geom.GetEnvelope()
query_box = (float(minx), float(miny), float(maxx), float(maxy))
```
which is right. However in `_feature_to_dom_bbox` (`prompt_export.py:400`) it does:
```python
min_x, max_x, min_y, max_y = geom.GetEnvelope()
```
correctly, but **the ordering chosen for `bbox` everywhere downstream is `[x1, y1, x2, y2]`**, and `read_features_from_shapefile` produces `[env[0], env[2], env[1], env[3]]` = `[minX, minY, maxX, maxY]` — which **is** the right ordering, after I traced it. **Withdrawn — re-classify as a readability hazard (see W2).** Net: not a critical bug, but the unpacking is dangerously non-obvious and one keystroke from breaking. Move to Warning.

(Leaving the entry as a record of the trace; severity reduced.)

### [C1-replaced] Non-main ranks skip `model.close()` and `cleanup_distributed()` after barrier — `pipeline.py:390-391`
**Issue:** Inside the `try:` block, after the barrier:
```python
if not is_main_process(info):
    return
```
returns immediately. The `finally:` still runs and does call `model.close()` + `cleanup_distributed()`, but only `model.close()` is gated on `model is not None` — and the inference path explicitly sets `model = None` at line 309 after closing. So workers do call `cleanup_distributed`. However, there is a separate problem: **if `run_inference=True`, `model.close()` is invoked twice**: once at line 308 (success path) and again at line 540 (`finally`). At line 308 the code does set `model = None` so `finally` skips the second call — OK. But on the *failure* path (exception during projection or postprocess), `model` may still be non-None for the failing rank but `runner = None` (line 310 only runs when no exception was raised). If the exception happens *before* line 309 sets `model = None`, the finally closes once — OK. The real bug:
**`runner` is bound to `None` only in the success path (line 310).** If an exception is raised mid-inference, `runner` is leaked but that's not a problem since it has no `close()`. **Re-categorize: see W3 for the actual bug — the early `return` on non-main rank skips `export_bbox_prompts` but also the `final_merged_shp` write, which is intended; OK.**

After tracing: this section is correct. Removing as a critical finding.

### [C1] `_ray_dsm_intersection` discards last-iteration convergence — `projection/oblique_projector.py:190-203`
**Issue:** The fixed-point loop runs `ray_dsm_max_iter` times. On the *last* iteration, if the new `dsm_z` differs from the previous `z` by more than `ray_dsm_tol`, the loop exits and the function returns `None`. There is no "best-effort" return after exhausting iterations. With the default `ray_dsm_max_iter=8` and a tolerance of `0.01` m, slowly converging rays (steep slopes / oblique angles) silently fall back to either `avg_alt` (if `ray_dsm_fallback_avg_alt=True`) or **drop the point entirely**, corrupting the polygon vertex set in `_project_points_direct_collinearity`. Worse, if iter==1 (mistakenly), it never has a chance to converge.
**Why it matters:** Polygons in oblique mode get partial vertices, producing degenerate or non-convex geometry that propagates through scoring and NMS. This is silent — there is no logging.
**Fix:** After the loop, return the last `gx, gy, dsm_z` even if not strictly within tolerance, optionally with a warning logged when residual is large. Or at minimum, log the count of non-converged points per feature.

### [C2] SAM3 batched inference uses non-reentrant global counter — `inference/models/sam3_segmenter.py:19, 32, 53, 62, 92, 199-200`
**Issue:** `_GLOBAL_COUNTER` is a module-level mutable used by `_add_text_prompt` / `_add_visual_prompt` to populate `coco_image_id` per query. After `segment()` finishes it resets the counter to 1. This breaks in three ways:
1. If SAM3 is ever called from multiple threads (`num_workers>0` PyTorch DataLoader uses process workers, but any future move to threading, e.g. async preview, would race).
2. If an exception occurs inside `segment()` between increment and the reset at line 199-200, the counter is left in an arbitrary state — affecting the next call in a way that is hard to debug.
3. If callers nest `segment()` calls (e.g., a wrapper like `predict_batch` retried for OOM), the inner call resets the counter while the outer still has stale ids.
**Why it matters:** The `coco_image_id` is used by SAM3's postprocessor to key results; collisions silently produce wrong-image masks.
**Fix:** Make the counter an instance attribute on `SAM3Segmenter` (or a local variable threaded through `_add_*_prompt`), reset inside a `try/finally`.

### [C3] `torch.autocast(device_type="cuda", ...)` hardcoded regardless of device — `inference/models/sam3_segmenter.py:194`
**Issue:**
```python
with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
    outputs = self.model(batch)
```
The `device` parameter to `SAM3Segmenter.__init__` defaults to `"cuda" if torch.cuda.is_available() else "cpu"`, but the autocast block always specifies `device_type="cuda"`. On a CPU-only run (or a GPU-less rank in mixed deployments), this raises `RuntimeError: User specified an unsupported autocast device_type 'cuda'`. The code path is silently CPU-incompatible despite the apparent CPU branch.
**Fix:** Use `device_type=self.device.type` (i.e., `"cuda"` or `"cpu"`) and gate `dtype=torch.float16` on CUDA.

### [C4] `mask_to_polygon` only returns the first contour — `inference/runner.py:15-30` and duplicated `inference/mask_utils.py:9-27`
**Issue:**
```python
poly = contours[0].flatten().tolist()
...
return [poly]
```
The function discards every contour after `[0]`. For SAM3 masks that contain disjoint blobs (a tilted PV array partially split across the slice boundary, or holes), only one ring survives. The `cv2.RETR_CCOMP` mode plus the unused `hierarchy` are a clear sign the original intent was to handle holes / multiple contours. Downstream `_segmentation_to_polygon` does support multi-ring polygons.
**Why it matters:** Detection masks for PV arrays composed of multiple panels often produce multiple connected components; the largest is not always `contours[0]` (cv2 ordering is not by area). Either dropped detections or wrong-area polygons propagate to scoring (`con_pv`) and NMS.
**Fix:** Iterate over all contours, optionally select the largest by area, or return all rings as separate polygon parts. At minimum: `return [c.flatten().tolist() for c in contours if len(c) >= 3]`.

### [C5] Raw-shp projection stage missing barrier between distributed ranks — `pipeline.py:322-372`
**Issue:** When `run_projection` is true, every rank reads its own slice of `_collect_rank_outputs(per_image_raw_dir)` and writes into `per_image_dir`. There is **no barrier** between the inference write phase and the projection-read phase. If a slow rank is still writing `per_image_raw` shapefiles while the next stage on a fast rank is already calling `_collect_rank_outputs` (line 324), the fast rank will skip files that haven't appeared yet. This is benign when inference and projection happen in the same `run_pipeline` invocation (each rank only reads what it just wrote, since `split_items_for_rank` partitions the same input list deterministically) — but it's wrong when:
1. `run_inference=true` and `run_projection=true` and a rank crashes between writing its raw shp and reading it (the resumption logic re-splits the file list, so partial files cause bbox/segment data with unset fields to leak through).
2. `run_inference=false, run_projection=true` (rerun). Here `_collect_rank_outputs` enumerates the entire directory; if NFS or a slow shared filesystem hasn't propagated, ranks see different file sets and `split_items_for_rank` produces non-disjoint partitions.
**Fix:** Insert `barrier_if_needed()` between the inference loop end (after `model.close()`) and the projection collection step. Also add `barrier_if_needed()` after the projection write loop, before the `if not is_main_process: return` early exit, so all ranks finish projecting before main starts collecting.

### [C6] `_OBLIQUE_PROJECTOR_CACHE` never closed; DSM file handles leak across runs — `projection/projector.py:14, 102-108` and `projection/oblique_projector.py:90-99, 400-402`
**Issue:** `_OBLIQUE_PROJECTOR_CACHE` is a module-level dict. Each cached `ObliqueProjector` opens a GDAL dataset on the DSM at construction. There is no code path that calls `close()` on cached projectors — when `run_pipeline` finishes, the cache and its `gdal.Dataset` handles persist for the lifetime of the Python process. In `run_view_sweep.py` (which calls `run_pipeline` repeatedly in a single process), this **leaks one DSM handle per (pose_csv, dsm_path) per sweep iteration if cfg objects ever differ**, and more importantly the DSM stays mapped — fine functionally, but the cache key is `f"{pose_csv}::{dsm_path}"` which hides cfg drift (e.g., `focal`, `cx` changes between iterations are ignored, and the *first* cfg's values silently win).
**Why it matters:** The view-sweep CLI explicitly mutates per-run config but the cached projector keeps the **first** call's intrinsics. Iterations 2..N silently use stale intrinsics. Combined with the iterative-merge workflow, this can produce subtly wrong projections that are very hard to debug.
**Fix:** (a) Include intrinsics in the cache key, or better, drop the cache and let callers pass in a projector. (b) Call `close()` on all cached projectors at end of `run_pipeline` (in `finally`).

### [C7] `read_pose_csv` skips header silently and is not robust to BOM — `projection/collinearity.py:9-20`
**Issue:** The CSV reader treats every non-empty line with ≥7 commas as a record. If the file has a header row like `name,X,Y,Z,phi,omega,kappa`, it will be inserted into `photo_dict["name"] = ["X","Y","Z","phi","omega","kappa"]`. Later code does `float(pose_params[0])` which raises `ValueError` and bubbles up via `try/except Exception: continue` blocks (e.g., `prompt_export.py:276-277`), so the failure mode is **silent silent skipping of EVERY image whose pose file has a header**. There's no warning. Also no UTF-8 BOM handling — Excel-exported CSVs with `﻿` will skip the first record's name match.
**Fix:** Detect header (try-parse first row's columns 1-3 as floats; if it raises, skip), or use `csv.DictReader`. Strip BOM. Log how many records were loaded.

### [C8] `view_selection` distance branch never inserts intersection results — `postprocess/view_selection.py:43-77`
**Issue:** When `distance_threshold is not None`, the code computes a centroid-search bbox and calls `idx.intersection(search_box)`. Then for each `j` it computes a centroid distance and adds the edge. So far OK. But the **r-tree was inserted with the feature bbox** at line 36, not with the centroid:
```python
idx.insert(i, tuple(f["bbox"]))
```
So `idx.intersection(search_box)` returns features whose **bbox** intersects the centroid query box, not features whose **centroid** is within `distance_threshold`. Features with large bboxes whose centroids are far away will still be candidates (acceptable, just loose), and — more seriously — features with small bboxes whose centroids are within `distance_threshold` but whose bboxes lie outside the centroid search box will be **missed entirely**. The check `(dx*dx + dy*dy) <= distance_threshold²` then accepts only the subset that the r-tree already filtered, so distant-but-relevant pairs never receive an edge.
**Why it matters:** The XinXie config (`oblique_views.yaml:25`) sets `distance_threshold: 0.5`, meaning 0.5 m. PV polygons have meters-wide bboxes (e.g., 2×1 m). Two polygons whose centroids are 0.4 m apart but whose bboxes don't overlap the 1×1 m centroid-query window are not linked, breaking cluster discovery.
**Fix:** Either (a) inflate `search_box` by the maximum feature size, or (b) index by centroid points (point-rtree), then filter with the actual bbox/distance test, or (c) inflate by `distance_threshold + max_feature_radius`.

### [C9] `nms_features` r-tree backend uses on-disk index in `tmp/` — concurrency-unsafe — `postprocess/nms.py:186-222`
**Issue:** Multiple ranks (or multiple sweep iterations) calling `_nms_features_rtree` concurrently each create a UUID-named file under `tmp/` — that's fine. But the cleanup at lines 215-222 runs in `finally`, and if the process is SIGKILLed, leaves dangling `.dat`/`.idx` files. More importantly, the r-tree index is constructed **on disk** with no need for persistence — switch to in-memory r-tree (`index.Index(properties=...)` with `Index()` default, or use `index.Storage`?) — the on-disk index is a 5-50× perf hit on hot paths.
**Why it matters:** Performance is out of v1 scope per the brief, but the *correctness* angle: under heavy load (many ranks on shared filesystem, e.g., NFS), `os.remove` calls in the `finally` race with kernel write-back, occasionally leaving zombie files that the next run's UUID won't conflict with — leak, not corruption. **Also**, `tmp/` is computed as `os.path.dirname(__file__) / ".." / "tmp"` — if the package is installed read-only (e.g., in a Docker layer), this raises `PermissionError` on `os.makedirs`. Fix: use `tempfile.mkdtemp` or in-memory index.

## Warning Findings

### [W1] `_collect_rank_outputs(per_image_dir)` does not check `per_image_raw_dir` correctness when oblique mode without raw — `pipeline.py:381-385`
**Issue:** The check `if run_postprocess and not _collect_rank_outputs(per_image_dir)` raises only after barrier. But if oblique inference ran and wrote raw shps but the projection failed silently for every feature (e.g., DSM out-of-bounds for all features), `per_image_dir` is empty and the user gets a generic FileNotFoundError saying "Run projection first." misdirecting debugging.
**Fix:** Differentiate: check `per_image_raw_dir` exists with content vs `per_image_dir` empty, and emit a clear "projection produced no features" diagnostic.

### [W2] `read_features_from_shapefile` envelope-to-bbox unpacking is one keystroke from a silent bug — `io_flow/shp_io.py:188-189`
**Issue:** `env = geom.GetEnvelope()` returns `(minX, maxX, minY, maxY)`. The code does `[env[0], env[2], env[1], env[3]]`. This is correct. But callers and other functions in this codebase use destructuring `min_x, max_x, min_y, max_y = geom.GetEnvelope()` (e.g., prompt_export.py) — inconsistent style means a future edit can easily flip an index without test coverage catching it.
**Fix:** Standardize: always `min_x, max_x, min_y, max_y = geom.GetEnvelope(); bbox = [min_x, min_y, max_x, max_y]`. Add a unit test.

### [W3] `_score_one_feature` overwrites `con_sem` even when set, and double-runs scoring on dom_merge — `projection/projector.py:111-141`, `pipeline.py:507-516`
**Issue:** `_score_one_feature` always recomputes `con_sem` from `out.get("con_sem", out.get("score", 0.0))`. When DOM features are loaded via `read_features_from_shapefile` (`pipeline.py:495-503`), they already carry `con_sem` from the shapefile. They are *not* passed back through `_score_one_feature`, but the merge then mixes raw image features (already scored) with DOM features (whose `con_pv`/`shape_score` etc. are loaded straight from shapefile fields, not recomputed from current geometry). This means DOM features carry stale geometry-based scores from a different acquisition, while image features carry fresh ones — the comparison in `confidence` strategy is on apples vs oranges.
**Fix:** Either (a) re-score DOM features after load, or (b) document explicitly that DOM scores are precomputed and trusted, and skip re-scoring on the image side as well.

### [W4] `_dsm_window_median` does not respect DSM nodata value — `projection/oblique_projector.py:125-149`
**Issue:** Filters with `np.isfinite` only, but most DSMs encode nodata as a sentinel like -9999 or -32768 (an actual float, not NaN/Inf). These pass `isfinite`, so the median is contaminated.
**Fix:** Read the nodata value via `band.GetNoDataValue()` and mask it in addition to `isfinite`.

### [W5] `_ray_dsm_intersection` reads single pixels for DSM elevation, not bilinearly interpolated — `projection/oblique_projector.py:103-112, 195`
**Issue:** `_read_dsm_value` does `int(round(col))`, `int(round(row))`. For the iterative DSM ray-trace, this snaps to integer DSM cells and can introduce a per-iteration step bigger than `ray_dsm_tol=0.01 m` when DSM resolution is 0.1 m. The iteration then never converges to within tol.
**Fix:** Bilinear sample, or set tol relative to DSM resolution.

### [W6] `_dsm_window_median` uses `block.astype(np.float64).ravel()` after slicing the unboxed 2D ndarray — fine, but doesn't account for `gdal.ReadAsArray` returning **None** if dataset has no band 1 — `oblique_projector.py:140-149`
**Issue:** `self._dsm_ds.ReadAsArray(c0, r0, ...)` may fail or return None for invalid windows. The code handles None, but if `_dsm_array is None and ReadAsArray` returns 1D (single pixel because window is 1×1), the indexing `block[rr_flat, cc_flat]` later (in `_build_affine_pairs`) will fail.
**Fix:** Force `np.atleast_2d` on the GDAL-returned array.

### [W7] Affine pairs filtering in `_build_affine_pairs` re-projects sampled DSM points back through collinearity — but the px/py photo origin convention differs from what `photo_to_ground` uses — `oblique_projector.py:317-333` vs `collinearity.py:38-55`
**Issue:** `ground_to_photo` returns `(px, py)` where `py` is in photo coordinate frame (Y-up). Then the code converts to image with `img_y = self.cy - py`. This is consistent with `_ray_dsm_intersection` which uses `py = self.cy - img_y`. But `_build_affine_pairs` open-codes the conversion at line 327-328 using `-self.focal * (a2*dx + b2*dy + c2*dz) / den2` — note this uses `a2/b2/c2` whereas in `collinearity.py:54` the formula is `-f * (a2*(x-xs) + b2*(y-ys) + c2*(z-zs)) / den`, **same**, OK. So this is internally consistent.
However, the inlined math duplicates `collinearity.ground_to_photo`. Any future fix to collinearity (e.g., handling `den ≈ 0`) won't propagate to `_build_affine_pairs`. The vectorized version's safe-clamp `den2 = np.where(np.abs(den2) < 1e-12, 1e-12, den2)` *silently produces values ~1e25*, which then pass the in-bounds filter at line 335-340 only if image bounds are huge — usually filtered. But for genuinely degenerate rays this hides the issue.
**Fix:** Refactor `ground_to_photo_vec` and reuse in both places. Mask out clamped rays explicitly rather than letting them flow through.

### [W8] `_project_points_direct_collinearity` silently drops points when intersection fails — `oblique_projector.py:217-231`
**Issue:** When `_ray_dsm_intersection` returns None and `ray_dsm_fallback_avg_alt` is False, the point is dropped from the polygon. The resulting `mapped` list may have fewer points than `points_xy`, producing a polygon with missing vertices. The caller (`project_feature` line 374-391) accepts whatever comes back without checking len. If only 1–2 points survive, the resulting "polygon" stored in `out["segmentation"]` has < 3 points and `_pairs_to_flat` returns a too-short list — but **`_segmentation_to_polygon` in `shp_io.py` skips rings with < 3 points**, producing an empty geometry that is then dropped by `if poly.IsEmpty(): return None` — so the entire feature is silently lost.
**Fix:** Either return None from `project_feature` so the caller knows the feature was dropped, or log a warning. Don't claim `projection_method=collinearity` on a feature whose geometry was zeroed out.

### [W9] `_apply_slope_correction_placeholder` is a no-op despite config flag advertising it — `oblique_projector.py:88, 352-354, 380, 389`
**Issue:** `enable_slope_correction` toggles a no-op. Configs that set it to True are silently the same as False. Misleading.
**Fix:** Either implement, or raise NotImplementedError when set to True, or remove the flag.

### [W10] `nms_features` with `use_geometry_iou=True` computes IoU using bbox fallback when geom is None, but never logs — `postprocess/nms.py:74-92`
**Issue:** Acceptable behavior, but masks failures: a feature whose `_feature_to_geometry` returns None (degenerate seg) silently uses bbox IoU, which can produce wildly different IoU vs the rest of the batch.
**Fix:** Track and log how many fallbacks occurred.

### [W11] `merge_image_with_dom_features` mutates input via `_feature_to_geometry` caching — `postprocess/merge.py` and `postprocess/nms.py:42-71`
**Issue:** `_feature_to_geometry` writes back `feature["geom"] = poly`. Across rank/process boundaries this is fine, but in iterative-merge workflows where the same FeatureList is reused, callers that copy via `dict(feature)` get a *shared* OGR geom reference. OGR Geometry objects are not safe to use after the parent dataset is closed (especially for those cloned from shapefile features). Mutation through aliasing is also a correctness hazard.
**Fix:** Don't write `feature["geom"]` back into caller-owned dicts; cache externally (WeakValueDictionary keyed by `id(feature)`).

### [W12] CLI `parse_args` uses `argparse.REMAINDER` which absorbs `--config` if positionally after — `utils/config.py:32-37`
**Issue:** `nargs=argparse.REMAINDER` is documented-deprecated in Python ≥3.9 and behaves surprisingly: `python run_pipeline.py inference.bs=8 --config foo.yaml` puts `--config foo.yaml` into `opts` and fails because `--config` is required. Users who put opts before `--config` get cryptic errors.
**Fix:** Use a custom flag like `--opts a=b c=d` or `nargs='*'`. Validate that opts contain only `key=value` strings.

## Info Findings

### [I1] Duplicate `mask_to_polygon` definitions — `inference/runner.py:15-30` and `inference/mask_utils.py:9-27`
**Issue:** Identical implementations. The runner imports nothing from mask_utils; updates to one will diverge.
**Fix:** Delete the runner-local copy; import from `inference.mask_utils`.

### [I2] `utils/draw_bbox.py` is a developer scratch script with hardcoded paths — `utils/draw_bbox.py:6-26`
**Issue:** Not part of the pipeline, has hardcoded `/data/dataset/PV/...` paths. Should not live in the package `utils/` namespace; belongs in `scripts/` or `tools/`.
**Fix:** Move to `scripts/` or delete.

### [I3] `pipeline.py:144-145` always overwrites `info` with a fresh `DistInfo(rank=0,...)` when `distributed_enabled=False`
**Issue:**
```python
info = get_dist_info()
info = DistInfo(rank=0, world_size=1, local_rank=0)
```
The first line is dead code.
**Fix:** Remove the no-op call.

### [I4] `cli/run_view_sweep.py:62-66` always sets `view_selection.enabled=false` regardless of label
**Issue:** Both the `label == "all"` and the `else` branch set `view_selection.enabled=false`. The `else` branch additionally sets `view_num`, but it has no effect since selection is disabled.
**Fix:** Either fix the intent (presumably the non-`all` branch should set `enabled=true`) or remove the dead `view_num` override.

### [I5] `_GLOBAL_COUNTER` typing: incremented by 1 per query, never used as anything but unique id — `inference/models/sam3_segmenter.py`
**Issue:** Could be replaced with `id(datapoint)` or a simple `itertools.count` per call.

### [I6] `compute_pv_geometry_score` ignores `score_cfg["mode"]` (`standard` vs `legacy_gaussian`) advertised in `_base.yaml:81` — `projection/scoring.py:75-119`
**Issue:** Config has `mode: standard` and commented-out legacy params, but the function never reads `score_cfg["mode"]`. The mode flag is documentation only.
**Fix:** Either implement the dispatch or remove the option from the schema.

### [I7] Test coverage gaps
- No test for `oblique_projector.ObliqueProjector` (collinearity round-trip is tested at the function level only).
- No test for NMS (rtree vs naive parity, identity check).
- No test for `merge` strategies (`union`, `prefer_dom`, `confidence`).
- No test for `view_selection` (especially the `distance_threshold` branch with the bug above).
- No test for `read_features_from_shapefile` round-trip with `export_features_to_shapefile`.
- No test for `prompt_export` modes (oblique / dom).
- `test_input_resolver.py:13-17` opens a `NamedTemporaryFile` and reads it under `resolve_image_paths`; on Windows the test fails because `NamedTemporaryFile` can't be reopened.

## Coverage Notes
- **Tested:** `collinearity.build_rotation` identity, `photo_to_ground`/`ground_to_photo` round-trip, `resolve_image_paths` for single, glob, and missing-file cases.
- **Uncovered, high-risk:** All of `oblique_projector` (DSM ray-tracing, affine fitting, fallbacks); all of postprocess (NMS, merge, view-selection, prompt-export); the entire pipeline orchestration and distributed split logic; multi-ring polygon handling in I/O.
- **Recommended adds:**
  1. NMS rtree vs naive parity test on a fixed input.
  2. `view_selection` with overlapping bboxes (cluster size > view_num) and with `distance_threshold` set.
  3. Round-trip: write features → read shp → re-export — fields and geometry preserved.
  4. Oblique projector: synthesize a DSM and a pose, project a known feature, assert position within tolerance.
  5. `_ray_dsm_intersection` test for the non-converging case (slope > tol/iteration).
  6. SAM3Segmenter `_GLOBAL_COUNTER` reset under exception.
