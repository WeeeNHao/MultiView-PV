"""Non-Maximum Merging (SAHI, Akyon et al. 2022) -- the merging counterpart to NMS.

NMS answers "which of these duplicate detections do I keep"; NMM answers "which
of these partial detections are the same object". Sliced inference needs both:
one object seen whole in several windows wants NMS, one object cut by a window
boundary wants NMM.

It is included here as a *baseline* control, not as a fix. M1 fragments each PV
module at its central busbar into two adjacent 1.10x1.10 m halves whose pairwise
IoU is 0.000 (median over 5866 detections; only 17.4% of fragments reach even
0.1 with any neighbour). NMM decides what to merge from overlap -- IoU, or
intersection-over-smaller -- so disjoint halves are as invisible to it as they
are to NMS. Running it demonstrates that the standard sliced-inference toolkit
cannot recover these instances, which is the point: the merge criterion has to
come from module geometry, not from overlap.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from osgeo import ogr

from .nms import _bbox_iou, _feature_to_geometry


def _intersection_over_smaller(f1: Dict[str, Any], f2: Dict[str, Any]) -> float:
    """IoS, the criterion SAHI's NMM uses by default.

    Chosen over IoU because a fragment is small relative to the whole object: a
    half-module inside a full module scores IoU 0.5 but IoS 1.0, so IoS still
    fires when one detection is contained in another. It remains 0 for disjoint
    fragments, which is exactly the limitation being demonstrated.
    """
    g1 = _feature_to_geometry(f1)
    g2 = _feature_to_geometry(f2)
    if g1 is None or g2 is None:
        return _bbox_iou(f1["bbox"], f2["bbox"])
    try:
        inter = g1.Intersection(g2)
    except Exception:
        try:
            inter = g1.Buffer(0).Intersection(g2.Buffer(0))
        except Exception:
            return _bbox_iou(f1["bbox"], f2["bbox"])
    if inter is None or inter.IsEmpty():
        return 0.0
    ia = inter.Area()
    if ia <= 0:
        return 0.0
    smaller = min(g1.Area(), g2.Area())
    return float(ia / smaller) if smaller > 0 else 0.0


def _union_geometry(geoms: Sequence[ogr.Geometry]) -> ogr.Geometry | None:
    out = None
    for g in geoms:
        if g is None or g.IsEmpty():
            continue
        try:
            out = g.Clone() if out is None else out.Union(g)
        except Exception:
            try:
                out = g.Buffer(0).Clone() if out is None else out.Buffer(0).Union(g.Buffer(0))
            except Exception:
                continue
    return out


def nmm_features(
    features: List[Dict[str, Any]],
    score_field: str = "con_weight",
    match_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """Greedy NMM: highest-scoring detection absorbs everything it overlaps.

    Mirrors SAHI's GREEDYNMM. Returns merged features; the survivor keeps the
    cluster's best score and the union of its geometry.
    """
    if not features:
        return []

    order = sorted(features, key=lambda x: float(x.get(score_field, 0.0)), reverse=True)
    used = [False] * len(order)
    merged: List[Dict[str, Any]] = []

    for i, base in enumerate(order):
        if used[i]:
            continue
        used[i] = True
        cluster = [base]
        bbox = list(base["bbox"])
        for j in range(i + 1, len(order)):
            if used[j]:
                continue
            cand = order[j]
            cb = cand["bbox"]
            # Cheap bbox reject before the overlay call.
            if cb[0] > bbox[2] or bbox[0] > cb[2] or cb[1] > bbox[3] or bbox[1] > cb[3]:
                continue
            if _intersection_over_smaller(base, cand) > match_threshold:
                used[j] = True
                cluster.append(cand)

        if len(cluster) == 1:
            merged.append(base)
            continue

        geom = _union_geometry([_feature_to_geometry(c) for c in cluster])
        if geom is None or geom.IsEmpty():
            merged.append(base)
            continue
        out = dict(base)
        out["geom"] = geom
        env = geom.GetEnvelope()
        out["bbox"] = [float(env[0]), float(env[2]), float(env[1]), float(env[3])]
        out["segmentation"] = None  # regenerated on export from geom
        out["area"] = float(geom.Area())
        merged.append(out)

    return merged
