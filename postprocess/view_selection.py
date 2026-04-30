from __future__ import annotations
import os
import uuid
import random
from typing import Any, Dict, List, Set
from rtree import index
import networkx as nx
from tqdm import tqdm

from postprocess.nms import _bbox_iou, _geometry_iou

def select_views_from_features(
    features: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not features:
        return []

    view_num = int(cfg.get("view_num", 5))
    iou_threshold = float(cfg.get("iou_threshold", 0.1))
    use_geometry_iou = bool(cfg.get("use_geometry_iou", True))
    distance_threshold = cfg.get("distance_threshold")
    if distance_threshold is not None:
        distance_threshold = float(distance_threshold)
    random_seed = int(cfg.get("random_seed", 42))

    rng = random.Random(random_seed)

    tmp_dir = os.path.join(os.path.dirname(__file__), "..", "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    index_path = os.path.join(tmp_dir, f"rtree_{uuid.uuid4().hex}")

    idx = index.Index(index_path)
    try:
        for i, f in enumerate(tqdm(features, desc="Building spatial index", leave=False)):
            idx.insert(i, tuple(f["bbox"]))

        G = nx.Graph()
        G.add_nodes_from(range(len(features)))

        # Build edges for overlaps or distance-based links
        edge_count = 0
        pbar = tqdm(enumerate(features), total=len(features), desc="Finding overlaps", leave=False)
        for i, f in pbar:
            if distance_threshold is None:
                candidates = list(idx.intersection(tuple(f["bbox"])))
            else:
                cx = (float(f["bbox"][0]) + float(f["bbox"][2])) * 0.5
                cy = (float(f["bbox"][1]) + float(f["bbox"][3])) * 0.5
                search_box = (
                    cx - distance_threshold,
                    cy - distance_threshold,
                    cx + distance_threshold,
                    cy + distance_threshold,
                )
                candidates = list(idx.intersection(search_box))
            for j in candidates:
                if i >= j:
                    continue

                if distance_threshold is None:
                    if use_geometry_iou:
                        iou = _geometry_iou(f, features[j])
                    else:
                        iou = _bbox_iou(f["bbox"], features[j]["bbox"])

                    if iou > iou_threshold:
                        G.add_edge(i, j)
                        edge_count += 1
                else:
                    cx_j = (float(features[j]["bbox"][0]) + float(features[j]["bbox"][2])) * 0.5
                    cy_j = (float(features[j]["bbox"][1]) + float(features[j]["bbox"][3])) * 0.5
                    dx = cx - cx_j
                    dy = cy - cy_j
                    if (dx * dx + dy * dy) <= (distance_threshold * distance_threshold):
                        G.add_edge(i, j)
                        edge_count += 1
            if i % 100 == 0:
                pbar.set_postfix(edges=edge_count)
    finally:
        idx.close()
        for ext in [".dat", ".idx"]:
            if os.path.exists(index_path + ext):
                try:
                    os.remove(index_path + ext)
                except OSError:
                    pass

    # Find connected components
    clusters = list(nx.connected_components(G))

    # Randomly select features from each cluster
    selected_features: List[Dict[str, Any]] = []
    for cluster in clusters:
        cluster_list = list(cluster)
        if len(cluster_list) <= 2:
            continue
        k = min(len(cluster_list), view_num)
        selected_indices = rng.sample(cluster_list, k=k)
        for idx_sel in selected_indices:
            selected_features.append(features[idx_sel])

    return selected_features
