from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class DistInfo:
    rank: int
    world_size: int
    local_rank: int


@dataclass
class GeoMeta:
    geotransform: Optional[List[float]]
    projection_wkt: Optional[str]
    width: int
    height: int


Feature = Dict[str, Any]
FeatureList = List[Feature]
