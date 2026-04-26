from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PipelineRunLogger:
    def __init__(
        self,
        log_dir: str,
        rank: int,
        world_size: int,
        enabled: bool = True,
        run_id: Optional[str] = None,
    ) -> None:
        self.enabled = enabled
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.log_dir = log_dir
        self.stage_stats: Dict[str, Dict[str, float]] = {}
        self.count_changes: List[Dict[str, Any]] = []

        os.makedirs(self.log_dir, exist_ok=True)
        self.events_path = os.path.join(
            self.log_dir, f"pipeline_events_{self.run_id}_rank{self.rank}.jsonl"
        )

        logger_name = f"pipeline.rank{self.rank}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "[%(asctime)s] [%(levelname)s] [rank=%(rank)s] %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _base_payload(self) -> Dict[str, Any]:
        return {
            "ts": _utc_now_iso(),
            "run_id": self.run_id,
            "rank": self.rank,
            "world_size": self.world_size,
        }

    def event(self, name: str, **fields: Any) -> None:
        payload = self._base_payload()
        payload.update({"event": name})
        payload.update(fields)

        if self.enabled:
            with open(self.events_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def info(self, message: str, **fields: Any) -> None:
        extra = {"rank": self.rank}
        if fields:
            detail = " ".join(f"{k}={fields[k]}" for k in sorted(fields))
            self.logger.info(f"{message} {detail}", extra=extra)
        else:
            self.logger.info(message, extra=extra)

    @contextmanager
    def stage(self, name: str, **fields: Any) -> Iterator[None]:
        start = time.perf_counter()
        self.event("stage_start", stage=name, **fields)
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.event("stage_end", stage=name, elapsed_ms=elapsed_ms, **fields)

            stat = self.stage_stats.setdefault(
                name,
                {
                    "count": 0.0,
                    "total_ms": 0.0,
                    "max_ms": 0.0,
                    "min_ms": 0.0,
                },
            )
            stat["count"] += 1.0
            stat["total_ms"] += elapsed_ms
            stat["max_ms"] = max(stat["max_ms"], elapsed_ms)
            if stat["min_ms"] <= 0.0:
                stat["min_ms"] = elapsed_ms
            else:
                stat["min_ms"] = min(stat["min_ms"], elapsed_ms)

    def log_count_change(self, stage: str, count_in: int, count_out: int, **fields: Any) -> None:
        delta = int(count_out) - int(count_in)
        record = {
            "stage": stage,
            "count_in": int(count_in),
            "count_out": int(count_out),
            "delta": delta,
        }
        record.update(fields)
        self.count_changes.append(record)
        self.event("count_change", **record)

    def write_summary(self, summary_path: str, **extra: Any) -> str:
        os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)

        stage_summary: Dict[str, Dict[str, float]] = {}
        for stage_name, stat in self.stage_stats.items():
            count = max(stat["count"], 1.0)
            stage_summary[stage_name] = {
                "count": stat["count"],
                "total_ms": stat["total_ms"],
                "avg_ms": stat["total_ms"] / count,
                "min_ms": stat["min_ms"],
                "max_ms": stat["max_ms"],
            }

        payload: Dict[str, Any] = {
            "ts": _utc_now_iso(),
            "run_id": self.run_id,
            "rank": self.rank,
            "world_size": self.world_size,
            "events_path": self.events_path,
            "stages": stage_summary,
            "count_changes": self.count_changes,
        }
        payload.update(extra)

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=True)

        return summary_path
