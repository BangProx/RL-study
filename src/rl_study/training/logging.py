"""Small JSONL metric writer that rejects non-finite values."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MetricRecord:
    step_type: str
    step: int
    name: str
    value: float
    unit: str
    aggregation: str
    sample_count: int
    wall_time: float

    def __post_init__(self) -> None:
        if self.step < 0 or self.sample_count < 0 or self.wall_time < 0:
            raise ValueError("step, sample_count, and wall_time must be non-negative")
        if not math.isfinite(self.value):
            raise ValueError("metric value must be finite")


class JsonlMetricWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: MetricRecord) -> None:
        payload = json.dumps(asdict(record), allow_nan=False, sort_keys=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")
            handle.flush()
