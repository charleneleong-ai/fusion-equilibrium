"""Phase timers, plus the autoresearch GPU monitor flattened into loggable metrics and rightsizing hints."""

from __future__ import annotations

import dataclasses
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from autoresearch.gpu_monitor import GPUMonitor


@dataclass
class Phases:
    minutes: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def __call__(self, name: str):
        t0 = time.time()
        try:
            yield
        finally:
            self.minutes[name] = self.minutes.get(name, 0.0) + (time.time() - t0) / 60

    def as_metrics(self) -> dict[str, float]:
        return {f"phase_min/{k}": round(v, 3) for k, v in self.minutes.items()}


def gpu_metrics(monitor: GPUMonitor) -> tuple[dict[str, float], list[str]]:
    """Numeric fields of the monitor's summary under `gpu/`, and its human rightsizing hints."""
    summary = monitor.summary()
    numeric = {f"gpu/{k}": float(v) for k, v in dataclasses.asdict(summary).items() if isinstance(v, int | float)}
    numeric["gpu/peak_mem_pct"] = float(summary.peak_mem_pct)
    return numeric, list(summary.hints)
