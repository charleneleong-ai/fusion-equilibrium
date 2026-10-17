"""The Final phase's rebuild budget (docs/final-budget.md): idle-A100 training hours of a blend, as (low, high)
ranges from measured run times. The rule caps training at 24 GPU-hours on one 80 GB A100; inference and CPU data
preparation are declared, not capped."""

from __future__ import annotations

from dataclasses import dataclass

from fusioneq.blend import BlendSpec

Hours = tuple[float, float]
LIMIT_HOURS = 24.0
MLP_HOURS: Hours = (16.5 / 60, 21 / 60)  # one 10 x 2048 MLP, 120 epochs, scoring included
TREE_HOURS: Hours = (1.0, 1.3)  # one XGBoost forest on all training frames, on the GPU
GRID_HOURS: dict[str, Hours] = {  # 48 channels x 15 epochs; 64 channels x 20 epochs (10.5 min per idle epoch)
    "grid_v2": (1.7, 2.8),
    "grid_v2_s1": (1.7, 2.8),
    "grid_v3": (3.6, 4.0),
    "grid_fno": (3.0, 3.3),  # FNO width 48 x 20 epochs: 3.0 h as run (12 Oct), up to 10 min per idle epoch
    "grid_fno_s1": (3.0, 3.3),
}


@dataclass(frozen=True)
class RebuildBudget:
    mlps: int
    forests: int
    grids: tuple[str, ...]

    @classmethod
    def of(cls, spec: BlendSpec) -> RebuildBudget:
        networks, forests, grids = spec.members()
        return cls(len(networks), len(forests), tuple(grids))

    def hours(self) -> Hours:
        if unknown := [g for g in self.grids if g not in GRID_HOURS]:
            raise KeyError(f"no measured training time for grid models {unknown}")
        lo, hi = (
            self.mlps * MLP_HOURS[i] + self.forests * TREE_HOURS[i] + sum(GRID_HOURS[g][i] for g in self.grids)
            for i in (0, 1)
        )
        return lo, hi
