"""Weights & Biases tracking behind one switch: WANDB_MODE=disabled (or no login) turns it into a no-op."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
# An empty `WANDB_API_KEY=` in .env would shadow the netrc login and break wandb.init: drop blank values.
for _k in [k for k, v in os.environ.items() if (k.startswith("WANDB_") or k.startswith("HF_")) and v == ""]:
    os.environ.pop(_k)


class Run:
    """Thin wrapper so callers never branch on whether wandb is available."""

    def __init__(self, name: str, config: dict[str, Any], job_type: str = "train") -> None:
        self.run = None
        if os.environ.get("WANDB_MODE", "").lower() == "disabled":
            return
        try:
            import wandb

            self.run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", "fusion-equilibrium"),
                entity=os.environ.get("WANDB_ENTITY") or None,
                name=name,
                job_type=job_type,
                config=config,
                reinit=True,
            )
        except Exception as exc:  # offline box, missing login: log locally and carry on
            logger.warning(f"wandb disabled: {exc}")

    @property
    def url(self) -> str:
        return getattr(self.run, "url", "") or ""

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self.run is not None:
            self.run.log(metrics, step=step)

    def log_images(self, paths: dict[str, Path]) -> None:
        if self.run is not None:
            import wandb

            self.run.log({k: wandb.Image(str(p)) for k, p in paths.items()})

    def summary(self, metrics: dict[str, Any]) -> None:
        if self.run is not None:
            for k, v in metrics.items():
                if isinstance(v, int | float | str | bool):
                    self.run.summary[k] = v

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
