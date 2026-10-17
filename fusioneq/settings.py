"""Paths and machine conventions shared by every command. FUSIONEQ_RESULTS overrides where runs are written."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, TypeVar

import torch

DATA_ROOT = Path("~/fusion/data").expanduser()
CACHE_ROOT = Path("~/fusion/cache").expanduser()
RESULTS = Path(os.environ.get("FUSIONEQ_RESULTS", "~/fusion/results")).expanduser()
SUBMISSION_DIR = Path("~/fusion/submission").expanduser()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T = TypeVar("T")


def load_config(cls: type[T], path: Path) -> T:
    """A config dataclass from a committed member recipe (configs/members/), whose `kind` names the command."""
    raw: dict[str, Any] = json.loads(path.read_text())
    raw.pop("kind", None)
    return cls(**raw)
