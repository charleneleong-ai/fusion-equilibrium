"""Paths and machine conventions shared by every command. FUSIONEQ_RESULTS overrides where runs are written."""

from __future__ import annotations

import os
from pathlib import Path

import torch

DATA_ROOT = Path("~/fusion/data").expanduser()
CACHE_ROOT = Path("~/fusion/cache").expanduser()
RESULTS = Path(os.environ.get("FUSIONEQ_RESULTS", "~/fusion/results")).expanduser()
SUBMISSION_DIR = Path("~/fusion/submission").expanduser()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
