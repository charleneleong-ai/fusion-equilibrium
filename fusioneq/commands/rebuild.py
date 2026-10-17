"""The Final phase's training entrypoint (fusioneq.rebuild)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from fusioneq.rebuild import Rebuild
from fusioneq.settings import SUBMISSION_DIR


def rebuild(
    spec: Annotated[Path, typer.Option(help="blend yaml, e.g. configs/blends/v21_16_grids3_fno2.yaml")],
    configs: str = "diii_d_public_test",
    out_dir: Path = SUBMISSION_DIR,
    dry_run: bool = False,
) -> None:
    """Rebuild `spec` from scratch and build `configs`. Finished steps are skipped, so an interrupted rebuild resumes
    (point FUSIONEQ_RESULTS at an empty directory for a clean one); step wall times go to
    results/rebuild_<spec>.jsonl."""
    Rebuild(spec).run(configs.split(","), out_dir, dry_run)


COMMANDS = [rebuild]
