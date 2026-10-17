"""Write submissions/pointer_<UTC time>_<rev>.zip for Codabench from HF_READ_TOKEN in .env. Never prints the token.

    uv run python scripts/write_pointer.py                # pins the latest commit
    uv run python scripts/write_pointer.py --revision <sha or prefix>

Refuses anything but a fine-grained read token (the starter kit's own check), and verifies with that token that the
scorer will see both .npz at the pinned commit before writing the zip.
"""

from __future__ import annotations

import json
import os
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "starter"))
from push_predictions import validate_read_token  # noqa: E402

REPO = "chaleong/fusion-eq-predictions"
EXPECTED = {"diii_d_public_test.npz", "mast_public_test.npz"}


def main(
    revision: Annotated[str, typer.Option(help="commit SHA or unique prefix to pin; default: the latest commit")] = "",
    repo: str = REPO,
    out: Annotated[Path | None, typer.Option(help="default: submissions/pointer_<UTC time>_<rev>.zip")] = None,
) -> None:
    load_dotenv(ROOT / ".env")
    token = os.environ.get("HF_READ_TOKEN", "").strip()
    if not token:
        raise typer.Exit("HF_READ_TOKEN is not set in .env")
    api = HfApi()
    if validate_read_token(api, token, repo):
        raise typer.Exit(1)
    commits = [c.commit_id for c in api.list_repo_commits(repo, repo_type="dataset", token=token)]
    matches = [c for c in commits if c.startswith(revision)] if revision else commits[:1]
    if len(matches) != 1:
        raise typer.Exit(f"revision {revision!r} matches {len(matches)} commits of {repo}; give more characters")
    revision = matches[0]
    seen = {
        f for f in api.list_repo_files(repo, repo_type="dataset", revision=revision, token=token) if f.endswith(".npz")
    }
    if not EXPECTED <= seen:
        raise typer.Exit(f"the read token sees {sorted(seen)} at {revision[:8]}, expected {sorted(EXPECTED)}")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%MZ")
    out = out or ROOT / "submissions" / f"pointer_{stamp}_{revision[:8]}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("manifest.json", json.dumps({"repo_id": repo, "revision": revision, "token": token}))
    typer.echo(f"wrote {out} ({out.stat().st_size} bytes): {repo}@{revision[:8]}, read token verified, {sorted(seen)}")


if __name__ == "__main__":
    typer.run(main)
