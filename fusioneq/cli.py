"""fusioneq command line: every command, collected from the COMMANDS list of each module in fusioneq.commands
(typer names them from the function: prepare_thomson -> prepare-thomson).

python -m fusioneq.cli prepare --config diii_d_train
python -m fusioneq.cli train --tag mlp
python -m fusioneq.cli build --model ~/fusion/results/<ensemble>
"""

from __future__ import annotations

import importlib
import pkgutil

import typer

import fusioneq.commands

app = typer.Typer(add_completion=False)
for info in pkgutil.iter_modules(fusioneq.commands.__path__):
    for command in importlib.import_module(f"fusioneq.commands.{info.name}").COMMANDS:
        app.command()(command)

if __name__ == "__main__":
    app()
