from pathlib import Path
from typing import Annotated

import typer

from .utils import submit_training_job

app = typer.Typer()


@app.command()
def submit(
    trainer_dir: Annotated[Path, typer.Argument(help="Trainer directory")] = Path.cwd(),
    config: Annotated[Path, typer.Option(help="Config file name")] = Path(
        "./config.yaml"
    ),
):
    submit_training_job(trainer_dir, config)
