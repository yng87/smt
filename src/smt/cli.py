import json
import os
import tarfile
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import Annotated, Any, Self

import sagemaker
import typer
import yaml
from pydantic import BaseModel, Field

app = typer.Typer()


logger = getLogger(__name__)
sagemaker_session = sagemaker.session.Session()  # type: ignore


class SagemakerTrainingConfig(BaseModel):
    aws_s3_bucket: str
    aws_sm_execution_role_arn: str
    image_uri: str
    run_id: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y%m%dT%H%M%S%f")
    )


def json_encode_hyperparameters(hyperparameters: dict[str, Any]) -> dict[str, str]:
    return {str(k): json.dumps(v) for (k, v) in hyperparameters.items()}


class EstimatorConfig(BaseModel):
    entry_point: str
    instance_count: int
    instance_type: str
    base_job_name: str
    use_spot_instances: bool
    max_run: int | None = None
    max_wait: int | None = None
    hyperparameters: dict[str, Any]

    def to_estimator_args(self) -> dict[str, Any]:
        args = self.model_dump()
        if not self.use_spot_instances:
            args.pop("max_run")
            args.pop("max_wait")
        args["hyperparameters"] = json_encode_hyperparameters(args["hyperparameters"])
        return args


class AppConfig(BaseModel):
    sagemaker_training_config: SagemakerTrainingConfig
    estimator_config: EstimatorConfig

    @classmethod
    def from_yaml(cls, filename: str) -> Self:
        with open(filename, "r") as f:
            return cls(**yaml.safe_load(f))


def create_tar_file(source_dir: str, target_filename: str):
    with tarfile.open(target_filename, "w:gz") as tar:
        for root, _, files in os.walk(source_dir):
            if root.endswith(".venv"):
                continue
            for file in files:
                full_path = os.path.join(root, file)
                tar.add(full_path, arcname=os.path.relpath(full_path, source_dir))


def prepare_training_code_on_s3(
    sm_settings: SagemakerTrainingConfig, trainer_dir: str
) -> str:
    trainer_filename = f"trainer_{sm_settings.run_id}.tar.gz"
    try:
        create_tar_file(trainer_dir, trainer_filename)
        sources = sagemaker_session.upload_data(
            trainer_filename, sm_settings.aws_s3_bucket, "code"
        )
    finally:
        os.remove(trainer_filename)

    return sources


@app.command()
def submit(
    trainer_dir: Annotated[Path, typer.Argument(help="Trainer directory")] = "trainer",
    config: Annotated[Path, typer.Option(help="Config file name")] = "config.yaml",
):
    app_config = AppConfig.from_yaml(config)
    sm_settings = app_config.sagemaker_training_config
    logger.info(f"Run ID: {sm_settings.run_id}")

    trainer_sources = prepare_training_code_on_s3(sm_settings, trainer_dir)

    logger.info("Start training")
    estimator = sagemaker.estimator.Estimator(
        sm_settings.image_uri,
        sm_settings.aws_sm_execution_role_arn,
        source_dir=trainer_sources,
        **app_config.estimator_config.to_estimator_args(),
    )
    estimator.fit(wait=False)
