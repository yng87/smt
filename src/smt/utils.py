import json
import os
import tarfile
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import Any, Self

import sagemaker
import yaml
from pydantic import BaseModel, Field

logger = getLogger(__name__)
sagemaker_session = sagemaker.session.Session()  # type: ignore


class SagemakerTrainingConfig(BaseModel):
    aws_s3_bucket: str
    aws_sm_execution_role_arn: str
    image_uri: str
    input_s3_uri: str | dict[str, str] | None = None
    run_id: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y%m%dT%H%M%S%f")
    )


def json_encode_hyperparameters(hyperparameters: dict[str, Any]) -> dict[str, str]:
    return {str(k): json.dumps(v) for (k, v) in hyperparameters.items()}


class AppConfig(BaseModel):
    sagemaker_training_config: SagemakerTrainingConfig
    estimator_config: dict[str, Any]

    @classmethod
    def from_yaml(cls, filename: str) -> Self:
        with open(filename, "r") as f:
            return cls(**yaml.safe_load(f))

    def to_estimator_args(self) -> dict[str, Any]:
        args = self.estimator_config.copy()
        if "hyperparameters" in args:
            args["hyperparameters"] = json_encode_hyperparameters(
                args["hyperparameters"]
            )
        return args

    def build_job_name(self) -> str:
        base_job_name = self.estimator_config.get("base_job_name", "job")
        return f"{base_job_name}-{self.sagemaker_training_config.run_id}"


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


def submit_training_job(trainer_dir: Path, config: Path):
    app_config = AppConfig.from_yaml(config)
    sm_settings = app_config.sagemaker_training_config
    logger.info(f"Run ID: {sm_settings.run_id}")

    trainer_sources = prepare_training_code_on_s3(sm_settings, trainer_dir)

    logger.info("Start training")
    estimator = sagemaker.estimator.Estimator(
        sm_settings.image_uri,
        sm_settings.aws_sm_execution_role_arn,
        source_dir=trainer_sources,
        **app_config.to_estimator_args(),
    )
    estimator.fit(
        job_name=app_config.build_job_name(),
        inputs=sm_settings.input_s3_uri,
        wait=False,
    )
