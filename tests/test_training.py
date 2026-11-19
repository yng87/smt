from pathlib import Path
from unittest.mock import Mock

import boto3
import pytest
import yaml
from pydantic import ValidationError
from sagemaker.core.helper.session_helper import Session
from sagemaker.train.model_trainer import ModelTrainer

from smt.utils import AppConfig, submit_training_job


@pytest.fixture
def config_data():
    data = yaml.safe_load(Path("example_config.yaml").read_text())
    data["sagemaker_training_config"]["run_id"] = "20261119T120000123456"
    return data


@pytest.fixture
def session(monkeypatch):
    # Explicit test credentials prevent credential discovery and metadata requests.
    boto_session = boto3.Session(
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
        region_name="us-east-1",
    )
    session = Session(boto_session=boto_session, default_bucket="my-sagemaker-bucket")
    monkeypatch.setattr("smt.utils.Session", lambda: session)
    # IAM role verification is an external service boundary.
    monkeypatch.setattr(
        "sagemaker.train.defaults.TrainDefaults.get_role",
        lambda role, sagemaker_session: role,
    )
    return session


@pytest.mark.parametrize("with_inputs", [True, False])
def test_submit_training_job_uses_v3_config(
    tmp_path, monkeypatch, config_data, session, with_inputs
):
    import tarfile

    if not with_inputs:
        del config_data["model_trainer_config"]["input_data_config"]
    config_data["model_trainer_config"]["hyperparameters"].update(
        label="hello world", enabled=True, layers=[32, 64]
    )
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config_data))
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "main.py").write_text('print("training")')
    (source_dir / "requirements.txt").write_text("")
    (source_dir / "ignored.pyc").write_text("ignored")
    monkeypatch.chdir(tmp_path)
    uploaded = []

    def upload_data(path, bucket, key_prefix):
        with tarfile.open(path) as archive:
            uploaded.extend(archive.getnames())
        return f"s3://{bucket}/{key_prefix}/{Path(path).name}"

    monkeypatch.setattr(session, "upload_data", upload_data)
    trainers = []

    def train(self, *, wait):
        assert wait is False
        trainers.append(self)

    # Keep the real SDK constructor to exercise our YAML-to-V3 boundary.
    monkeypatch.setattr(ModelTrainer, "train", train)
    submit_training_job(source_dir, config_file)

    (trainer,) = trainers
    assert trainer.sagemaker_session is session
    assert (
        trainer.role
        == config_data["sagemaker_training_config"]["sagemaker_execution_role_arn"]
    )
    assert (
        trainer.training_image
        == config_data["sagemaker_training_config"]["training_image_uri"]
    )
    assert trainer.base_job_name == "my-training-job-20261119T120000123456"
    assert (
        trainer.source_code.source_dir
        == "s3://my-sagemaker-bucket/code/trainer_20261119T120000123456.tar.gz"
    )
    assert trainer.source_code.entry_script == "main.py"
    assert trainer.source_code.requirements == "requirements.txt"
    assert trainer.compute.instance_type == "ml.m5.large"
    assert trainer.compute.enable_managed_spot_training is True
    assert trainer.stopping_condition.max_runtime_in_seconds == 3600
    assert trainer.stopping_condition.max_wait_time_in_seconds == 3600
    assert trainer.checkpoint_config.s3_uri.endswith("/20261119T120000123456")
    assert trainer.output_data_config.s3_output_path.endswith("/20261119T120000123456")
    assert trainer.training_input_mode == "FastFile"
    if with_inputs:
        (channel,) = trainer.input_data_config
        assert channel.channel_name == "train"
        assert channel.data_source == "s3://my-sagemaker-bucket/input"
    else:
        assert trainer.input_data_config is None
    assert trainer.hyperparameters == {
        "epochs": 10,
        "learning_rate": 0.001,
        "label": "hello world",
        "enabled": True,
        "layers": [32, 64],
    }
    assert sorted(uploaded) == ["main.py", "requirements.txt"]
    assert not list(tmp_path.glob("trainer_*.tar.gz"))


def test_config_substitution_does_not_mutate_original(config_data):
    config = AppConfig.model_validate(config_data)
    args = config.to_model_trainer_args("s3://bucket/code.tar.gz")
    assert args["output_data_config"]["s3_output_path"].endswith(
        "/20261119T120000123456"
    )
    assert config.model_trainer_config["output_data_config"]["s3_output_path"].endswith(
        "/${run_id}"
    )
    assert "source_dir" not in config.model_trainer_config["source_code"]


@pytest.mark.parametrize("legacy_key", ["estimator_config", "input_s3_uri"])
def test_rejects_v2_config(config_data, legacy_key):
    if legacy_key == "estimator_config":
        config_data[legacy_key] = config_data.pop("model_trainer_config")
    else:
        config_data["sagemaker_training_config"][legacy_key] = "s3://bucket/input"
    with pytest.raises(ValidationError, match=legacy_key):
        AppConfig.model_validate(config_data)


def test_upload_failure_is_propagated(tmp_path, monkeypatch, config_data, session):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config_data))
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "main.py").write_text("")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        session, "upload_data", Mock(side_effect=RuntimeError("upload failed"))
    )
    train = Mock()
    monkeypatch.setattr(ModelTrainer, "train", train)
    with pytest.raises(RuntimeError, match="upload failed"):
        submit_training_job(source_dir, config_file)
    train.assert_not_called()
    assert not list(tmp_path.glob("trainer_*.tar.gz"))
