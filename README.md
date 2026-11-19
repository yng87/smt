# smt

CLI tool to submit your script to SageMaker training.

## Installation
```bash
$ uv tool install https://github.com/yng87/smt.git
```

## Usage

You need to specify python dependencies by `requirements.txt`. One way to do this is using uv
```bash
$ uv export -o requirements.txt --no-hashes
```

Then execute `smt`
```bash
$ smt --config config.yaml .
```

This command makes `tar.gz` from specified directory (current working directory in the above example), submits it to S3, and executes the job on SageMaker training.

Job configuration is specified by yaml (`config.yaml` in the above example) as
```yaml
sagemaker_training_config:
  code_destination_s3_uri: s3://<bucket>/<path>/to/store/code
  sagemaker_execution_role_arn: <SageMaker Execution IAM role ARN>
  training_image_uri: <Docker image URI to run your script>
model_trainer_config:
  # ModelTrainer V3 constructor arguments (snake_case)
  # https://sagemaker.readthedocs.io/en/v3docs/api/generated/sagemaker.train.model_trainer.html
  source_code:
    entry_script: main.py
    requirements: requirements.txt
  compute:
    instance_count: 1
    instance_type: ml.m5.large
    enable_managed_spot_training: true
  base_job_name: test-${run_id}
  stopping_condition:
    max_runtime_in_seconds: 3600
    max_wait_time_in_seconds: 3600
  hyperparameters:
    param: 42
  checkpoint_config:
    s3_uri: s3://<bucket>/checkpoints/${run_id}
  output_data_config:
    s3_output_path: s3://<bucket>/outputs/${run_id}
  input_data_config:
    - channel_name: train
      input_mode: FastFile
      data_source:
        s3_data_source:
          s3_data_type: S3Prefix
          s3_uri: s3://<bucket>/train
          s3_data_distribution_type: FullyReplicated
    - channel_name: val
      input_mode: FastFile
      data_source:
        s3_data_source:
          s3_data_type: S3Prefix
          s3_uri: s3://<bucket>/val
          s3_data_distribution_type: FullyReplicated

# Optional: Files/directories to exclude from tar.gz
exclude_patterns:
  - "*.pyc"
  - "__pycache__"
  - "*.log"
```

## Variable Substitution

You can use variable substitution in your configuration files. Currently supported variables:
- `${run_id}`: Automatically generated run ID (timestamp-based)

Example usage:
```yaml
model_trainer_config:
  checkpoint_config:
    s3_uri: s3://my-bucket/checkpoints/${run_id}
  output_data_config:
    s3_output_path: s3://my-bucket/outputs/${run_id}
```

## How to find SageMaker docker image URI?

See [AWS document](https://docs.aws.amazon.com/sagemaker/latest/dg-ecr-paths/sagemaker-algo-docker-registry-paths.html). It is also found programatically by, e.g., 

```python
from sagemaker.core import image_uris
image_uris.retrieve(framework='pytorch',region='ap-northeast-1',version='2.6.0',py_version='py312',image_scope='training', instance_type='ml.t3.large')
```

## SageMaker Training Data Access

Understanding how your training code accesses input data, saves outputs, and manages checkpoints in SageMaker:

### Input Data Access

Declare input channels in `model_trainer_config.input_data_config`. In File mode, data is copied from S3; FastFile mode provides file access on demand:

```yaml
model_trainer_config:
  input_data_config:
    - channel_name: train
      data_source: s3://my-bucket/data/train      # /opt/ml/input/data/train
    - channel_name: val
      data_source: s3://my-bucket/data/validation # /opt/ml/input/data/val
```

```python
train_path = os.environ['SM_CHANNEL_TRAIN']    # /opt/ml/input/data/train
val_path = os.environ['SM_CHANNEL_VAL']        # /opt/ml/input/data/val
```

### Output Data

SageMaker provides two types of output directories, both compressed as tar.gz and uploaded to S3:

```python
model_dir = os.environ['SM_MODEL_DIR']                 # /opt/ml/model → model.tar.gz
output_dir = os.environ['SM_OUTPUT_DIR']               # /opt/ml/output
output_data_dir = os.path.join(output_dir, "data")     # /opt/ml/output/data → output.tar.gz
```

**Config**:
```yaml
model_trainer_config:
  output_data_config:
    s3_output_path: s3://my-bucket/output/${run_id}
```

Note: `SM_OUTPUT_DIR` points to `/opt/ml/output`, but only files written under `/opt/ml/output/data` are automatically uploaded to S3. Save artifacts under `output_data_dir`.

### Checkpoints

For long training jobs, save checkpoints to `/opt/ml/checkpoints/` for automatic S3 sync:

```python
checkpoint_dir = '/opt/ml/checkpoints'
```

**Config**:
```yaml
model_trainer_config:
  checkpoint_config:
    s3_uri: s3://my-bucket/checkpoints/${run_id}
```
