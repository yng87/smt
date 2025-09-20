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
  input_s3_uri:
    # str or mapping
    train: s3://<bucket>/train
    val: s3://<bucket>/val
estimator_config:
  # see sagemaker document for available configurations
  # https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html#sagemaker.estimator.Estimator
  entry_point: main.py
  instance_count: 1
  instance_type: ml.m5.large
  base_job_name: test
  use_spot_instances: true
  max_run: 3600
  max_wait: 3600
  hyperparameters:
    param: 42
  checkpoint_s3_uri: s3://<bucket>/checkpoints/${run_id}
  output_path: s3://<bucket>/outputs/${run_id}
  input_mode: FastFile

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
estimator_config:
  checkpoint_s3_uri: s3://my-bucket/checkpoints/${run_id}
  output_path: s3://my-bucket/outputs/${run_id}
```

## How to find SageMaker docker image URI?

See [AWS document](https://docs.aws.amazon.com/sagemaker/latest/dg-ecr-paths/sagemaker-algo-docker-registry-paths.html). It is also found programatically by, e.g., 

```python
from sagemaker import image_uris
image_uris.retrieve(framework='pytorch',region='ap-northeast-1',version='2.6.0',py_version='py312',image_scope='training', instance_type='ml.t3.large')
```
