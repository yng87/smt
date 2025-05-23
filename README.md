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

This command makes `tar.gz` from specified directory (current working directory in the above example), submit it to S3, and executes the job on SageMaker training.

Job configuration is specified by yaml (`config.yaml` in the above example) as
```yaml
sagemaker_training_config:
  aws_s3_bucket: <bucket name to save codes>
  aws_sm_execution_role_arn: <SageMaker Execution IAM role ARN>
  image_uri: <Docker image URI to run your script>
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
```
