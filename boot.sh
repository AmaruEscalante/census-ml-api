#!/bin/sh
export $(xargs < .env)
echo "Configuring AWS credentials"
aws configure set aws_access_key_id ${AWS_ACCESS_KEY_ID}
aws configure set aws_secret_access_key ${AWS_SECRET_ACCESS_KEY}

dvc init
echo "Configuring DVC remote"
dvc remote add -d s3remote s3://${S3_BUCKET_NAME}

echo "Pulling DVC data"
dvc pull

echo "Starting API"
exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000