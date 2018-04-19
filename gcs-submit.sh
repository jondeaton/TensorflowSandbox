#!/usr/bin/env bash
# Script for submitting job to Google ML Engine

DIR="kmer-classifier"

JOB_NAME="kmer_classifier_job_12344"
BUCKET_NAME="jons-gcs-123345454"
CLOUD_CONFIG="$DIR/cloudml-gpu.yaml"
JOB_DIR="gs://jons-gcs-123345454/test-model"  # where to save
MODULE="$DIR.model-trainer"
PACKAGE="./$DIR"
REGION="us-east1"
RUNTIME="1.2"
TRAIN_FILE="gs://jons-gcs-123345454/bacteria.kmer"

gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $JOB_DIR \
    --runtime-version $RUNTIME \
    --module-name $MODULE \
    --package-path $PACKAGE \
    --region $REGION \
    --config=$CLOUD_CONFIG \
    -- \
    --train-file $TRAIN_FILE \
    --job-name $JOB_NAME
