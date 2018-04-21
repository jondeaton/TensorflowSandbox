#!/usr/bin/env bash
# Script for submitting job to Google ML Engine

DIR="kmer-classifier"

timestamp=`date +%s`

PROJECT_ID=`gcloud config list project --format "value(core.project)"`

JOB_NAME="kmer_classifier_job_$timestamp"
BUCKET_NAME="jons-gcs-123345454"
CLOUD_CONFIG="$DIR/cloudml-gpu.yaml"
JOB_DIR="gs://$BUCKET_NAME/test-model"  # where to save
MODULE="$DIR.model-trainer"
PACKAGE="./$DIR"
REGION="us-east1"
RUNTIME="1.0"

virus_file="gs://$BUCKET_NAME/viruses.kmer"
bacteria_file="gs://$BUCKET_NAME/bacteria.kmer"

gcloud ml-engine jobs submit training "$JOB_NAME" \
    --job-dir "$JOB_DIR" \
    --runtime-version "$RUNTIME" \
    --module-name "$MODULE" \
    --package-path "$PACKAGE" \
    --region "$REGION" \
    --config="$CLOUD_CONFIG" \
    -- \
    --virus-file "$virus_file" \
    --bacteria-file "$bacteria_file"
