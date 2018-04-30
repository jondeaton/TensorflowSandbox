#!/usr/bin/env bash
# Script for submitting SIGNS job to Google ML Engine

dir="SIGNS"
proj_name="signs"

timestamp=`date +%s`

# Get the current project ID
project_id=`gcloud config list project --format "value(core.project)"`

job_name=$proj_name"_job_$timestamp"
bucket_name="jons-gcs-123345454"
cloud_config="$dir/cloudml-gpu.yaml"
job_dir="gs://$bucket_name/test-model"  # where to save
module="$dir.train"
package="./$dir"
region="us-east1"
runtime="1.0"

# Project specific Google Cloud Storage locations
dataset_dir="gs://$bucket_name/datasets/$proj_name"
tensorboard_dir="gs://$bucket_name/$proj_name/tensorboard"
save_file="gs://$bucket_name/$proj_name/models/$job_name"
log_file="gs://$bucket_name/$proj_name/logs/$job_name"

gcloud ml-engine jobs submit training "$job_name" \
    --job-dir "$job_dir" \
    --runtime-version "$runtime" \
    --module-name "$module" \
    --package-path "$package" \
    --region "$region" \
    --config="$cloud_config" \
    -- \
    --job-name "$job_name" \
    --tensorboard "$tensorboard_dir" \
    --learning-rate "0.0001" \
    --epochs 1500 \
    --mini-batch 512 \
    --log=DEBUG \
    --log-file "$log_file" \
    --cloud