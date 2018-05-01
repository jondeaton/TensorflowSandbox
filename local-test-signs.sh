#!/usr/bin/env bash
# Test the model locally before submitting to the cloud

timestamp=`date +%s`

job_name="signs_local_$timestamp"
tensorboard_dir="SIGNS/tensorboard"
dataset_dir="SIGNS/datasets"

python -m SIGNS.train \
    --job-name "$job_name" \
    --tensorboard "$tensorboard_dir" \
    --dataset-dir "$dataset_dir" \
    --learning-rate "0.0001" \
    --epochs 1500 \
    --mini-batch 512 \
    --log=DEBUG