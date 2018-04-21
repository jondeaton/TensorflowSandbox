#!/usr/bin/env bash


# to list the instances that you have
gcloud compute instances list

# List the types of accelerator types available
gcloud compute accelerator-types list

# Create an instance
gcloud compute instances create	gpu-instance \
	--image-family ubuntu-1604-xenial-v20180405 \
	--accelerator type=nvidia-tesla-p100 \
	--boot-disk-auto-delete \
	--zone us-west1-b \
	--boot-disk-size 60GB \
	--boot-disk-type local-ssd

# Create a Google Cloud Storage bucket
BUCKET_NAME="my_bucket"
gsutil mb -l us-central1 "gs://$BUCKET_NAME"
