


gcloud compute instances list



gcloud compute accelerator-types list



gcloud compute instances create	gpu-instance \
	--image-family ubuntu-1604-xenial-v20180405 \
	--accelerator type=nvidia-tesla-p100 \
	--boot-disk-auto-delete \
	--zone us-west1-b \
	--boot-disk-size 60GB \
	--boot-disk-type local-ssd
