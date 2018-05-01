#!/usr/bin/env bash
# Test the model locally before submitting to the cloud

python -m kmer-classifier.model-trainer \
    --virus-file kmer-classifier/viral_kmers/virus.kmer \
    --bacteria-file kmer-classifier/viral_kmers/bacteria.kmer \
    --log=DEBUG