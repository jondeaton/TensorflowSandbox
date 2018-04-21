#!/usr/bin/env bash
# Test the model locally before submitting to the cloud

python -m kmer-classifier.model-trainer \
    --virus_file kmer-classifier/viral_kmers/virus.kmer \
    --bacteria_file kmer-classifier/viral_kmers/bacteria.kmer