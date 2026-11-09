#!/bin/bash
# run_cnn_condor.sh
set -e

echo "Job started on $(hostname) at $(date)"

# If you use a specific conda environment or virtualenv, activate it here
# (Even with getenv=True, explicitly activating in batch jobs prevents path bugs)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your_env_name

python3 -u train_cnn.py "$@"

echo "Job finished at $(date)"