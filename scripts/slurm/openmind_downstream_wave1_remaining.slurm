#!/bin/bash
#SBATCH --job-name=om-wave1-remain
#SBATCH --output=/lus/lfs1aip2/projects/u6mn/openmind_jepa/downstream_benchmark/om_wave1_remaining_%A_%a.out
#SBATCH --error=/lus/lfs1aip2/projects/u6mn/openmind_jepa/downstream_benchmark/om_wave1_remaining_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=workq
# Four independent one-GPU models, with at most two running concurrently.
# Command-line --array can override this selection for reuse.
#SBATCH --array=1,3,5,7%2

set -euo pipefail

# Reuse the checkpoint validation, resume, fine-tuning and full-volume
# validation logic from the generic downstream array runner. Environment
# overrides such as DATASET_ID, REPO and RESULTS_ROOT are forwarded unchanged.
export MODEL_SUITE=wave1
REPO=${REPO:-/lus/lfs1aip2/projects/u6mn/nnssl}

exec bash "${REPO}/scripts/slurm/openmind_downstream_checkpoint_array.slurm"
