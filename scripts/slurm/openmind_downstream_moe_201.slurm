#!/bin/bash
#SBATCH --job-name=om-moe-ds201
#SBATCH --output=/lus/lfs1aip2/projects/u6mn/openmind_jepa/downstream_benchmark/om_moe_ds201_%A_%a.out
#SBATCH --error=/lus/lfs1aip2/projects/u6mn/openmind_jepa/downstream_benchmark/om_moe_ds201_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=workq
# Six independent models: one GPU per model, up to six GPUs concurrently.
# Use --array=0-5%N at submission time to impose a concurrency limit.
#SBATCH --array=0-5

set -euo pipefail

REPO=${REPO:-/lus/lfs1aip2/projects/u6mn/nnssl}
export MODEL_MANIFEST=${MODEL_MANIFEST:-${REPO}/scripts/configs/openmind_mae_jepa_moe_checkpoints.tsv}
export MODEL_SUITE=moe
export DATASET_ID=${DATASET_ID:-201}

exec bash "${REPO}/scripts/slurm/openmind_downstream_checkpoint_array.slurm"
