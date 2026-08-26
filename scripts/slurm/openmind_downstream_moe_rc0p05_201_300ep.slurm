#!/bin/bash
#SBATCH --job-name=om-moe300-ds201
#SBATCH --output=/lus/lfs1aip2/projects/u6mn/openmind_jepa/downstream_benchmark/om_moe300_ds201_%j.out
#SBATCH --error=/lus/lfs1aip2/projects/u6mn/openmind_jepa/downstream_benchmark/om_moe300_ds201_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --array=0-0

set -euo pipefail

REPO=${REPO:-/lus/lfs1aip2/projects/u6mn/nnssl}
export MODEL_MANIFEST=${MODEL_MANIFEST:-${REPO}/scripts/configs/openmind_moe_rc0p05_300ep_checkpoint.tsv}
export MODEL_SUITE=moe_rc0p05_300ep
export DATASET_ID=${DATASET_ID:-201}
export DOWNSTREAM_EPOCHS=300

exec bash "${REPO}/scripts/slurm/openmind_downstream_checkpoint_array.slurm"
