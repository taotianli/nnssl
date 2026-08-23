#!/bin/bash
#SBATCH --job-name=om-block-b-latest
#SBATCH --output=/lus/lfs1aip2/projects/u6mn/openmind_jepa/downstream_benchmark/om_block_b_latest_%A_%a.out
#SBATCH --error=/lus/lfs1aip2/projects/u6mn/openmind_jepa/downstream_benchmark/om_block_b_latest_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=workq
# Array 0-7 maps to Block B pretraining experiment IDs 7-14.
# Each model requests one GPU. Add %N at submission to throttle.
#SBATCH --array=0-7

set -euo pipefail

REPO=${REPO:-/lus/lfs1aip2/projects/u6mn/nnssl}
VENV=${VENV:-/home/u6mn/taotl.u6mn/vllm_tutorial/.venv}
MODEL_MANIFEST=${MODEL_MANIFEST:-${REPO}/scripts/configs/openmind_mae_jepa_regularization_block_b_latest_checkpoints.tsv}

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: this script must run as a Slurm array job" >&2
    exit 2
fi
for required in "${MODEL_MANIFEST}" "${VENV}/bin/python"; do
    if [[ ! -e "${required}" ]]; then
        echo "ERROR: required path does not exist: ${required}" >&2
        exit 2
    fi
done

mapfile -t MODEL_ROWS < <(grep -Ev '^[[:space:]]*(#|$)' "${MODEL_MANIFEST}")
TASK_ID=${SLURM_ARRAY_TASK_ID}
if (( TASK_ID < 0 || TASK_ID >= ${#MODEL_ROWS[@]} )); then
    echo "ERROR: array index ${TASK_ID} is outside manifest range 0..$((${#MODEL_ROWS[@]} - 1))" >&2
    exit 2
fi

IFS=$'\t' read -r MODEL_LABEL SNAPSHOT RUN_NAME EXTRA_FIELD <<< "${MODEL_ROWS[${TASK_ID}]}"
if [[ -z "${MODEL_LABEL}" || -z "${SNAPSHOT}" || -z "${RUN_NAME}" || -n "${EXTRA_FIELD:-}" ]]; then
    echo "ERROR: manifest row ${TASK_ID} must contain exactly three tab-separated fields" >&2
    exit 2
fi

SOURCE_CHECKPOINT="$(dirname "${SNAPSHOT}")/checkpoint_latest.pth"
if [[ ! -f "${SNAPSHOT}" ]]; then
    if [[ ! -f "${SOURCE_CHECKPOINT}" ]]; then
        echo "ERROR: latest checkpoint is missing for ${MODEL_LABEL}: ${SOURCE_CHECKPOINT}" >&2
        exit 2
    fi

    TEMP_SNAPSHOT="${SNAPSHOT}.tmp.${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    trap 'rm -f "${TEMP_SNAPSHOT:-}"' EXIT
    echo "Freezing ${SOURCE_CHECKPOINT} -> ${SNAPSHOT}"
    cp --reflink=auto --preserve=timestamps "${SOURCE_CHECKPOINT}" "${TEMP_SNAPSHOT}"
    mv "${TEMP_SNAPSHOT}" "${SNAPSHOT}"
    trap - EXIT
else
    echo "Reusing frozen checkpoint: ${SNAPSHOT}"
fi

if [[ ! -s "${SNAPSHOT}" ]]; then
    echo "ERROR: frozen checkpoint is empty: ${SNAPSHOT}" >&2
    exit 2
fi

CHECKPOINT_EPOCH=$("${VENV}/bin/python" -c \
    'import sys, torch; d=torch.load(sys.argv[1], map_location="cpu", weights_only=False); print(d.get("current_epoch", d.get("epoch", "unknown")))' \
    "${SNAPSHOT}")
echo "Block B manifest index: ${TASK_ID}; pretraining experiment ID: $((TASK_ID + 7))"
echo "Frozen checkpoint epoch: ${CHECKPOINT_EPOCH}"

export MODEL_MANIFEST
export MODEL_SUITE=block_b_24h_latest
export DATASET_ID=${DATASET_ID:-201}
export RUN_NAME_SUFFIX="_E${CHECKPOINT_EPOCH}"

exec bash "${REPO}/scripts/slurm/openmind_downstream_checkpoint_array.slurm"
