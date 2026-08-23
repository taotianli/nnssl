#!/bin/bash
#SBATCH --job-name=om-block-a-latest
#SBATCH --output=/lus/lfs1aip2/projects/u6mn/openmind_jepa/downstream_benchmark/om_block_a_latest_%A_%a.out
#SBATCH --error=/lus/lfs1aip2/projects/u6mn/openmind_jepa/downstream_benchmark/om_block_a_latest_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=workq
# Seven Block A models, one GPU per model. Add %N at submission to throttle.
#SBATCH --array=0-6

set -euo pipefail

REPO=${REPO:-/lus/lfs1aip2/projects/u6mn/nnssl}
VENV=${VENV:-/home/u6mn/taotl.u6mn/vllm_tutorial/.venv}
MODEL_MANIFEST=${MODEL_MANIFEST:-${REPO}/scripts/configs/openmind_mae_jepa_regularization_block_a_latest_checkpoints.tsv}

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

    # Freeze the interrupted 24-hour checkpoint before downstream training.
    # The atomic rename prevents a partial copy from being treated as valid.
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
echo "Frozen checkpoint epoch: ${CHECKPOINT_EPOCH}"

export MODEL_MANIFEST
export MODEL_SUITE=block_a_24h_latest
export DATASET_ID=${DATASET_ID:-201}
export RUN_NAME_SUFFIX="_E${CHECKPOINT_EPOCH}"

exec bash "${REPO}/scripts/slurm/openmind_downstream_checkpoint_array.slurm"
