#!/bin/bash
#SBATCH --job-name=om-downstream-array
#SBATCH --output=/lus/lfs1aip2/projects/u6mn/openmind_jepa/downstream_benchmark/om_downstream_%A_%a.out
#SBATCH --error=/lus/lfs1aip2/projects/u6mn/openmind_jepa/downstream_benchmark/om_downstream_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=workq
# Wave 2 has five models: five independent one-GPU tasks, no concurrency cap.
# A command-line --array overrides this default when selecting another suite.
# For Wave 1 models 1,3,5,7 with a two-GPU concurrency cap, use
# openmind_downstream_wave1_remaining.slurm.
#SBATCH --array=0-4

set -euo pipefail

REPO=${REPO:-/lus/lfs1aip2/projects/u6mn/nnssl}
NNUNET_REPO=${NNUNET_REPO:-/lus/lfs1aip2/projects/u6mn/nnUNet-openmind}
VENV=${VENV:-/home/u6mn/taotl.u6mn/vllm_tutorial/.venv}
PREPROCESSED_ROOT=${PREPROCESSED_ROOT:-/lus/lfs1aip2/projects/u6mn/datasets/OpenMind_downstream/segmentation_preprocessed}
RESULTS_ROOT=${RESULTS_ROOT:-/lus/lfs1aip2/projects/u6mn/openmind_jepa/downstream_benchmark}
WEIGHTS_ROOT=${WEIGHTS_ROOT:-/lus/lfs1aip2/projects/u6mn/openmind_jepa/weights}
MODEL_SUITE=${MODEL_SUITE:-}
if [[ -z "${MODEL_MANIFEST:-}" ]]; then
    MODEL_SUITE=${MODEL_SUITE:-wave2}
    case "${MODEL_SUITE}" in
        wave1)
            MODEL_MANIFEST=${REPO}/scripts/configs/openmind_mae_jepa_method_sweep_checkpoints.tsv
            ;;
        wave2)
            MODEL_MANIFEST=${REPO}/scripts/configs/openmind_mae_jepa_second_wave_checkpoints.tsv
            ;;
        *)
            echo "ERROR: unknown MODEL_SUITE=${MODEL_SUITE}; use wave1, wave2, or set MODEL_MANIFEST" >&2
            exit 2
            ;;
    esac
else
    MODEL_SUITE=${MODEL_SUITE:-custom}
fi
DATASET_ID=${DATASET_ID:-201}
TRAINER=${TRAINER:-PretrainedTrainer_Primus_150ep}
CONFIGURATION=${CONFIGURATION:-3d_fullres}
FOLD=${FOLD:-0}

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: this script must run as a Slurm array job" >&2
    exit 2
fi

for required in \
    "${REPO}/scripts/openmind_downstream_benchmark.py" \
    "${MODEL_MANIFEST}" \
    "${VENV}/bin/python" \
    "${NNUNET_REPO}/nnunetv2"; do
    if [[ ! -e "${required}" ]]; then
        echo "ERROR: required path does not exist: ${required}" >&2
        exit 2
    fi
done

# Comments and blank lines do not consume array indices.
mapfile -t MODEL_ROWS < <(grep -Ev '^[[:space:]]*(#|$)' "${MODEL_MANIFEST}")
TASK_ID=${SLURM_ARRAY_TASK_ID}
if (( TASK_ID < 0 || TASK_ID >= ${#MODEL_ROWS[@]} )); then
    echo "ERROR: array index ${TASK_ID} is outside manifest range 0..$((${#MODEL_ROWS[@]} - 1))" >&2
    exit 2
fi

IFS=$'\t' read -r MODEL_LABEL CHECKPOINT RUN_NAME EXTRA_FIELD <<< "${MODEL_ROWS[${TASK_ID}]}"
if [[ -z "${MODEL_LABEL}" || -z "${CHECKPOINT}" || -z "${RUN_NAME}" || -n "${EXTRA_FIELD:-}" ]]; then
    echo "ERROR: manifest row ${TASK_ID} must contain exactly three tab-separated fields" >&2
    exit 2
fi
if [[ -n "${RUN_NAME_SUFFIX:-}" ]]; then
    if [[ ! "${RUN_NAME_SUFFIX}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
        echo "ERROR: RUN_NAME_SUFFIX contains unsafe characters: ${RUN_NAME_SUFFIX}" >&2
        exit 2
    fi
    RUN_NAME="${RUN_NAME}${RUN_NAME_SUFFIX}"
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "ERROR: checkpoint for ${MODEL_LABEL} is not ready: ${CHECKPOINT}" >&2
    exit 2
fi
mapfile -t DATASET_DIRS < <(compgen -G "${PREPROCESSED_ROOT}/Dataset${DATASET_ID}_*" || true)
if [[ ${#DATASET_DIRS[@]} -ne 1 ]]; then
    echo "ERROR: Dataset ${DATASET_ID} matched ${#DATASET_DIRS[@]} preprocessed directories" >&2
    printf '  %s\n' "${DATASET_DIRS[@]:-no matches}" >&2
    exit 2
fi

DATASET_NAME=$(basename "${DATASET_DIRS[0]}")
OUTPUT_DIR="${RESULTS_ROOT}/${DATASET_NAME}/${TRAINER}__${RUN_NAME}__${CONFIGURATION}/fold_${FOLD}"
RECORD="${OUTPUT_DIR}/benchmark_run.json"
FINAL_CHECKPOINT="${OUTPUT_DIR}/checkpoint_final.pth"
LATEST_CHECKPOINT="${OUTPUT_DIR}/checkpoint_latest.pth"
BEST_CHECKPOINT="${OUTPUT_DIR}/checkpoint_best.pth"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
export OPENMIND_DOWNSTREAM_PREPROCESSED="${PREPROCESSED_ROOT}"
export OPENMIND_DOWNSTREAM_RESULTS="${RESULTS_ROOT}"
export OPENMIND_WEIGHTS="${WEIGHTS_ROOT}"
export NNUNET_ADAPTATION_REPO="${NNUNET_REPO}"
export nnUNet_preprocessed="${PREPROCESSED_ROOT}"
export nnUNet_results="${RESULTS_ROOT}"
export nnUNet_compile=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}

TASK_TMPDIR=/tmp/openmind_downstream_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export TMPDIR="${TASK_TMPDIR}"
mkdir -p "${TASK_TMPDIR}"
trap 'rm -rf "${TASK_TMPDIR}"' EXIT

source "${VENV}/bin/activate"
cd "${REPO}"

echo "=== OpenMind reusable downstream checkpoint benchmark ==="
echo "Array/task:    ${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}"
echo "Node:          $(hostname)"
echo "GPU:           $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Dataset:       ${DATASET_NAME}"
echo "Model suite:   ${MODEL_SUITE}"
echo "Manifest:      ${MODEL_MANIFEST}"
echo "Model label:   ${MODEL_LABEL}"
echo "Checkpoint:    ${CHECKPOINT}"
echo "Run name:      ${RUN_NAME}"
echo "Output:        ${OUTPUT_DIR}"
echo "Started:       $(date --iso-8601=seconds)"

if [[ -f "${FINAL_CHECKPOINT}" && -f "${RECORD}" ]] && grep -q '"status": "completed"' "${RECORD}"; then
    echo "Run is already completed; nothing to do."
    exit 0
fi

EXTRA_ARGS=()
if [[ -f "${FINAL_CHECKPOINT}" ]]; then
    echo "Fine-tuning is complete; running/resuming full-volume validation."
    EXTRA_ARGS+=(--validation-only)
elif [[ -f "${LATEST_CHECKPOINT}" || -f "${BEST_CHECKPOINT}" ]]; then
    echo "Found an intermediate checkpoint; resuming fine-tuning."
    EXTRA_ARGS+=(--continue)
fi

python scripts/openmind_downstream_benchmark.py \
    --model "${MODEL_LABEL}" \
    --checkpoint "${CHECKPOINT}" \
    --dataset "${DATASET_ID}" \
    --fold "${FOLD}" \
    --run-name "${RUN_NAME}" \
    --trainer "${TRAINER}" \
    --configuration "${CONFIGURATION}" \
    --nnunet-repo "${NNUNET_REPO}" \
    --preprocessed-root "${PREPROCESSED_ROOT}" \
    --results-root "${RESULTS_ROOT}" \
    --weights-root "${WEIGHTS_ROOT}" \
    "${EXTRA_ARGS[@]}"

echo "Completed: $(date --iso-8601=seconds)"
