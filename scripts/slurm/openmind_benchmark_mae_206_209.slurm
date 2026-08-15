#!/bin/bash
#SBATCH --job-name=om-mae-206-209
#SBATCH --output=/lus/lfs1aip2/projects/u6mn/openmind_jepa/downstream_benchmark/om_mae_206_209_%j.out
#SBATCH --error=/lus/lfs1aip2/projects/u6mn/openmind_jepa/downstream_benchmark/om_mae_206_209_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=workq

# Do not use `set -e`: if one dataset fails, record it and continue with the
# remaining datasets. Undefined variables and failed pipelines still abort the
# affected command.
set -uo pipefail

REPO=/lus/lfs1aip2/projects/u6mn/nnssl
NNUNET_REPO=/lus/lfs1aip2/projects/u6mn/nnUNet-openmind
VENV=/home/u6mn/taotl.u6mn/vllm_tutorial/.venv
PREPROCESSED_ROOT=/lus/lfs1aip2/projects/u6mn/datasets/OpenMind_downstream/segmentation_preprocessed
RESULTS_ROOT=/lus/lfs1aip2/projects/u6mn/openmind_jepa/downstream_benchmark
WEIGHTS_ROOT=/lus/lfs1aip2/projects/u6mn/openmind_jepa/weights
MODEL=PrimusM-OpenMind-MAE
CHECKPOINT=${WEIGHTS_ROOT}/${MODEL}/checkpoint_final.pth
TRAINER=PretrainedTrainer_Primus_150ep
RUN_NAME=OMBench_PrimusM_OpenMind_MAE
CONFIGURATION=3d_fullres
FOLD=0
DATASETS=(206 207 208 209)

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
export OPENMIND_DOWNSTREAM_PREPROCESSED="${PREPROCESSED_ROOT}"
export OPENMIND_DOWNSTREAM_RESULTS="${RESULTS_ROOT}"
export OPENMIND_WEIGHTS="${WEIGHTS_ROOT}"
export NNUNET_ADAPTATION_REPO="${NNUNET_REPO}"
export nnUNet_preprocessed="${PREPROCESSED_ROOT}"
export nnUNet_results="${RESULTS_ROOT}"
export nnUNet_compile=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

TASK_TMPDIR=/tmp/openmind_benchmark_${SLURM_JOB_ID}
export TMPDIR="${TASK_TMPDIR}"
mkdir -p "${TASK_TMPDIR}"
trap 'rm -rf "${TASK_TMPDIR}"' EXIT

source "${VENV}/bin/activate"
cd "${REPO}"

echo "=== OpenMind downstream benchmark: public PrimusM MAE, datasets 206-209 ==="
echo "Job:        ${SLURM_JOB_ID}"
echo "Node:       $(hostname)"
echo "GPU:        $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Checkpoint: ${CHECKPOINT}"
echo "Datasets:   ${DATASETS[*]}"
echo "Started:    $(date --iso-8601=seconds)"

for required in "${REPO}/scripts/openmind_downstream_benchmark.py" "${CHECKPOINT}" "${VENV}/bin/python"; do
    if [[ ! -e "${required}" ]]; then
        echo "ERROR: required path does not exist: ${required}" >&2
        exit 2
    fi
done

declare -a SUCCEEDED=()
declare -a FAILED=()
declare -a SKIPPED=()

for DATASET_ID in "${DATASETS[@]}"; do
    mapfile -t DATASET_DIRS < <(compgen -G "${PREPROCESSED_ROOT}/Dataset${DATASET_ID}_*" || true)
    if [[ ${#DATASET_DIRS[@]} -ne 1 ]]; then
        echo "ERROR: Dataset ${DATASET_ID} matched ${#DATASET_DIRS[@]} preprocessed directories" >&2
        FAILED+=("${DATASET_ID}:input")
        continue
    fi

    DATASET_NAME=$(basename "${DATASET_DIRS[0]}")
    OUTPUT_DIR="${RESULTS_ROOT}/${DATASET_NAME}/${TRAINER}__${RUN_NAME}__${CONFIGURATION}/fold_${FOLD}"
    RECORD="${OUTPUT_DIR}/benchmark_run.json"
    FINAL_CHECKPOINT="${OUTPUT_DIR}/checkpoint_final.pth"
    LATEST_CHECKPOINT="${OUTPUT_DIR}/checkpoint_latest.pth"

    if [[ -f "${FINAL_CHECKPOINT}" && -f "${RECORD}" ]] && grep -q '"status": "completed"' "${RECORD}"; then
        echo "=== Dataset ${DATASET_ID} (${DATASET_NAME}) already completed; skipping ==="
        SKIPPED+=("${DATASET_ID}")
        continue
    fi

    EXTRA_ARGS=()
    if [[ -f "${FINAL_CHECKPOINT}" ]]; then
        echo "Dataset ${DATASET_ID}: training is complete; resuming full-volume validation"
        EXTRA_ARGS+=(--validation-only)
    elif [[ -f "${LATEST_CHECKPOINT}" ]]; then
        echo "Dataset ${DATASET_ID}: found checkpoint_latest.pth; resuming training"
        EXTRA_ARGS+=(--continue)
    fi

    echo "=== Dataset ${DATASET_ID} (${DATASET_NAME}) started at $(date --iso-8601=seconds) ==="
    if python scripts/openmind_downstream_benchmark.py \
        --model "${MODEL}" \
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
        "${EXTRA_ARGS[@]}"; then
        SUCCEEDED+=("${DATASET_ID}")
        echo "========== Dataset ${DATASET_ID} completed =========="
    else
        STATUS=$?
        FAILED+=("${DATASET_ID}:${STATUS}")
        echo "========== Dataset ${DATASET_ID} failed with exit ${STATUS}; continuing ==========" >&2
    fi
done

echo "=== OpenMind benchmark job summary ==="
echo "Succeeded: ${SUCCEEDED[*]:-none}"
echo "Skipped:   ${SKIPPED[*]:-none}"
echo "Failed:    ${FAILED[*]:-none}"
echo "Finished:  $(date --iso-8601=seconds)"

if [[ ${#FAILED[@]} -gt 0 ]]; then
    exit 1
fi
