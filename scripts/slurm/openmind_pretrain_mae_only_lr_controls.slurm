#!/bin/bash
#SBATCH --job-name=om-mae-lr-ctrl
#SBATCH --output=/lus/lfs1aip2/projects/u6mn/openmind_jepa/nnssl_results/om_mae_lr_%A_%a.out
#SBATCH --error=/lus/lfs1aip2/projects/u6mn/openmind_jepa/nnssl_results/om_mae_lr_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --array=0-1%2
#SBATCH --signal=USR1@600

set -euo pipefail

REPO=/lus/lfs1aip2/projects/u6mn/nnssl
VENV=/home/u6mn/taotl.u6mn/vllm_tutorial/.venv
NNSSL_RAW=/lus/lfs1aip2/projects/u6mn/openmind_jepa/nnssl_raw
NNSSL_PREPROCESSED=/lus/lfs1aip2/projects/u6mn/openmind_jepa/nnssl_preprocessed
NNSSL_RESULTS=/lus/lfs1aip2/projects/u6mn/openmind_jepa/nnssl_results
MAE_CHECKPOINT=/lus/lfs1aip2/projects/u6mn/openmind_jepa/weights/PrimusM-OpenMind-MAE/checkpoint_final.pth

DATASET_ID=746
CONFIGURATION=onemmiso
PLANS=nnsslPlans
TRAINERS=(
    PrimusMAEOnlyTrainer_200ep_BS8_LR3e4
    PrimusMAEOnlyTrainer_200ep_BS8_LR3e5
)
LEARNING_RATES=(3e-4 3e-5)

TASK_ID=${SLURM_ARRAY_TASK_ID}
if (( TASK_ID < 0 || TASK_ID >= ${#TRAINERS[@]} )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${TASK_ID} is outside the supported range" >&2
    exit 2
fi
TRAINER=${TRAINERS[${TASK_ID}]}
LEARNING_RATE=${LEARNING_RATES[${TASK_ID}]}

export nnssl_raw="${NNSSL_RAW}"
export nnssl_preprocessed="${NNSSL_PREPROCESSED}"
export nnssl_results="${NNSSL_RESULTS}"
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export nnUNet_compile=false
export nnUNet_n_proc_DA=20
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_MODE=offline

TASK_TMPDIR=/tmp/openmind_mae_only_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export TMPDIR="${TASK_TMPDIR}"
mkdir -p "${TASK_TMPDIR}"
trap 'rm -rf "${TASK_TMPDIR}"' EXIT

source "${VENV}/bin/activate"
cd "${REPO}"

for required in "${VENV}/bin/python" "${MAE_CHECKPOINT}" "${NNSSL_PREPROCESSED}" "${NNSSL_RESULTS}"; do
    if [[ ! -e "${required}" ]]; then
        echo "ERROR: required path does not exist: ${required}" >&2
        exit 2
    fi
done

mapfile -t DATASET_DIRS < <(compgen -G "${NNSSL_PREPROCESSED}/Dataset${DATASET_ID}_*" || true)
if [[ ${#DATASET_DIRS[@]} -ne 1 ]]; then
    echo "ERROR: Dataset ${DATASET_ID} matched ${#DATASET_DIRS[@]} preprocessed directories" >&2
    printf '  %s\n' "${DATASET_DIRS[@]:-no matches}" >&2
    exit 2
fi

DATASET_NAME=$(basename "${DATASET_DIRS[0]}")
OUTPUT_DIR="${NNSSL_RESULTS}/${DATASET_NAME}/${TRAINER}__${PLANS}__${CONFIGURATION}/fold_all"
LOAD_ARGS=(-pretrained_weights "${MAE_CHECKPOINT}")

# Continue an interrupted run in the trainer-specific output folder. A fresh
# run starts from the same public PrimusM MAE checkpoint as the JEPA sweep.
if [[ -f "${OUTPUT_DIR}/checkpoint_final.pth" || \
      -f "${OUTPUT_DIR}/checkpoint_latest.pth" || \
      -f "${OUTPUT_DIR}/checkpoint_best.pth" ]]; then
    LOAD_ARGS=(--c)
fi

echo "=== OpenMind PrimusM MAE-only continued-pretraining control ==="
echo "Array job:       ${SLURM_ARRAY_JOB_ID}"
echo "Array task:      ${SLURM_ARRAY_TASK_ID}"
echo "Node:            $(hostname)"
echo "GPU:             $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Dataset:         ${DATASET_NAME}"
echo "Trainer:         ${TRAINER}"
echo "Learning rate:   ${LEARNING_RATE}"
echo "MAE checkpoint:  ${MAE_CHECKPOINT}"
echo "Output:          ${OUTPUT_DIR}"
echo "Load arguments:  ${LOAD_ARGS[*]}"
echo "Started:         $(date --iso-8601=seconds)"

python -m nnssl.run.run_training \
    "${DATASET_ID}" "${CONFIGURATION}" \
    -tr "${TRAINER}" \
    -p "${PLANS}" \
    "${LOAD_ARGS[@]}"

echo "Completed: $(date --iso-8601=seconds)"
