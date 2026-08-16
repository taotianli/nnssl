#!/bin/bash
#SBATCH --job-name=om-jepa-weight
#SBATCH --output=/lus/lfs1aip2/projects/u6mn/openmind_jepa/nnssl_results/om_jepa_weight_%A_%a.out
#SBATCH --error=/lus/lfs1aip2/projects/u6mn/openmind_jepa/nnssl_results/om_jepa_weight_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --array=0-1%1
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
    PrimusJEPATrainer_200ep_BS8_JEPA0p01
    PrimusJEPATrainer_200ep_BS8_JEPA0p005
)
JEPA_WEIGHTS=(0.01 0.005)

TASK_ID=${SLURM_ARRAY_TASK_ID}
TRAINER=${TRAINERS[${TASK_ID}]}
JEPA_WEIGHT=${JEPA_WEIGHTS[${TASK_ID}]}

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

TASK_TMPDIR=/tmp/openmind_jepa_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
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

# A wall-time signal makes nnSSL finish the current epoch and save a checkpoint.
# On resubmission, continue that trainer instead of loading the public MAE again.
if [[ -f "${OUTPUT_DIR}/checkpoint_final.pth" || \
      -f "${OUTPUT_DIR}/checkpoint_latest.pth" || \
      -f "${OUTPUT_DIR}/checkpoint_best.pth" ]]; then
    LOAD_ARGS=(--c)
fi

echo "=== OpenMind PrimusM MAE + JEPA weight experiment ==="
echo "Array job:       ${SLURM_ARRAY_JOB_ID}"
echo "Array task:      ${SLURM_ARRAY_TASK_ID}"
echo "Node:            $(hostname)"
echo "GPU:             $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Dataset:         ${DATASET_NAME}"
echo "Trainer:         ${TRAINER}"
echo "JEPA weight:     ${JEPA_WEIGHT}"
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
