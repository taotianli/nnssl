#!/bin/bash
#SBATCH --job-name=om-jepa-wave2
#SBATCH --output=/lus/lfs1aip2/projects/u6mn/openmind_jepa/nnssl_results/om_jepa_wave2_%A_%a.out
#SBATCH --error=/lus/lfs1aip2/projects/u6mn/openmind_jepa/nnssl_results/om_jepa_wave2_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --array=0-4
#SBATCH --signal=USR1@600

set -euo pipefail

REPO=${REPO:-/lus/lfs1aip2/projects/u6mn/nnssl}
VENV=${VENV:-/home/u6mn/taotl.u6mn/vllm_tutorial/.venv}
NNSSL_RAW=${NNSSL_RAW:-/lus/lfs1aip2/projects/u6mn/openmind_jepa/nnssl_raw}
NNSSL_PREPROCESSED=${NNSSL_PREPROCESSED:-/lus/lfs1aip2/projects/u6mn/openmind_jepa/nnssl_preprocessed}
NNSSL_RESULTS=${NNSSL_RESULTS:-/lus/lfs1aip2/projects/u6mn/openmind_jepa/nnssl_results}
MAE_CHECKPOINT=${MAE_CHECKPOINT:-/lus/lfs1aip2/projects/u6mn/openmind_jepa/weights/PrimusM-OpenMind-MAE/checkpoint_final.pth}

DATASET_ID=${DATASET_ID:-746}
CONFIGURATION=${CONFIGURATION:-onemmiso}
PLANS=${PLANS:-nnsslPlans}
TRAINERS=(
    PrimusJEPASerialCAE_200ep_BS8_L0p005
    PrimusJEPAGatedSkip_200ep_BS8_G0p25
    PrimusJEPAIndependentCrossView_200ep_BS4_Weak
    PrimusJEPAStructure_200ep_BS8_E0p05
    PrimusJEPAGradGuard_200ep_BS4_R0p25
)

TASK_ID=${SLURM_ARRAY_TASK_ID}
if (( TASK_ID < 0 || TASK_ID >= ${#TRAINERS[@]} )); then
    echo "ERROR: array index ${TASK_ID} is outside 0..$((${#TRAINERS[@]} - 1))" >&2
    exit 2
fi
TRAINER=${TRAINERS[${TASK_ID}]}

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

TASK_TMPDIR=/tmp/openmind_jepa_wave2_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
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
    echo "ERROR: Dataset ${DATASET_ID} matched ${#DATASET_DIRS[@]} directories" >&2
    printf '  %s\n' "${DATASET_DIRS[@]:-no matches}" >&2
    exit 2
fi

DATASET_NAME=$(basename "${DATASET_DIRS[0]}")
OUTPUT_DIR="${NNSSL_RESULTS}/${DATASET_NAME}/${TRAINER}__${PLANS}__${CONFIGURATION}/fold_all"
LOAD_ARGS=(-pretrained_weights "${MAE_CHECKPOINT}")
if [[ -f "${OUTPUT_DIR}/checkpoint_final.pth" || -f "${OUTPUT_DIR}/checkpoint_latest.pth" || -f "${OUTPUT_DIR}/checkpoint_best.pth" ]]; then
    LOAD_ARGS=(--c)
fi

echo "=== OpenMind MAE+JEPA second-wave sweep ==="
echo "Array/task: ${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}"
echo "Node:       $(hostname)"
echo "GPU:        $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Dataset:    ${DATASET_NAME}"
echo "Trainer:    ${TRAINER}"
echo "Checkpoint: ${MAE_CHECKPOINT}"
echo "Output:     ${OUTPUT_DIR}"
echo "Load args:  ${LOAD_ARGS[*]}"

python -m nnssl.run.run_training \
    "${DATASET_ID}" "${CONFIGURATION}" \
    -tr "${TRAINER}" \
    -p "${PLANS}" \
    "${LOAD_ARGS[@]}"

echo "Completed: $(date --iso-8601=seconds)"
