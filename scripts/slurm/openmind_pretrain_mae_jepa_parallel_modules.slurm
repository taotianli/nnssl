#!/bin/bash
#SBATCH --job-name=om-jepa-modules
#SBATCH --output=/lus/lfs1aip2/projects/u6mn/openmind_jepa/nnssl_results/om_jepa_modules_%A_%a.out
#SBATCH --error=/lus/lfs1aip2/projects/u6mn/openmind_jepa/nnssl_results/om_jepa_modules_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --array=0-2
#SBATCH --signal=B:USR1@600

set -euo pipefail

REPO=${REPO:-/lus/lfs1aip2/projects/u6mn/nnssl}
VENV=${VENV:-/home/u6mn/taotl.u6mn/vllm_tutorial/.venv}
NNSSL_RAW=${NNSSL_RAW:-/lus/lfs1aip2/projects/u6mn/openmind_jepa/nnssl_raw}
NNSSL_PREPROCESSED=${NNSSL_PREPROCESSED:-/lus/lfs1aip2/projects/u6mn/openmind_jepa/nnssl_preprocessed}
NNSSL_RESULTS=${NNSSL_RESULTS:-/lus/lfs1aip2/projects/u6mn/openmind_jepa/nnssl_results}
MAE_CHECKPOINT=${MAE_CHECKPOINT:-/lus/lfs1aip2/projects/u6mn/openmind_jepa/weights/PrimusM-OpenMind-MAE/checkpoint_final.pth}
SWEEP_MANIFEST=${SWEEP_MANIFEST:-${REPO}/scripts/configs/openmind_mae_jepa_parallel_modules.tsv}

DATASET_ID=${DATASET_ID:-746}
CONFIGURATION=${CONFIGURATION:-onemmiso}
PLANS=${PLANS:-nnsslPlans}
EXPERIMENT_SEED=${EXPERIMENT_SEED:-20260824}

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: submit with sbatch --array=<ids>" >&2
    exit 2
fi
ROW=$(awk -F '\t' -v id="${SLURM_ARRAY_TASK_ID}" '$1 == id {print; exit}' "${SWEEP_MANIFEST}")
if [[ -z "${ROW}" ]]; then
    echo "ERROR: no manifest row for task ${SLURM_ARRAY_TASK_ID}: ${SWEEP_MANIFEST}" >&2
    exit 2
fi
IFS=$'\t' read -r TASK_ID TRAINER DESCRIPTION <<< "${ROW}"

export nnssl_raw="${NNSSL_RAW}"
export nnssl_preprocessed="${NNSSL_PREPROCESSED}"
export nnssl_results="${NNSSL_RESULTS}"
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export nnUNet_compile=false
export nnUNet_n_proc_DA=${NNUNET_N_PROC_DA:-20}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_MODE=${WANDB_MODE:-offline}
export PYTHONHASHSEED="${EXPERIMENT_SEED}"
export NNSSL_EXPERIMENT_SEED="${EXPERIMENT_SEED}"
# Do not inherit a leftover calibration override from the submit shell. These
# three matched experiments intentionally use the fixed global VICReg weight.
export NNSSL_REGULARIZER_WEIGHT=0.005

TASK_TMPDIR=/tmp/openmind_jepa_modules_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export TMPDIR="${TASK_TMPDIR}"
mkdir -p "${TASK_TMPDIR}"
trap 'rm -rf "${TASK_TMPDIR}"' EXIT

source "${VENV}/bin/activate"
cd "${REPO}"

for required in "${VENV}/bin/python" "${NNSSL_PREPROCESSED}" "${NNSSL_RESULTS}" "${SWEEP_MANIFEST}"; do
    if [[ ! -e "${required}" ]]; then
        echo "ERROR: required path does not exist: ${required}" >&2
        exit 2
    fi
done

mapfile -t DATASET_DIRS < <(compgen -G "${NNSSL_PREPROCESSED}/Dataset${DATASET_ID}_*" || true)
if [[ ${#DATASET_DIRS[@]} -ne 1 ]]; then
    echo "ERROR: Dataset ${DATASET_ID} matched ${#DATASET_DIRS[@]} directories" >&2
    exit 2
fi
DATASET_NAME=$(basename "${DATASET_DIRS[0]}")
OUTPUT_DIR="${NNSSL_RESULTS}/${DATASET_NAME}/${TRAINER}__${PLANS}__${CONFIGURATION}/fold_all"
LOAD_ARGS=(-pretrained_weights "${MAE_CHECKPOINT}")
if [[ -f "${OUTPUT_DIR}/checkpoint_final.pth" || -f "${OUTPUT_DIR}/checkpoint_latest.pth" || -f "${OUTPUT_DIR}/checkpoint_best.pth" ]]; then
    LOAD_ARGS=(--c)
fi
if [[ "${LOAD_ARGS[0]}" == "-pretrained_weights" && ! -f "${MAE_CHECKPOINT}" ]]; then
    echo "ERROR: public MAE checkpoint is required for a fresh run: ${MAE_CHECKPOINT}" >&2
    exit 2
fi

echo "=== OpenMind parallel module experiments ==="
echo "Array/task:     ${SLURM_ARRAY_JOB_ID}/${TASK_ID}"
echo "Trainer:        ${TRAINER}"
echo "Description:    ${DESCRIPTION}"
echo "Seed:           ${EXPERIMENT_SEED}"
echo "Workers:        ${nnUNet_n_proc_DA}"
echo "Node/GPU:       $(hostname) / $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Dataset:        ${DATASET_NAME}"
echo "MAE checkpoint: ${MAE_CHECKPOINT}"
echo "Output:         ${OUTPUT_DIR}"
echo "Load arguments: ${LOAD_ARGS[*]}"

python -m nnssl.run.run_training \
    "${DATASET_ID}" "${CONFIGURATION}" \
    -tr "${TRAINER}" \
    -p "${PLANS}" \
    "${LOAD_ARGS[@]}"
