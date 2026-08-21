#!/bin/bash
#SBATCH --job-name=om-reg-cal
#SBATCH --output=/lus/lfs1aip2/projects/u6mn/openmind_jepa/nnssl_results/om_reg_cal_%A_%a.out
#SBATCH --error=/lus/lfs1aip2/projects/u6mn/openmind_jepa/nnssl_results/om_reg_cal_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --partition=workq
#SBATCH --array=0-6

set -euo pipefail

REPO=${REPO:-/lus/lfs1aip2/projects/u6mn/nnssl}
VENV=${VENV:-/home/u6mn/taotl.u6mn/vllm_tutorial/.venv}
NNSSL_RAW=${NNSSL_RAW:-/lus/lfs1aip2/projects/u6mn/openmind_jepa/nnssl_raw}
NNSSL_PREPROCESSED=${NNSSL_PREPROCESSED:-/lus/lfs1aip2/projects/u6mn/openmind_jepa/nnssl_preprocessed}
NNSSL_RESULTS=${NNSSL_RESULTS:-/lus/lfs1aip2/projects/u6mn/openmind_jepa/nnssl_results}
MAE_CHECKPOINT=${MAE_CHECKPOINT:-/lus/lfs1aip2/projects/u6mn/openmind_jepa/weights/PrimusM-OpenMind-MAE/checkpoint_final.pth}
CALIBRATION_DIR=${CALIBRATION_DIR:-/lus/lfs1aip2/projects/u6mn/openmind_jepa/regularizer_calibration}
CALIBRATION_MANIFEST=${CALIBRATION_MANIFEST:-${REPO}/scripts/configs/openmind_mae_jepa_regularization_calibration.tsv}
DATASET_ID=${DATASET_ID:-746}
CONFIGURATION=${CONFIGURATION:-onemmiso}
PLANS=${PLANS:-nnsslPlans}
CALIBRATION_BATCHES=${CALIBRATION_BATCHES:-200}
EXPERIMENT_SEED=${EXPERIMENT_SEED:-20260821}

ROW=$(awk -F '\t' -v id="${SLURM_ARRAY_TASK_ID}" '$1 == id {print; exit}' "${CALIBRATION_MANIFEST}")
if [[ -z "${ROW}" ]]; then
    echo "ERROR: no calibration row for task ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
fi
IFS=$'\t' read -r TASK_ID KEY TRAINER DESCRIPTION <<< "${ROW}"

export nnssl_raw="${NNSSL_RAW}"
export nnssl_preprocessed="${NNSSL_PREPROCESSED}"
export nnssl_results="${NNSSL_RESULTS}"
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export nnUNet_compile=false
export nnUNet_n_proc_DA=${NNUNET_N_PROC_DA:-0}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONHASHSEED="${EXPERIMENT_SEED}"
export NNSSL_EXPERIMENT_SEED="${EXPERIMENT_SEED}"
mkdir -p "${CALIBRATION_DIR}"

source "${VENV}/bin/activate"
cd "${REPO}"

echo "=== Regularizer gradient calibration ==="
echo "Task/key:    ${TASK_ID}/${KEY}"
echo "Trainer:     ${TRAINER}"
echo "Batches:     ${CALIBRATION_BATCHES}"
echo "Output:      ${CALIBRATION_DIR}/${KEY}.json"

python scripts/calibrate_primus_regularizer.py measure \
    --dataset "${DATASET_ID}" \
    --configuration "${CONFIGURATION}" \
    --plans "${PLANS}" \
    --trainer "${TRAINER}" \
    --pretrained-weights "${MAE_CHECKPOINT}" \
    --batches "${CALIBRATION_BATCHES}" \
    --seed "${EXPERIMENT_SEED}" \
    --manifest "${CALIBRATION_MANIFEST}" \
    --output "${CALIBRATION_DIR}/${KEY}.json"
