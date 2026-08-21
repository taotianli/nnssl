#!/bin/bash
#SBATCH --job-name=om-jepa-reg
#SBATCH --output=/lus/lfs1aip2/projects/u6mn/openmind_jepa/nnssl_results/om_jepa_reg_%A_%a.out
#SBATCH --error=/lus/lfs1aip2/projects/u6mn/openmind_jepa/nnssl_results/om_jepa_reg_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --array=0-6
#SBATCH --signal=USR1@600

set -euo pipefail

REPO=${REPO:-/lus/lfs1aip2/projects/u6mn/nnssl}
VENV=${VENV:-/home/u6mn/taotl.u6mn/vllm_tutorial/.venv}
NNSSL_RAW=${NNSSL_RAW:-/lus/lfs1aip2/projects/u6mn/openmind_jepa/nnssl_raw}
NNSSL_PREPROCESSED=${NNSSL_PREPROCESSED:-/lus/lfs1aip2/projects/u6mn/openmind_jepa/nnssl_preprocessed}
NNSSL_RESULTS=${NNSSL_RESULTS:-/lus/lfs1aip2/projects/u6mn/openmind_jepa/nnssl_results}
MAE_CHECKPOINT=${MAE_CHECKPOINT:-/lus/lfs1aip2/projects/u6mn/openmind_jepa/weights/PrimusM-OpenMind-MAE/checkpoint_final.pth}
SWEEP_MANIFEST=${SWEEP_MANIFEST:-${REPO}/scripts/configs/openmind_mae_jepa_regularization_sweep.tsv}
CALIBRATION_DIR=${CALIBRATION_DIR:-/lus/lfs1aip2/projects/u6mn/openmind_jepa/regularizer_calibration}

DATASET_ID=${DATASET_ID:-746}
CONFIGURATION=${CONFIGURATION:-onemmiso}
PLANS=${PLANS:-nnsslPlans}
EXPERIMENT_SEED=${EXPERIMENT_SEED:-20260821}

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: submit this script with sbatch --array=<ids>" >&2
    exit 2
fi
if [[ ! -f "${SWEEP_MANIFEST}" ]]; then
    echo "ERROR: sweep manifest not found: ${SWEEP_MANIFEST}" >&2
    exit 2
fi

ROW=$(awk -F '\t' -v id="${SLURM_ARRAY_TASK_ID}" '$1 == id {print; exit}' "${SWEEP_MANIFEST}")
if [[ -z "${ROW}" ]]; then
    echo "ERROR: no manifest row for task ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
fi
IFS=$'\t' read -r TASK_ID BLOCK TRAINER CALIBRATION_KEY RHO_MULTIPLIER DESCRIPTION <<< "${ROW}"

export nnssl_raw="${NNSSL_RAW}"
export nnssl_preprocessed="${NNSSL_PREPROCESSED}"
export nnssl_results="${NNSSL_RESULTS}"
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export nnUNet_compile=false
export nnUNet_n_proc_DA=${NNUNET_N_PROC_DA:-0}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_MODE=${WANDB_MODE:-offline}
export PYTHONHASHSEED="${EXPERIMENT_SEED}"
export NNSSL_EXPERIMENT_SEED="${EXPERIMENT_SEED}"

TASK_TMPDIR=/tmp/openmind_jepa_reg_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export TMPDIR="${TASK_TMPDIR}"
mkdir -p "${TASK_TMPDIR}"
trap 'rm -rf "${TASK_TMPDIR}"' EXIT

source "${VENV}/bin/activate"
cd "${REPO}"

CALIBRATION_STATUS="no regularizer"
APPLIED_CALIBRATION_JSON=""
if [[ "${CALIBRATION_KEY}" != "-" ]]; then
    REFERENCE_JSON="${CALIBRATION_DIR}/vic.json"
    CANDIDATE_JSON="${CALIBRATION_DIR}/${CALIBRATION_KEY}.json"
    if [[ ! -f "${REFERENCE_JSON}" || ! -f "${CANDIDATE_JSON}" ]]; then
        echo "ERROR: calibrated run requires ${REFERENCE_JSON} and ${CANDIDATE_JSON}" >&2
        echo "Run scripts/slurm/openmind_calibrate_mae_jepa_regularizers.slurm first." >&2
        exit 2
    fi
    APPLIED_CALIBRATION_JSON="${TASK_TMPDIR}/regularizer_calibration_applied.json"
    python scripts/calibrate_primus_regularizer.py derive \
        --reference-json "${REFERENCE_JSON}" \
        --candidate-json "${CANDIDATE_JSON}" \
        --rho-multiplier "${RHO_MULTIPLIER}" \
        --json > "${APPLIED_CALIBRATION_JSON}"
    NNSSL_REGULARIZER_WEIGHT=$(python -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["recommended_lambda"])' \
        "${APPLIED_CALIBRATION_JSON}")
    export NNSSL_REGULARIZER_WEIGHT
    CALIBRATION_STATUS="${CALIBRATION_KEY}, rho multiplier ${RHO_MULTIPLIER}, lambda ${NNSSL_REGULARIZER_WEIGHT}"
fi

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
mkdir -p "${OUTPUT_DIR}"
if [[ -n "${APPLIED_CALIBRATION_JSON}" ]]; then
    OUTPUT_CALIBRATION_JSON="${OUTPUT_DIR}/regularizer_calibration_applied.json"
    if [[ -f "${OUTPUT_CALIBRATION_JSON}" ]] && \
       ! cmp -s "${APPLIED_CALIBRATION_JSON}" "${OUTPUT_CALIBRATION_JSON}"; then
        echo "ERROR: calibration metadata changed for an existing output directory" >&2
        exit 2
    fi
    cp "${APPLIED_CALIBRATION_JSON}" "${OUTPUT_CALIBRATION_JSON}"
fi
LOAD_ARGS=(-pretrained_weights "${MAE_CHECKPOINT}")
if [[ -f "${OUTPUT_DIR}/checkpoint_final.pth" || \
      -f "${OUTPUT_DIR}/checkpoint_latest.pth" || \
      -f "${OUTPUT_DIR}/checkpoint_best.pth" ]]; then
    LOAD_ARGS=(--c)
fi

echo "=== OpenMind regularization sweep ==="
echo "Array/task:     ${SLURM_ARRAY_JOB_ID}/${TASK_ID}"
echo "Block:          ${BLOCK}"
echo "Trainer:        ${TRAINER}"
echo "Description:    ${DESCRIPTION}"
echo "Calibration:    ${CALIBRATION_STATUS}"
echo "Seed:           ${EXPERIMENT_SEED}"
echo "Node/GPU:       $(hostname) / $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Dataset:        ${DATASET_NAME}"
echo "MAE checkpoint: ${MAE_CHECKPOINT}"
echo "Output:         ${OUTPUT_DIR}"
echo "Load arguments: ${LOAD_ARGS[*]}"
echo "Started:        $(date --iso-8601=seconds)"

python -m nnssl.run.run_training \
    "${DATASET_ID}" "${CONFIGURATION}" \
    -tr "${TRAINER}" \
    -p "${PLANS}" \
    "${LOAD_ARGS[@]}"

echo "Completed: $(date --iso-8601=seconds)"
