#!/bin/bash
# Optuna sweep of Mamba-1/2/3 on EigenWorms, on ETH Euler.
#
# One shared Optuna study per model, several workers per study, so trials
# are distributed and the search resumes after any interruption. Mirrors
# the pattern in Physiome-ODE/experiments/training/submit_v12_phase1.sh.
#
# Usage:
#   bash submit_mamba_eigenworms.sh [--dry-run]
#
# Mamba-1 is the control: it has a published EigenWorms entry
# (70.9 +/- 15.8, Walker et al.) obtained under a different protocol, so
# re-running it here separates an architecture effect from a protocol
# effect in the Mamba-2 and Mamba-3 columns.

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# ── Config ────────────────────────────────────────────────────────────────────
MODELS=(${SWEEP_MODELS:-mamba2 mamba3})   # override with SWEEP_MODELS="mamba3"
SWEEP_TAG="v3"                # bump for a fresh sweep; names are derived
N_WORKERS=4
# Worker index offset, so a second batch joining the same study gets
# distinct TPE sampler seeds instead of repeating 1..N.
WORKER_OFFSET="${WORKER_OFFSET:-0}"
TRIALS_PER_WORKER=25          # 4 x 25 = 100 counted trials per model
NUM_SEEDS=1                   # search on seed 42; finals use 42-46
EARLY_STOP_PATIENCE=30
N_STARTUP_TRIALS=20
DATASET=TSC_EigenWorms

PROJECT_ROOT="/cluster/scratch/rbarreira/ssm_input_dependent"
WORKDIR="${PROJECT_ROOT}/tides/uea"
ENV_PATH="${PROJECT_ROOT}/mamba3_env_t29"
DATA_DIR="${PROJECT_ROOT}/uea_data"
RESULTS_DIR="${PROJECT_ROOT}/results/mamba_eigenworms_${SWEEP_TAG}"
STUDIES_DIR="${PROJECT_ROOT}/studies/mamba_eigenworms_${SWEEP_TAG}"
LOGS_DIR="${RESULTS_DIR}/logs"

mkdir -p "$RESULTS_DIR" "$STUDIES_DIR" "$LOGS_DIR"

maybe_sbatch() {
    if [[ "$DRY_RUN" == true ]]; then
        echo "DRY-RUN: sbatch $*" >&2
        echo "0"
    else
        sbatch --parsable "$@"
    fi
}

# ── Submit ────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "Mamba-1/2/3 sweep on EigenWorms"
echo "  Sweep tag       : ${SWEEP_TAG}"
echo "  Models          : ${MODELS[*]}"
echo "  Workers/model   : ${N_WORKERS}"
echo "  Trials/worker   : ${TRIALS_PER_WORKER}"
echo "  Target/model    : $((N_WORKERS * TRIALS_PER_WORKER)) trials"
echo "  Search seed     : 42 (finals use 42-46 via run_top_configs.py)"
echo "  Data            : ${DATA_DIR} (aeon extract_path)"
echo "  W&B             : online, entity ruibmpinto-"
echo "  Dry-run         : ${DRY_RUN}"
echo "============================================================"
echo ""

TOTAL=0
for MODEL in "${MODELS[@]}"; do
    # A distinct study, project and database per model and per sweep tag,
    # so nothing is appended to an earlier search by accident.
    STUDY_NAME="sweep_${SWEEP_TAG}_${MODEL}_eigenworms"
    WANDB_PROJECT="eigenworms_${MODEL}_${SWEEP_TAG}"
    STORAGE="sqlite:///${STUDIES_DIR}/${MODEL}.db"
    JOB_IDS=()

    for WORKER in $(seq $((WORKER_OFFSET + 1)) $((WORKER_OFFSET + N_WORKERS))); do
        JOB_ID=$(maybe_sbatch \
            --job-name="sw_${MODEL}_w${WORKER}" \
            --ntasks=1 \
            --cpus-per-task=4 \
            --mem-per-cpu=4G \
            --gpus=rtx_4090:1 \
            --time=24:00:00 \
            --output="${LOGS_DIR}/${MODEL}_w${WORKER}_%j.out" \
            --export="ALL,MODEL=${MODEL},WORKER=${WORKER},ENV_PATH=${ENV_PATH},WORKDIR=${WORKDIR},DATA_DIR=${DATA_DIR},RESULTS_DIR=${RESULTS_DIR},DATASET=${DATASET},TRIALS_PER_WORKER=${TRIALS_PER_WORKER},NUM_SEEDS=${NUM_SEEDS},EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE},N_STARTUP_TRIALS=${N_STARTUP_TRIALS},STUDY_NAME=${STUDY_NAME},WANDB_PROJECT=${WANDB_PROJECT},STORAGE=${STORAGE}" \
            "${WORKDIR}/sweep_worker.sh")
        JOB_IDS+=("$JOB_ID")
        TOTAL=$((TOTAL + 1))
    done

    echo "  ${MODEL}: workers [${JOB_IDS[*]}]"
done

echo ""
echo "============================================================"
echo "Submitted ${TOTAL} workers"
echo "Studies : ${STUDIES_DIR}/<model>.db"
echo "Results : ${RESULTS_DIR}/results_<model>_w<worker>.csv"
echo "Monitor : squeue -u rbarreira -n sw_mamba_w1"
echo ""
echo "A trial at the top of the grid (hidden 128, 2 layers, batch 5,"
echo "600 epochs) can approach the 24 h limit for Mamba-3. Such a trial"
echo "is lost if the job ends, but the study is not: resubmit this script"
echo "and the workers continue from the shared SQLite study."
echo ""
echo "Finals, after the search, for each model:"
echo "  python run_top_configs.py --results_file <csv> \\"
echo "      --dataset ${DATASET} --top_k 10 --seeds 43 44 45 46 \\"
echo "      --data_dir ${DATA_DIR}"
echo "============================================================"
