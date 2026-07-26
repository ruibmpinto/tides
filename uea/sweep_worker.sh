#!/bin/bash
# One Optuna worker for the EigenWorms Mamba sweep.
#
# Submitted by submit_mamba_eigenworms.sh, which passes every setting
# through --export. Kept as a real script rather than an sbatch --wrap
# body because --wrap runs under a non-login /bin/sh, where `module` is
# not defined and `set -o pipefail` is invalid.
#
# Expected in the environment: MODEL, WORKER, ENV_PATH, WORKDIR, DATA_DIR,
# RESULTS_DIR, DATASET, TRIALS_PER_WORKER, NUM_SEEDS,
# EARLY_STOP_PATIENCE, N_STARTUP_TRIALS, STUDY_NAME, WANDB_PROJECT,
# STORAGE.

set -eu

module load stack/2024-06 python_cuda/3.11.6 eth_proxy

source "${ENV_PATH}/bin/activate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Online logging. hypersearch.py authenticates with wandb.login(), which
# reads WANDB_API_KEY; verified on a compute node that runs then land
# under entity ruibmpinto-. eth_proxy above provides the route to
# api.wandb.ai.
export WANDB_API_KEY=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
export WANDB_MODE=online
export WANDB_DIR="${RESULTS_DIR}"
export WANDB_CACHE_DIR="${RESULTS_DIR}/.wandb_cache"

echo "node   : $(hostname)"
echo "model  : ${MODEL}  worker: ${WORKER}"
echo "study  : ${STUDY_NAME}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo ""

# Stagger worker starts so they do not race to create the study.
sleep $(( WORKER * 5 ))

cd "${WORKDIR}"
python hypersearch.py \
    --model "${MODEL}" \
    --datasets "${DATASET}" \
    --data_dir "${DATA_DIR}" \
    --num_trials "${TRIALS_PER_WORKER}" \
    --num_seeds "${NUM_SEEDS}" \
    --no_random_drop \
    --early_stop_patience "${EARLY_STOP_PATIENCE}" \
    --n_startup_trials "${N_STARTUP_TRIALS}" \
    --results_file "${RESULTS_DIR}/results_${MODEL}_w${WORKER}.csv" \
    --study_name "${STUDY_NAME}" \
    --storage "${STORAGE}" \
    --wandb_project "${WANDB_PROJECT}" \
    --seed "${WORKER}"
