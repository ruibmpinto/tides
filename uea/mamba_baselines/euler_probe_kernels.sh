#!/bin/bash
#SBATCH --job-name=m3_probe
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_4090:1
#SBATCH --time=1:00:00
#SBATCH --output=/cluster/scratch/rbarreira/ssm_input_dependent/results/probe_%j.out
# Phase 1 gate: check which mamba-ssm layers run on an RTX 4090.
#
# Submit from the repo root on Euler, optionally choosing the env:
#   sbatch mamba_baselines/euler_probe_kernels.sh
#   sbatch --export=ALL,MAMBA_ENV=mamba3_env_t29 \
#       mamba_baselines/euler_probe_kernels.sh

set -euo pipefail

PROJECT_ROOT="/cluster/scratch/rbarreira/ssm_input_dependent"
ENV_PATH="${PROJECT_ROOT}/${MAMBA_ENV:-mamba3_env}"
REPO_ROOT="${PROJECT_ROOT}/tides/uea"

# No cuda module is loaded: cuda/12.6.2 cannot be loaded alongside
# python_cuda/3.11.6 in stack/2024-06, and it is not needed. The cu126 pip
# wheels bundle their own CUDA runtime; only the node's driver is required.
module load stack/2024-06 python_cuda/3.11.6

source "${ENV_PATH}/bin/activate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "Node : $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

cd "${REPO_ROOT}"
python mamba_baselines/euler_probe_kernels.py
