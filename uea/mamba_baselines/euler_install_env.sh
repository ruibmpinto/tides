#!/bin/bash
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
# Install a mamba2/mamba3 venv on the Euler SLURM cluster.
# Run interactively from the repo root (not submitted as a job):
#   bash mamba_baselines/euler_install_env.sh [t27|t29]
#
# Two rungs of the install ladder:
#   t27  torch 2.7.0 + triton 3.3. Mamba and Mamba2 run. Mamba3 does NOT:
#        its Triton kernel calls tl.make_tensor_descriptor, absent before
#        triton 3.4 (measured: probe job 8620743).
#   t29  torch 2.9.0 + triton 3.5, which has that API. Used to determine
#        whether Mamba3's Triton kernel is merely version-blocked or
#        genuinely Hopper-bound (TMA is Hopper hardware; this GPU is sm89).

module load stack/2024-06 python_cuda/3.11.6
module load eth_proxy

set -euo pipefail

PROJECT_ROOT="/cluster/scratch/rbarreira/ssm_input_dependent"
STACK="${1:-t27}"

# Prebuilt wheels are pinned to a measured-good Euler stack: python 3.11.6
# and CXX11 ABI True (as in env_trm). Matching the wheel variant exactly
# avoids a CUDA source build.
MAMBA_RELEASE="https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1"

case "${STACK}" in
    t27)
        ENV_PATH="${PROJECT_ROOT}/mamba3_env"
        TORCH_SPEC="torch==2.7.0"
        TORCH_EXPECTED="2.7.0"
        MAMBA_WHEEL="mamba_ssm-2.3.2.post1+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
        CONV_RELEASE="https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.4"
        CONV_WHEEL="causal_conv1d-1.5.4+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
        ;;
    t29)
        ENV_PATH="${PROJECT_ROOT}/mamba3_env_t29"
        TORCH_SPEC="torch==2.9.0"
        TORCH_EXPECTED="2.9.0"
        MAMBA_WHEEL="mamba_ssm-2.3.2.post1+cu12torch2.9cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
        CONV_RELEASE="https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.2.post1"
        CONV_WHEEL="causal_conv1d-1.6.2.post1+cu12torch2.9cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
        ;;
    *)
        echo "Unknown stack '${STACK}'. Use t27 or t29." >&2
        exit 1
        ;;
esac

echo ">>> Stack ${STACK}: ${TORCH_SPEC} -> ${ENV_PATH}"

# -----------------------------------------------------------------------------
# 1. Create venv (skip if already exists)
# -----------------------------------------------------------------------------
if [ -d "$ENV_PATH" ]; then
    echo ">>> Environment already exists at $ENV_PATH — skipping creation."
    echo "    Delete it first if you want a clean install: rm -rf $ENV_PATH"
else
    echo ">>> Creating venv at $ENV_PATH ..."
    python -m venv "$ENV_PATH"
fi

source "$ENV_PATH/bin/activate"

# -----------------------------------------------------------------------------
# 2. Bootstrap pip and upgrade core build tooling
# -----------------------------------------------------------------------------
python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel

# -----------------------------------------------------------------------------
# 3. PyTorch
# -----------------------------------------------------------------------------
# Pin torch to CUDA 12.6 wheels to match the cluster's cuda/12.6.2 module.
# The default index ships +cu130 wheels, which require
# libnvrtc-builtins.so.13.0 and crash at JIT-compile time. Version 2.7.0 is
# also what the prebuilt mamba-ssm wheel below was compiled against.
echo ">>> Installing ${TORCH_SPEC} (cu126)..."
pip install --index-url https://download.pytorch.org/whl/cu126 \
    "${TORCH_SPEC}"

# -----------------------------------------------------------------------------
# 4. Verify the ABI/CUDA assumptions the prebuilt wheels depend on
# -----------------------------------------------------------------------------
# Fail here rather than at import time: an ABI or torch-version mismatch
# makes the mamba-ssm wheel load a broken C extension.
echo ">>> Checking stack against the prebuilt wheel variant..."
TORCH_EXPECTED="${TORCH_EXPECTED}" python - <<'EOF'
import os
import sys

import torch

expected_torch = os.environ['TORCH_EXPECTED']
actual = (torch.__version__.split('+')[0], torch.version.cuda,
          torch._C._GLIBCXX_USE_CXX11_ABI)
expected = (expected_torch, '12.6', True)
try:
    import triton
    triton_version = triton.__version__
except Exception as exc:
    triton_version = f'unavailable ({exc})'
print(f'python={sys.version.split()[0]}  torch={torch.__version__}  '
      f'cuda={torch.version.cuda}  '
      f'abi={torch._C._GLIBCXX_USE_CXX11_ABI}  '
      f'triton={triton_version}')
if actual != expected:
    raise SystemExit(
        f'Mismatch: got {actual}, expected {expected}. The prebuilt '
        f'wheels for this stack will not load; pick a different wheel '
        f'variant before continuing.'
    )
print('stack matches the prebuilt wheel variant')
EOF

# -----------------------------------------------------------------------------
# 5. Dependencies of mamba-ssm, pinned by hand
# -----------------------------------------------------------------------------
# mamba-ssm 2.3.2.post1 declares triton>=3.5.0 (a torch 2.9-era pin) while
# publishing wheels for torch 2.6/2.7, and declares tilelang + quack-kernels
# for its MIMO and Hopper decode kernels. Installing with --no-deps below
# avoids that conflict; the triton shipped with torch 2.7 is the one the
# wheel was built against.
#
# tilelang and quack-kernels are deliberately omitted: Euler has no Hopper
# part, so the MIMO kernel is replaced by a pure-PyTorch implementation in
# mamba_baselines/mamba3_recurrence.py.
echo ">>> Installing dependencies..."
pip install \
    einops==0.8.1 \
    transformers==4.53.2 \
    ninja \
    packaging

# jax is needed only to unpickle the processed UEA arrays, which were
# saved as jax device arrays. CPU build is sufficient.
pip install "jax[cpu]==0.4.30"

# -----------------------------------------------------------------------------
# 6. mamba-ssm and causal-conv1d (prebuilt, --no-deps)
# -----------------------------------------------------------------------------
echo ">>> Installing causal-conv1d..."
pip install --no-deps "${CONV_RELEASE}/${CONV_WHEEL}"

echo ">>> Installing mamba-ssm 2.3.2.post1..."
pip install --no-deps "${MAMBA_RELEASE}/${MAMBA_WHEEL}"

# -----------------------------------------------------------------------------
# 7. Verify what can be verified without a GPU
# -----------------------------------------------------------------------------
# `import mamba_ssm` cannot be checked here: its import chain reaches
# mamba_ssm.ops.triton.layer_norm, whose @triton.autotune decorator runs at
# import time and needs an active GPU driver. On the login node that raises
# "0 active drivers". The import check therefore lives in the GPU probe
# (mamba_baselines/euler_probe_kernels.py), not in this script.
echo ">>> Verifying installed distributions (no GPU needed)..."
python - <<'EOF'
from importlib.metadata import version

for dist in ('torch', 'mamba_ssm', 'causal_conv1d', 'einops', 'jax'):
    try:
        print(f'{dist}={version(dist)}')
    except Exception as exc:
        raise SystemExit(f'{dist} is not installed: {exc}')
EOF

echo ">>> Checking that the EigenWorms pickles load..."
python - <<'EOF'
import pickle

import numpy as np

base = ('/cluster/scratch/rbarreira/ssm_input_dependent/data_dir'
        '/processed/UEA/EigenWorms')
for name in ('labels.pkl', 'data.pkl'):
    with open(f'{base}/{name}', 'rb') as f:
        obj = pickle.load(f)
    arr = np.array(obj)
    print(f'{name}: {type(obj).__name__} {arr.shape} {arr.dtype}')
EOF

echo ""
echo "Done. Environment installed at $ENV_PATH"
echo "Activate with:"
echo "  module load stack/2024-06 python_cuda/3.11.6 cuda/12.6.2 eth_proxy"
echo "  source $ENV_PATH/bin/activate"
echo ""
echo "Next (Phase 1 gate, needs a GPU):"
echo "  python mamba_baselines/test_mamba3_recurrence.py"
