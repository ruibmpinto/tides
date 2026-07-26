"""Mamba-1/2/3 baselines for the UEA classification pipeline.

Added to answer Reviewer 1's question about how the newer Mamba variants
perform. The models plug into the existing pipeline (uea/main.py,
uea/hypersearch.py, uea/run_top_configs.py) so that the baselines are
searched and evaluated under exactly the protocol and budget the proposed
model was.

Modules
-------
models
    Mamba-1/2/3 classifiers sharing one block structure.
mamba3_torch_kernels
    Pure-PyTorch replacement for the Mamba-3 prefill kernels.
test_mamba3_torch_kernels
    Validation gate for that replacement.
"""
