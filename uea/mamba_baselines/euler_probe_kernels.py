"""Phase 1 gate: probe which mamba-ssm layers run on this GPU.

Exercises a forward and backward pass of Mamba, Mamba2, Mamba3 (SISO) and
Mamba3 (MIMO) at the EigenWorms sequence length, and reports for each
whether it runs, in which dtype, at what speed and at what peak memory.

The MIMO probe is expected to fail on Euler: its prefill kernel comes from
TileLang, which is not installed because Euler has no Hopper part. That
failure is the specification for the pure-PyTorch replacement written in
mamba3_recurrence.py; this probe records the exact error so the
replacement can be checked against a known baseline.

Functions
---------
build_mamba
    Instantiate the original Mamba (S6) layer.
build_mamba2
    Instantiate the Mamba2 layer.
build_mamba3_siso
    Instantiate the Mamba3 layer in SISO mode.
build_mamba3_mimo
    Instantiate the Mamba3 layer in MIMO mode.
report_backends
    Print whether the optional upstream kernel backends are available.
time_forward_backward
    Time forward and backward passes and record peak memory.
probe_layer
    Run one layer through the probe and collect its result.
main
    Probe every layer and print a summary table.
"""

#
#                                                                       Modules
# =============================================================================
# Standard
import time
import traceback
# Third-party
import torch

#
# =============================================================================


def build_mamba(hidden_dim, state_dim, device):
    """Instantiate the original Mamba (S6) layer.

    Parameters
    ----------
    hidden_dim : int
        Model dimension of the layer.
    state_dim : int
        SSM state dimension.
    device : torch.device
        Device to place the layer on.

    Returns
    -------
    layer : torch.nn.Module
        The instantiated layer.

    Notes
    -----
    d_conv and expand match the published EigenWorms Mamba baseline
    configuration (convdim 2, expansion 1).
    """
    from mamba_ssm.modules.mamba_simple import Mamba

    return Mamba(d_model=hidden_dim, d_state=state_dim, d_conv=2,
                 expand=1).to(device)
# -----------------------------------------------------------------------------


def build_mamba2(hidden_dim, state_dim, device):
    """Instantiate the Mamba2 layer.

    Parameters
    ----------
    hidden_dim : int
        Model dimension of the layer.
    state_dim : int
        SSM state dimension.
    device : torch.device
        Device to place the layer on.

    Returns
    -------
    layer : torch.nn.Module
        The instantiated layer.

    Notes
    -----
    Mamba2 asserts that d_inner = expand * d_model is divisible by
    headdim, so headdim is chosen as the largest power of two that
    divides d_inner and does not exceed 64.
    """
    from mamba_ssm.modules.mamba2 import Mamba2

    expand = 2
    d_inner = expand * hidden_dim
    headdim = min(64, d_inner)
    while d_inner % headdim != 0:
        headdim //= 2
    return Mamba2(d_model=hidden_dim, d_state=state_dim, d_conv=4,
                  expand=expand, headdim=headdim).to(device)
# -----------------------------------------------------------------------------


def build_mamba3_siso(hidden_dim, state_dim, device):
    """Instantiate the Mamba3 layer in SISO mode.

    Parameters
    ----------
    hidden_dim : int
        Model dimension of the layer.
    state_dim : int
        SSM state dimension.
    device : torch.device
        Device to place the layer on.

    Returns
    -------
    layer : torch.nn.Module
        The instantiated layer.

    Notes
    -----
    fuse_pregate_headwise_norm must be disabled: upstream asserts that the
    fused pre-gate path requires both MIMO and output-projection norm.
    """
    from mamba_ssm.modules.mamba3 import Mamba3

    expand = 2
    d_inner = expand * hidden_dim
    headdim = min(64, d_inner)
    while d_inner % headdim != 0:
        headdim //= 2
    return Mamba3(d_model=hidden_dim, d_state=state_dim, expand=expand,
                  headdim=headdim, is_mimo=False,
                  fuse_pregate_headwise_norm=False,
                  chunk_size=64).to(device)
# -----------------------------------------------------------------------------


def build_mamba3_mimo(hidden_dim, state_dim, device):
    """Instantiate the Mamba3 layer in MIMO mode.

    Parameters
    ----------
    hidden_dim : int
        Model dimension of the layer.
    state_dim : int
        SSM state dimension.
    device : torch.device
        Device to place the layer on.

    Returns
    -------
    layer : torch.nn.Module
        The instantiated layer.

    Notes
    -----
    chunk_size follows the upstream recommendation of 64 / mimo_rank.
    """
    from mamba_ssm.modules.mamba3 import Mamba3

    expand = 2
    mimo_rank = 4
    d_inner = expand * hidden_dim
    headdim = min(64, d_inner)
    while d_inner % headdim != 0:
        headdim //= 2
    return Mamba3(d_model=hidden_dim, d_state=state_dim, expand=expand,
                  headdim=headdim, is_mimo=True, mimo_rank=mimo_rank,
                  is_outproj_norm=True,
                  chunk_size=64 // mimo_rank).to(device)
# -----------------------------------------------------------------------------


def report_backends():
    """Print whether the optional upstream kernel backends are available.

    Notes
    -----
    Both backends are imported behind try/except upstream and fall back to
    None. None is the expected state on Euler and is what the PyTorch
    replacements are for.
    """
    import mamba_ssm.modules.mamba3 as mamba3

    for name in ('mamba3_siso_combined', 'mamba3_mimo_combined',
                 'mamba3_step_fn'):
        value = getattr(mamba3, name, None)
        status = 'available' if value is not None else 'None (absent)'
        print(f'  upstream {name:24s}: {status}')
# -----------------------------------------------------------------------------


def time_forward_backward(layer, batch_size, seq_len, hidden_dim, device,
                          dtype, n_steps):
    """Time forward and backward passes and record peak memory.

    Parameters
    ----------
    layer : torch.nn.Module
        Layer to exercise.
    batch_size : int
        Number of sequences per step.
    seq_len : int
        Sequence length.
    hidden_dim : int
        Model dimension, i.e. the input feature count.
    device : torch.device
        Device to run on.
    dtype : torch.dtype
        Input dtype.
    n_steps : int
        Number of timed steps, after one warm-up step.

    Returns
    -------
    ms_per_step : float
        Mean wall-clock milliseconds per forward+backward step.
    peak_mb : float
        Peak allocated memory in mebibytes.
    """
    inputs = torch.randn(batch_size, seq_len, hidden_dim, device=device,
                         dtype=dtype)
    torch.cuda.reset_peak_memory_stats(device)

    for _ in range(1):
        out = layer(inputs)
        out.float().pow(2).mean().backward()
    torch.cuda.synchronize(device)

    start = time.time()
    for _ in range(n_steps):
        out = layer(inputs)
        out.float().pow(2).mean().backward()
    torch.cuda.synchronize(device)
    elapsed = time.time() - start

    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return 1000.0 * elapsed / n_steps, peak_mb
# -----------------------------------------------------------------------------


def probe_layer(name, builder, hidden_dim, state_dim, batch_size, seq_len,
                device, n_steps):
    """Run one layer through the probe and collect its result.

    Parameters
    ----------
    name : str
        Label for the layer.
    builder : callable
        Function (hidden_dim, state_dim, device) -> torch.nn.Module.
    hidden_dim : int
        Model dimension.
    state_dim : int
        SSM state dimension.
    batch_size : int
        Number of sequences per step.
    seq_len : int
        Sequence length.
    device : torch.device
        Device to run on.
    n_steps : int
        Number of timed steps.

    Returns
    -------
    result : dict
        Keys: name, hidden_dim, ok, dtype, ms_per_step, peak_mb, n_params,
        error.

    Notes
    -----
    float32 is attempted first and bfloat16 second, because some of the
    Mamba-3 kernels are written for reduced precision only. The dtype that
    succeeded is reported so later comparisons are made like for like.
    """
    result = {'name': name, 'hidden_dim': hidden_dim, 'ok': False,
              'dtype': '-', 'ms_per_step': float('nan'),
              'peak_mb': float('nan'), 'n_params': 0, 'error': ''}

    for dtype in (torch.float32, torch.bfloat16):
        try:
            layer = builder(hidden_dim, state_dim, device)
            if dtype is torch.bfloat16:
                layer = layer.to(dtype)
            result['n_params'] = sum(p.numel() for p in layer.parameters())
            ms, peak = time_forward_backward(
                layer, batch_size, seq_len, hidden_dim, device, dtype,
                n_steps)
            result.update({'ok': True, 'dtype': str(dtype).split('.')[-1],
                           'ms_per_step': ms, 'peak_mb': peak, 'error': ''})
            return result
        except Exception as exc:
            result['error'] = f'{type(exc).__name__}: {exc}'
            print(f'  {name} (hidden={hidden_dim}, {dtype}) failed:')
            traceback.print_exc()
        finally:
            torch.cuda.empty_cache()

    return result
# -----------------------------------------------------------------------------


def main():
    """Probe every layer and print a summary table."""
    batch_size = 2
    seq_len = 17984
    state_dim = 64
    hidden_dims = (16, 128)
    n_steps = 3

    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA device visible. This probe must run in '
                           'a GPU job, not on the login node.')

    device = torch.device('cuda')
    print(f'torch      : {torch.__version__}')
    print(f'cuda       : {torch.version.cuda}')
    print(f'gpu        : {torch.cuda.get_device_name(device)}')
    print(f'capability : {torch.cuda.get_device_capability(device)}')
    print(f'seq_len    : {seq_len}  batch_size: {batch_size}')
    print('')
    print('Optional kernel backends:')
    report_backends()
    print('')

    builders = (('mamba', build_mamba),
                ('mamba2', build_mamba2),
                ('mamba3_siso', build_mamba3_siso),
                ('mamba3_mimo', build_mamba3_mimo))

    results = []
    for hidden_dim in hidden_dims:
        for name, builder in builders:
            print(f'--- probing {name} (hidden={hidden_dim}) ---')
            results.append(probe_layer(
                name, builder, hidden_dim, state_dim, batch_size, seq_len,
                device, n_steps))

    print('')
    print(f'{"layer":14s} {"hidden":>6s} {"ok":>5s} {"dtype":>9s} '
          f'{"ms/step":>9s} {"peak MB":>9s} {"params":>9s}')
    for r in results:
        print(f'{r["name"]:14s} {r["hidden_dim"]:6d} {str(r["ok"]):>5s} '
              f'{r["dtype"]:>9s} {r["ms_per_step"]:9.1f} '
              f'{r["peak_mb"]:9.1f} {r["n_params"]:9d}')

    print('')
    for r in results:
        if not r['ok']:
            print(f'FAILED {r["name"]} (hidden={r["hidden_dim"]}): '
                  f'{r["error"]}')


if __name__ == '__main__':
    main()
