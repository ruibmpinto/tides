"""Check the consolidated Mamba-3 core against its references.

Run on a GPU node:
    python mamba_baselines/test_mamba3_recurrence.py

Exits non-zero if any check fails.

C1  rank-1 core vs the proven-exact SISO transcription, float64
C2  SISO adapter vs the Triton kernel, judged against the float64answer
C3  gradcheck of the rank-4 MIMO path, float64
C4  full-scale forward and backward for both modes at L=17984

Functions
---------
relative_error
    Max absolute difference scaled by the reference magnitude.
make_inputs
    Random inputs for the rank-general core.
check_c1_rank1_vs_siso
    Compare the rank-1 core with the SISO transcription.
check_c2_adapter_vs_triton
    Compare the SISO adapter with the Triton kernel.
check_c3_gradcheck_mimo
    Run gradcheck on the rank-4 MIMO path.
check_c4_full_scale
    Forward and backward at the EigenWorms length, both modes.
main
    Run the checks and summarise.
"""

#
#                                                                       Modules
# =============================================================================
# Standard
import os
import sys
import time
import traceback
# Third-party
import torch
# Local
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mamba3_recurrence import (install_torch_backend, mamba3_recurrence,
                               mamba3_siso_combined_torch)
from mamba3_siso_recurrence import mamba3_siso_recurrence

#
# =============================================================================


def relative_error(actual, reference):
    """Max absolute difference scaled by the reference magnitude.

    Parameters
    ----------
    actual : torch.Tensor
        Tensor under test.
    reference : torch.Tensor
        Reference tensor.

    Returns
    -------
    error : float
        Maximum absolute difference over the maximum reference value.
    """
    scale = reference.abs().max().clamp(min=1e-12)
    return float((actual - reference).abs().max() / scale)
# -----------------------------------------------------------------------------


def make_inputs(batch, seqlen, rank, nheads, n_state, headdim, n_angles,
                device, dtype, seed=0, with_mimo=False, with_out=False):
    """Random inputs for the rank-general core.

    Parameters
    ----------
    batch, seqlen, rank, nheads, n_state, headdim, n_angles : int
        Shape parameters.
    device : torch.device
        Device to allocate on.
    dtype : torch.dtype
        Dtype for the tensors.
    seed : int, default=0
        RNG seed.
    with_mimo : bool, default=False
        Include the MIMO up-projections.
    with_out : bool, default=False
        Include the MIMO down-projection.

    Returns
    -------
    inputs : dict
        Keyword arguments for mamba3_recurrence.
    """
    generator = torch.Generator(device='cpu').manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=generator,
                           dtype=torch.float64).to(device=device,
                                                   dtype=dtype)

    dt = (torch.rand(batch, nheads, seqlen, generator=generator,
                     dtype=torch.float64) * 0.1 + 0.01).to(
                         device=device, dtype=dtype)
    a_neg = (-torch.rand(batch, nheads, seqlen, generator=generator,
                         dtype=torch.float64) - 0.5).to(device=device,
                                                        dtype=dtype)

    return {
        'Q': randn(batch, seqlen, rank, nheads, n_state),
        'K': randn(batch, seqlen, rank, nheads, n_state),
        'V': randn(batch, seqlen, nheads, headdim),
        'ADT': a_neg * dt,
        'DT': dt,
        'Trap': randn(batch, nheads, seqlen),
        'Q_bias': randn(nheads, rank, n_state),
        'K_bias': randn(nheads, rank, n_state),
        'Angles': randn(batch, seqlen, nheads, n_angles),
        'MIMO_V': randn(nheads, rank, headdim) if with_mimo else None,
        'MIMO_Z': randn(nheads, rank, headdim) if with_mimo else None,
        'MIMO_Out': randn(nheads, rank, headdim) if with_out else None,
        'D': randn(nheads),
        'Z': randn(batch, seqlen, nheads, headdim),
    }
# -----------------------------------------------------------------------------


def check_c1_rank1_vs_siso(device):
    """Compare the rank-1 core with the SISO transcription.

    Parameters
    ----------
    device : torch.device
        Device to run on.

    Returns
    -------
    ok : bool
        True if the two agree to float64 precision.

    Notes
    -----
    This is what makes the consolidation safe: the rank-general core must
    reduce exactly to the transcription already proven exact against the
    upstream kernel, so generalising did not perturb the arithmetic.
    """
    ok = True
    for chunk_size in (8, 16, 32):
        inputs = make_inputs(2, 64, 1, 2, 32, 16, 8, device,
                             torch.float64, seed=21)
        core = mamba3_recurrence(chunk_size=chunk_size, **inputs)
        siso = mamba3_siso_recurrence(
            Q=inputs['Q'].squeeze(2), K=inputs['K'].squeeze(2),
            V=inputs['V'], ADT=inputs['ADT'], DT=inputs['DT'],
            Trap=inputs['Trap'], Q_bias=inputs['Q_bias'].squeeze(1),
            K_bias=inputs['K_bias'].squeeze(1), Angles=inputs['Angles'],
            D=inputs['D'], Z=inputs['Z'], chunk_size=chunk_size)
        error = relative_error(core, siso)
        print(f'  chunk_{chunk_size:<3d} rel_err={error:.3e}')
        ok = ok and error < 1e-12
    return ok
# -----------------------------------------------------------------------------


def check_c2_adapter_vs_triton(device):
    """Compare the SISO adapter with the Triton kernel.

    Parameters
    ----------
    device : torch.device
        Device to run on.

    Returns
    -------
    ok : bool
        True if our result is at float32 machine precision from the exact
        answer and the gap to the kernel is attributable to the kernel.
    """
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import \
        mamba3_siso_combined

    inputs = make_inputs(2, 512, 1, 2, 64, 64, 16, device,
                         torch.float32, seed=22)
    squeezed = {
        'Q': inputs['Q'].squeeze(2).contiguous(),
        'K': inputs['K'].squeeze(2).contiguous(),
        'V': inputs['V'].contiguous(), 'ADT': inputs['ADT'].contiguous(),
        'DT': inputs['DT'].contiguous(),
        'Trap': inputs['Trap'].contiguous(),
        'Q_bias': inputs['Q_bias'].squeeze(1).contiguous(),
        'K_bias': inputs['K_bias'].squeeze(1).contiguous(),
        'Angles': inputs['Angles'].contiguous(),
        'D': inputs['D'].contiguous(), 'Z': inputs['Z'].contiguous(),
    }

    mine = mamba3_siso_combined_torch(chunk_size=64, **squeezed)
    theirs = mamba3_siso_combined(chunk_size=64, Input_States=None,
                                  return_final_states=False,
                                  cu_seqlens=None, **squeezed)
    if isinstance(theirs, tuple):
        theirs = theirs[0]

    exact = mamba3_siso_combined_torch(
        chunk_size=64,
        **{k: (v.double() if torch.is_tensor(v) else v)
           for k, v in squeezed.items()})

    mine_err = relative_error(mine.double(), exact)
    theirs_err = relative_error(theirs.double(), exact)
    gap = relative_error(mine, theirs)
    print(f'  mine_vs_triton={gap:.3e}  mine_vs_fp64={mine_err:.3e}  '
          f'triton_vs_fp64={theirs_err:.3e}')
    return mine_err < 1e-6 and gap <= 2.0 * theirs_err + 1e-6
# -----------------------------------------------------------------------------


def check_c3_gradcheck_mimo(device):
    """Run gradcheck on the rank-4 MIMO path.

    Parameters
    ----------
    device : torch.device
        Device to run on.

    Returns
    -------
    ok : bool
        True if gradcheck passes.
    """
    inputs = make_inputs(1, 16, 4, 2, 16, 8, 4, device, torch.float64,
                         seed=23, with_mimo=True, with_out=True)
    names = ('Q', 'K', 'V', 'ADT', 'DT', 'Trap', 'Angles', 'Z',
             'MIMO_V', 'MIMO_Z', 'MIMO_Out')
    tensors = [inputs[name].detach().clone().requires_grad_(True)
               for name in names]

    def function(*args):
        kwargs = dict(inputs)
        for name, tensor in zip(names, args):
            kwargs[name] = tensor
        return mamba3_recurrence(chunk_size=8, **kwargs)

    ok = torch.autograd.gradcheck(function, tuple(tensors), eps=1e-6,
                                  atol=1e-7, rtol=1e-4, nondet_tol=0.0)
    print(f'  gradcheck {"passed" if ok else "failed"} for '
          f'{len(names)} inputs')
    return bool(ok)
# -----------------------------------------------------------------------------


def check_c4_full_scale(device):
    """Forward and backward at the EigenWorms length, both modes.

    Parameters
    ----------
    device : torch.device
        Device to run on.

    Returns
    -------
    ok : bool
        True if both modes give finite losses and gradients.
    """
    from mamba_ssm.modules.mamba3 import Mamba3

    install_torch_backend(verbose=False, siso=True)
    batch, seqlen, hidden = 2, 17984, 64
    ok = True

    for label, kwargs in (
            ('siso', {'is_mimo': False,
                      'fuse_pregate_headwise_norm': False,
                      'chunk_size': 64}),
            ('mimo', {'is_mimo': True, 'mimo_rank': 4,
                      'is_outproj_norm': True,
                      'fuse_pregate_headwise_norm': True,
                      'chunk_size': 16})):
        try:
            torch.cuda.reset_peak_memory_stats(device)
            layer = Mamba3(d_model=hidden, d_state=64, expand=2,
                           headdim=64, **kwargs).to(device)
            x = torch.randn(batch, seqlen, hidden, device=device)
            start = time.time()
            out = layer(x)
            loss = out.float().pow(2).mean()
            loss.backward()
            torch.cuda.synchronize(device)
            elapsed = time.time() - start
            peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            grads = [p.grad for p in layer.parameters()
                     if p.grad is not None]
            finite = (torch.isfinite(loss)
                      and all(torch.isfinite(g).all() for g in grads))
            print(f'  {label:5s} loss={loss.item():.4e} '
                  f'finite={bool(finite)} {elapsed * 1000:.0f} ms '
                  f'peak={peak:.0f} MB grads={len(grads)}')
            ok = ok and bool(finite)
        except Exception as exc:
            print(f'  {label:5s} FAILED: {type(exc).__name__}: {exc}')
            traceback.print_exc()
            ok = False
        finally:
            torch.cuda.empty_cache()
    return ok
# -----------------------------------------------------------------------------


def main():
    """Run the checks and summarise."""
    if not torch.cuda.is_available():
        raise RuntimeError('These checks need a GPU. Submit as a job.')

    device = torch.device('cuda')
    print(f'torch {torch.__version__} on '
          f'{torch.cuda.get_device_name(device)}\n')

    checks = (('C1 rank-1 core vs SISO transcription',
               lambda: check_c1_rank1_vs_siso(device)),
              ('C2 SISO adapter vs Triton',
               lambda: check_c2_adapter_vs_triton(device)),
              ('C3 gradcheck rank-4 MIMO',
               lambda: check_c3_gradcheck_mimo(device)),
              ('C4 full scale L=17984',
               lambda: check_c4_full_scale(device)))

    results = {}
    for name, check in checks:
        print(f'--- {name} ---')
        try:
            results[name] = bool(check())
        except Exception as exc:
            print(f'  ERROR {type(exc).__name__}: {exc}')
            traceback.print_exc()
            results[name] = False
        print(f'  => {"PASS" if results[name] else "FAIL"}\n')

    print('=== summary ===')
    for name in results:
        print(f'{"PASS" if results[name] else "FAIL"}  {name}')
    if not all(results.values()):
        sys.exit(1)


if __name__ == '__main__':
    main()
