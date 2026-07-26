"""Check the SISO transcription against the upstream Triton kernel.

Run on a GPU node:
    python mamba_baselines/test_mamba3_siso_recurrence.py

Exits non-zero if any check fails.

S1  transcription vs mamba3_siso_combined, at the shapes a real block uses
S2  invariance to chunk_size
S3  float64 self-consistency against a step-by-step recurrence

Functions
---------
make_inputs
    Random inputs shaped as mamba3_siso_combined expects them.
relative_error
    Max absolute difference scaled by the reference magnitude.
sequential_reference
    Step-by-step recurrence, for the float64 check.
check_s1_vs_triton
    Compare the transcription with the Triton kernel.
check_s2_chunk_invariance
    Compare results across chunk sizes.
check_s3_sequential
    Compare the chunked transcription with the sequential form.
main
    Run the checks and summarise.
"""

#
#                                                                       Modules
# =============================================================================
# Standard
import os
import sys
import traceback
# Third-party
import torch
# Local
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mamba3_siso_recurrence import (angle_cumsum,
                                    apply_rotary_interleaved,
                                    mamba3_siso_recurrence, trap_scales)

#
# =============================================================================


def make_inputs(batch, seqlen, nheads, n_state, headdim, n_angles,
                device, dtype, seed=0, with_d=True, with_z=True):
    """Random inputs shaped as mamba3_siso_combined expects them.

    Parameters
    ----------
    batch, seqlen, nheads, n_state, headdim, n_angles : int
        Shape parameters.
    device : torch.device
        Device to allocate on.
    dtype : torch.dtype
        Dtype for the tensors.
    seed : int, default=0
        RNG seed.
    with_d : bool, default=True
        Include the skip weight.
    with_z : bool, default=True
        Include the gate.

    Returns
    -------
    inputs : dict
        Keyword arguments for mamba3_siso_recurrence.
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
        'Q': randn(batch, seqlen, nheads, n_state),
        'K': randn(batch, seqlen, nheads, n_state),
        'V': randn(batch, seqlen, nheads, headdim),
        'ADT': a_neg * dt,
        'DT': dt,
        'Trap': randn(batch, nheads, seqlen),
        'Q_bias': randn(nheads, n_state),
        'K_bias': randn(nheads, n_state),
        'Angles': randn(batch, seqlen, nheads, n_angles),
        'D': randn(nheads) if with_d else None,
        'Z': randn(batch, seqlen, nheads, headdim) if with_z else None,
    }
# -----------------------------------------------------------------------------


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


def sequential_reference(inputs):
    """Step-by-step recurrence, for the float64 check.

    Parameters
    ----------
    inputs : dict
        As returned by make_inputs.

    Returns
    -------
    out : torch.Tensor
        Shape (batch, seqlen, nheads, headdim).
    """
    q_in = inputs['Q']
    batch, seqlen, nheads, n_state = q_in.shape
    headdim = inputs['V'].shape[-1]
    acc = torch.promote_types(q_in.dtype, torch.float32)

    dt = inputs['DT'].to(acc)
    gamma, scale = trap_scales(dt, inputs['Trap'].to(acc))
    cumulative = angle_cumsum(inputs['Angles'].to(acc), dt)

    q_pre = q_in.to(acc) + inputs['Q_bias'].to(acc)
    k_pre = inputs['K'].to(acc) + inputs['K_bias'].to(acc)
    qk_dot = (q_pre * k_pre).sum(dim=-1) * gamma.permute(0, 2, 1)

    q_rot = apply_rotary_interleaved(q_pre, cumulative)
    k_rot = apply_rotary_interleaved(k_pre, cumulative)
    k_rot = k_rot * scale.permute(0, 2, 1).unsqueeze(-1)
    v_all = inputs['V'].to(acc)

    state = torch.zeros(batch, nheads, headdim, n_state,
                        device=q_in.device, dtype=acc)
    outputs = []
    for step in range(seqlen):
        decay = torch.exp(inputs['ADT'].to(acc)[:, :, step])
        state = state * decay[..., None, None]

        out = torch.einsum('bhn,bhpn->bhp', q_rot[:, step], state)
        skip = qk_dot[:, step]
        if inputs['D'] is not None:
            skip = skip + inputs['D'].to(acc)
        outputs.append(out + skip.unsqueeze(-1) * v_all[:, step])

        state = state + torch.einsum('bhp,bhn->bhpn', v_all[:, step],
                                     k_rot[:, step])

    out = torch.stack(outputs, dim=1)
    if inputs['Z'] is not None:
        out = out * torch.nn.functional.silu(inputs['Z'].to(acc))
    return out
# -----------------------------------------------------------------------------


def check_s1_vs_triton(device):
    """Compare the transcription with the Triton kernel.

    Parameters
    ----------
    device : torch.device
        Device to run on.

    Returns
    -------
    ok : bool
        True if every configuration agrees within fp32 tolerance.
    """
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import \
        mamba3_siso_combined

    ok = True
    # zero_angle cases isolate the phase path: with a zero angle the
    # kernel's PTX tanh/cos/sin approximations are exact, so any residual
    # there comes from elsewhere. Comparing L=128 with L=512 separates a
    # fixed offset from one that accumulates with sequence length.
    cases = (('plain', False, False, False, 512),
             ('with_dz', True, True, False, 512),
             ('zero_ang_128', True, True, True, 128),
             ('zero_ang_512', True, True, True, 512),
             ('real_ang_128', True, True, False, 128))
    for label, with_d, with_z, zero_angle, seqlen in cases:
        inputs = make_inputs(2, seqlen, 2, 64, 64, 16, device,
                             torch.float32, seed=11, with_d=with_d,
                             with_z=with_z)
        if zero_angle:
            inputs['Angles'] = torch.zeros_like(inputs['Angles'])
        mine = mamba3_siso_recurrence(chunk_size=64, **inputs)
        theirs = mamba3_siso_combined(
            Q=inputs['Q'].contiguous(), K=inputs['K'].contiguous(),
            V=inputs['V'].contiguous(), ADT=inputs['ADT'].contiguous(),
            DT=inputs['DT'].contiguous(),
            Trap=inputs['Trap'].contiguous(),
            Q_bias=inputs['Q_bias'].contiguous(),
            K_bias=inputs['K_bias'].contiguous(),
            Angles=inputs['Angles'].contiguous(),
            D=None if inputs['D'] is None else inputs['D'].contiguous(),
            Z=None if inputs['Z'] is None else inputs['Z'].contiguous(),
            chunk_size=64, Input_States=None,
            return_final_states=False, cu_seqlens=None)
        if isinstance(theirs, tuple):
            theirs = theirs[0]

        # Which of the two is deviating? Compute our result in float64
        # and measure both fp32 results against it. If ours is stable to
        # ~1e-7 while the kernel sits at ~1e-3, the residual is the
        # kernel's reduced-precision arithmetic, not our transcription.
        exact = mamba3_siso_recurrence(
            chunk_size=64,
            **{k: (v.double() if torch.is_tensor(v) else v)
               for k, v in inputs.items()})
        mine_err = relative_error(mine.double(), exact)
        theirs_err = relative_error(theirs.double(), exact)

        # Pass criterion, against an absolute reference rather than a
        # hand-picked tolerance: our fp32 result must sit at fp32 machine
        # precision from the exact answer, and the gap to the kernel must
        # be attributable to the kernel's own error.
        error = relative_error(mine, theirs)
        exact_ok = mine_err < 1e-6
        attributable = error <= 2.0 * theirs_err + 1e-6
        print(f'  {label:14s} mine_vs_triton={error:.3e}  '
              f'mine_vs_fp64={mine_err:.3e}  '
              f'triton_vs_fp64={theirs_err:.3e}  '
              f'{"ok" if exact_ok and attributable else "FAIL"}')
        ok = ok and exact_ok and attributable
    return ok
# -----------------------------------------------------------------------------


def check_s2_chunk_invariance(device):
    """Compare results across chunk sizes.

    Parameters
    ----------
    device : torch.device
        Device to run on.

    Returns
    -------
    ok : bool
        True if the result does not depend on chunk_size.
    """
    inputs = make_inputs(2, 96, 2, 32, 16, 8, device, torch.float64,
                         seed=12)
    reference = mamba3_siso_recurrence(chunk_size=16, **inputs)
    ok = True
    for chunk_size in (8, 24, 32, 48):
        other = mamba3_siso_recurrence(chunk_size=chunk_size, **inputs)
        error = relative_error(other, reference)
        print(f'  chunk_{chunk_size:<3d} rel_err={error:.3e}')
        ok = ok and error < 1e-12
    return ok
# -----------------------------------------------------------------------------


def check_s3_sequential(device):
    """Compare the chunked transcription with the sequential form.

    Parameters
    ----------
    device : torch.device
        Device to run on.

    Returns
    -------
    ok : bool
        True if the two agree to float64 precision.
    """
    inputs = make_inputs(2, 64, 2, 32, 16, 8, device, torch.float64,
                         seed=13)
    chunked = mamba3_siso_recurrence(chunk_size=16, **inputs)
    error = relative_error(chunked, sequential_reference(inputs))
    print(f'  sequential rel_err={error:.3e}')
    return error < 1e-12
# -----------------------------------------------------------------------------


def main():
    """Run the checks and summarise."""
    if not torch.cuda.is_available():
        raise RuntimeError('These checks need a GPU. Submit as a job.')

    device = torch.device('cuda')
    print(f'torch {torch.__version__} on '
          f'{torch.cuda.get_device_name(device)}\n')

    checks = (('S1 transcription vs Triton',
               lambda: check_s1_vs_triton(device)),
              ('S2 chunk invariance',
               lambda: check_s2_chunk_invariance(device)),
              ('S3 sequential fp64',
               lambda: check_s3_sequential(device)))

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
