"""Mamba-3 prefill in PyTorch, for both SISO and MIMO.

One core recurrence, generalised over the MIMO rank, with SISO being the
rank-1 case with unit projections. The arithmetic is the transcription of
the Triton SISO kernel validated in test_mamba3_siso_recurrence.py, which
sits at float32 machine precision (3.5e-07) from the exact float64 answer
while that kernel is ~8e-03 away from it -- the kernel loses about three
digits to TF32 tensor cores and PTX fast-math.

Why this exists: Mamba-3's MIMO prefill upstream is a TileLang kernel
that needs a Hopper GPU, so on any other card MIMO cannot run at all
(the import falls back to None and Mamba3 then asserts). SISO is routed
through here too, so that a SISO-versus-MIMO comparison is not
confounded by a change of kernel or of arithmetic precision.

Rotary layout: the two upstream kernels disagree. The Triton SISO kernel
pairs channel 2i with 2i+1 over all headdim_qk // 2 pairs, taking the
angle as zero past headdim_angles; the TileLang MIMO kernel pairs n with
headdim_qk // 2 + n over headdim_qk // rotary_dim_divisor pairs. The two
are related by a fixed permutation of state channels, verified exactly,
which the learned Q/K projections and biases absorb -- so the choice
cannot change what the model can represent. This module uses the
interleaved layout for both modes, because that is the one proven exact
against an upstream kernel.

Being a plain differentiable forward, autograd supplies the backward.
Upstream hand-writes one only because its forward is opaque compiled
code.

Functions
---------
angle_cumsum
    Cumulative tanh(angle) * pi * dt, reduced modulo 2*pi.
apply_rotary
    Rotate channel pairs, interleaved or split-half.
trap_scales
    Per-step gamma and gamma + shifted_gamma.
mamba3_recurrence
    Chunked prefill, generalised over the MIMO rank.
mamba3_siso_combined_torch
    Adapter matching the upstream Triton SISO signature.
mamba3_mimo_combined_torch
    Adapter matching the upstream TileLang MIMO signature.
install_torch_backend
    Point mamba_ssm.modules.mamba3 at these implementations.
"""

#
#                                                                       Modules
# =============================================================================
# Standard
import math
# Third-party
import torch
import torch.nn.functional as F

#
# =============================================================================


def angle_cumsum(angles, dt):
    """Cumulative tanh(angle) * pi * dt, reduced modulo 2*pi.

    Parameters
    ----------
    angles : torch.Tensor
        Raw angle rates, shape (batch, seqlen, nheads, n_angles).
    dt : torch.Tensor
        Time deltas, shape (batch, nheads, seqlen).

    Returns
    -------
    cumulative : torch.Tensor
        Accumulated phase, same shape as angles.

    Notes
    -----
    Mirrors angle_dt_fwd, which squashes the raw angle by tanh, scales by
    pi, multiplies by dt, accumulates over the whole sequence and reduces
    mod 2*pi. The reduction is a no-op for cos and sin and has the
    gradient of the identity almost everywhere, but it bounds the phase
    and so avoids precision loss over long sequences.
    """
    scaled = torch.tanh(angles) * math.pi * dt.permute(0, 2, 1).unsqueeze(-1)
    cumulative = torch.cumsum(scaled, dim=1)
    two_pi = 2.0 * math.pi
    return cumulative - two_pi * torch.floor(cumulative / two_pi)
# -----------------------------------------------------------------------------


def apply_rotary(tensor, cumulative, layout='interleaved',
                 rotary_dim_divisor=4):
    """Rotate channel pairs, interleaved or split-half.

    Parameters
    ----------
    tensor : torch.Tensor
        Shape (batch, seqlen, rank, nheads, n_state).
    cumulative : torch.Tensor
        Accumulated angles, shape (batch, seqlen, nheads, n_angles).
    layout : {'interleaved', 'split'}, default='interleaved'
        'interleaved' pairs channel 2i with 2i+1 over all n_state // 2
        pairs, zero-padding angles past n_angles, as the Triton SISO
        kernel does. 'split' pairs n with n_state // 2 + n over
        n_state // rotary_dim_divisor pairs, as the TileLang MIMO kernel
        does.
    rotary_dim_divisor : int, default=4
        Used by the 'split' layout only.

    Returns
    -------
    rotated : torch.Tensor
        Same shape as the input.
    """
    n_state = tensor.shape[-1]
    angles = cumulative.unsqueeze(2)

    if layout == 'interleaved':
        n_pairs = n_state // 2
        n_angles = angles.shape[-1]
        if n_angles < n_pairs:
            pad = torch.zeros(*angles.shape[:-1], n_pairs - n_angles,
                              device=angles.device, dtype=angles.dtype)
            angles = torch.cat([angles, pad], dim=-1)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        pairs = tensor.reshape(*tensor.shape[:-1], n_pairs, 2)
        first = pairs[..., 0]
        second = pairs[..., 1]
        rotated = torch.stack([cos * first - sin * second,
                               sin * first + cos * second], dim=-1)
        return rotated.reshape(tensor.shape)

    if layout == 'split':
        n_rot = n_state // rotary_dim_divisor
        half = n_state // 2
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        first = tensor[..., :n_rot]
        second = tensor[..., half:half + n_rot]
        rotated = tensor.clone()
        rotated[..., :n_rot] = cos * first - sin * second
        rotated[..., half:half + n_rot] = sin * first + cos * second
        return rotated

    raise RuntimeError(f'Unknown rotary layout {layout!r}.')
# -----------------------------------------------------------------------------


def trap_scales(dt, trap):
    """Per-step gamma and gamma + shifted_gamma.

    Parameters
    ----------
    dt : torch.Tensor
        Time deltas, shape (batch, nheads, seqlen).
    trap : torch.Tensor
        Trapezoidal factor before the sigmoid, same shape.

    Returns
    -------
    gamma : torch.Tensor
        dt * sigmoid(trap), weighting the same-step term.
    scale : torch.Tensor
        gamma plus dt_{t+1} * (1 - sigmoid(trap_{t+1})), weighting K in
        both the intra-chunk product and the state update. The shifted
        term is zero at the final step.

    Notes
    -----
    The shift crosses chunk boundaries, so this is computed on the whole
    sequence before any chunking.
    """
    sigmoid_trap = torch.sigmoid(trap)
    gamma = dt * sigmoid_trap

    shifted = dt * (1.0 - sigmoid_trap)
    shifted_gamma = torch.zeros_like(gamma)
    shifted_gamma[..., :-1] = shifted[..., 1:]

    return gamma, gamma + shifted_gamma
# -----------------------------------------------------------------------------


def mamba3_recurrence(Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles,
                      MIMO_V=None, MIMO_Z=None, MIMO_Out=None, D=None,
                      Z=None, chunk_size=64, layout='interleaved',
                      rotary_dim_divisor=4):
    """Chunked prefill, generalised over the MIMO rank.

    Parameters
    ----------
    Q : torch.Tensor
        Queries, shape (batch, seqlen, rank, ngroups, n_state).
    K : torch.Tensor
        Keys, same shape as Q.
    V : torch.Tensor
        Values, shape (batch, seqlen, nheads, headdim).
    ADT : torch.Tensor
        A * dt, shape (batch, nheads, seqlen).
    DT : torch.Tensor
        dt, same shape.
    Trap : torch.Tensor
        Trapezoidal factor before the sigmoid, same shape.
    Q_bias : torch.Tensor
        Shape (nheads, rank, n_state).
    K_bias : torch.Tensor
        Shape (nheads, rank, n_state).
    Angles : torch.Tensor
        Raw angle rates, shape (batch, seqlen, nheads, n_angles).
    MIMO_V : {torch.Tensor, None}, default=None
        Value up-projection, shape (nheads, rank, headdim). None means
        unit, i.e. the SISO case.
    MIMO_Z : {torch.Tensor, None}, default=None
        Gate up-projection, same shape. None means unit.
    MIMO_Out : {torch.Tensor, None}, default=None
        Output down-projection, same shape. None leaves the rank axis
        unreduced.
    D : {torch.Tensor, None}, default=None
        Skip weight, shape (nheads,).
    Z : {torch.Tensor, None}, default=None
        Gate, shape (batch, seqlen, nheads, headdim).
    chunk_size : int, default=64
        Chunk length. The result does not depend on it.
    layout : {'interleaved', 'split'}, default='interleaved'
        Rotary channel pairing.
    rotary_dim_divisor : int, default=4
        Used by the 'split' layout only.

    Returns
    -------
    out : torch.Tensor
        Shape (batch, seqlen, nheads, headdim) when MIMO_Out is given or
        the rank is 1, otherwise (batch, seqlen, rank, nheads, headdim).

    Notes
    -----
    Trap arrives before the sigmoid, matching the upstream prefill
    convention. Accumulation is promoted to at least float32 and never
    downcast, so a float64 caller stays in float64.
    """
    batch, seqlen, rank, ngroups, n_state = Q.shape
    nheads, headdim = V.shape[2], V.shape[3]

    if seqlen % chunk_size != 0:
        raise RuntimeError(
            f'seqlen ({seqlen}) must be a multiple of chunk_size '
            f'({chunk_size}).')
    nchunks = seqlen // chunk_size

    acc = torch.promote_types(Q.dtype, torch.float32)
    dt = DT.to(acc)
    adt = ADT.to(acc)
    repeat = nheads // ngroups

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Bias, same-step term from the unrotated pair, then rotary.
    gamma, scale = trap_scales(dt, Trap.to(acc))

    q_pre = (Q.to(acc).repeat_interleave(repeat, dim=3)
             + Q_bias.to(acc).permute(1, 0, 2))
    k_pre = (K.to(acc).repeat_interleave(repeat, dim=3)
             + K_bias.to(acc).permute(1, 0, 2))

    qk_dot = torch.einsum('blahn,blehn->blhae', q_pre, k_pre)
    qk_dot = qk_dot * gamma.permute(0, 2, 1)[..., None, None]

    cumulative = angle_cumsum(Angles.to(acc), dt)
    q_rot = apply_rotary(q_pre, cumulative, layout, rotary_dim_divisor)
    k_rot = apply_rotary(k_pre, cumulative, layout, rotary_dim_divisor)
    k_rot = k_rot * scale.permute(0, 2, 1)[..., None, :, None]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Up-project the values and the gate.
    psi_v = V.to(acc).unsqueeze(2)
    if MIMO_V is not None:
        psi_v = psi_v * MIMO_V.to(acc).permute(1, 0, 2)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Chunked main loop.
    shape = (batch, nchunks, chunk_size)
    q_c = q_rot.reshape(*shape, rank, nheads, n_state)
    k_c = k_rot.reshape(*shape, rank, nheads, n_state)
    v_c = psi_v.reshape(*shape, rank, nheads, headdim)
    qk_c = qk_dot.reshape(*shape, nheads, rank, rank)
    adt_c = adt.reshape(batch, nheads, nchunks, chunk_size)

    da_cs = torch.cumsum(adt_c, dim=-1)
    da_cs_last = da_cs[..., -1]
    da_cs_rev = da_cs_last.unsqueeze(-1) - da_cs

    causal = torch.tril(
        torch.ones(chunk_size, chunk_size, device=Q.device,
                   dtype=torch.bool), diagonal=-1)
    decay = torch.clamp(da_cs.unsqueeze(-1) - da_cs.unsqueeze(-2),
                        max=0.0)

    # Intra-chunk: strictly earlier steps within the chunk.
    s_block = torch.einsum('bcirhn,bcjshn->bhcirjs', q_c, k_c)
    weight = (torch.exp(decay) * causal)[:, :, :, :, None, :, None]
    o_intra = torch.einsum('bhcirjs,bcjshp->bcirhp', s_block * weight,
                           v_c)

    # Same-step term and the D skip.
    o_diag = torch.einsum('bcihae,bciehp->bciahp', qk_c, v_c)
    if D is not None:
        o_diag = o_diag + D.to(acc)[None, None, None, None, :, None] * v_c

    # Inter-chunk: the state recurrence is sequential over chunks.
    v_scaled = v_c * torch.exp(da_cs_rev).permute(
        0, 2, 3, 1)[..., None, :, None]
    chunk_delta = torch.einsum('bcirhn,bcirhp->bchnp', k_c, v_scaled)
    chunk_decay = torch.exp(da_cs_last).permute(0, 2, 1)

    states = []
    state = torch.zeros(batch, nheads, n_state, headdim, device=Q.device,
                        dtype=acc)
    for chunk in range(nchunks):
        states.append(state)
        state = state * chunk_decay[:, chunk][..., None, None] \
            + chunk_delta[:, chunk]
    states = torch.stack(states, dim=1)

    o_inter = torch.einsum('bcirhn,bchnp->bcirhp', q_c, states)
    o_inter = o_inter * torch.exp(da_cs).permute(
        0, 2, 3, 1)[..., None, :, None]

    out = (o_inter + o_intra + o_diag).reshape(batch, seqlen, rank,
                                               nheads, headdim)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Gate, then optionally reduce the rank axis.
    if Z is not None:
        gate = Z.to(acc).unsqueeze(2)
        if MIMO_Z is not None:
            gate = gate * MIMO_Z.to(acc).permute(1, 0, 2)
        out = out * F.silu(gate)

    if MIMO_Out is not None:
        out = torch.einsum('blrhp,hrp->blhp', out, MIMO_Out.to(acc))
    elif rank == 1:
        out = out.squeeze(2)

    return out.to(V.dtype)
# -----------------------------------------------------------------------------


def mamba3_siso_combined_torch(Q, K, V, ADT, DT, Trap, Q_bias, K_bias,
                               Angles, D, Z, chunk_size,
                               Input_States=None,
                               return_final_states=False,
                               cu_seqlens=None):
    """Adapter matching the upstream Triton SISO signature.

    Parameters
    ----------
    Q, K : torch.Tensor
        Shape (batch, seqlen, ngroups, n_state), rank axis squeezed out.
    V : torch.Tensor
        Shape (batch, seqlen, nheads, headdim).
    ADT, DT, Trap : torch.Tensor
        Shape (batch, nheads, seqlen). Trap is pre-sigmoid.
    Q_bias, K_bias : torch.Tensor
        Shape (nheads, n_state), rank axis squeezed out.
    Angles : torch.Tensor
        Shape (batch, seqlen, nheads, n_angles).
    D : {torch.Tensor, None}
        Skip weight, shape (nheads,).
    Z : {torch.Tensor, None}
        Gate, shape (batch, seqlen, nheads, headdim).
    chunk_size : int
        Chunk length.
    Input_States : {tuple, None}, default=None
        Unsupported; prefill starts from a zero state.
    return_final_states : bool, default=False
        Unsupported; only prefill is needed for classification.
    cu_seqlens : {torch.Tensor, None}, default=None
        Unsupported.

    Returns
    -------
    out : torch.Tensor
        Shape (batch, seqlen, nheads, headdim).
    """
    if Input_States is not None:
        raise NotImplementedError(
            'Input_States is not supported by the PyTorch backend.')
    if return_final_states:
        raise NotImplementedError(
            'return_final_states is not supported by the PyTorch '
            'backend: only the prefill path is needed here.')
    if cu_seqlens is not None:
        raise NotImplementedError(
            'cu_seqlens is not supported by the PyTorch backend.')

    return mamba3_recurrence(
        Q=Q.unsqueeze(2), K=K.unsqueeze(2), V=V, ADT=ADT, DT=DT,
        Trap=Trap, Q_bias=Q_bias.unsqueeze(1),
        K_bias=K_bias.unsqueeze(1), Angles=Angles, D=D, Z=Z,
        chunk_size=chunk_size, layout='interleaved')
# -----------------------------------------------------------------------------


def mamba3_mimo_combined_torch(Q, K, V, ADT, DT, Trap, Q_bias, K_bias,
                               MIMO_V, MIMO_Z, MIMO_Out, Angles, D, Z,
                               chunk_size, rotary_dim_divisor, dtype,
                               return_state=False, cu_seqlens=None):
    """Adapter matching the upstream TileLang MIMO signature.

    Parameters
    ----------
    Q, K : torch.Tensor
        Shape (batch, seqlen, rank, ngroups, n_state).
    V : torch.Tensor
        Shape (batch, seqlen, nheads, headdim).
    ADT, DT, Trap : torch.Tensor
        Shape (batch, nheads, seqlen). Trap is pre-sigmoid.
    Q_bias, K_bias : torch.Tensor
        Shape (nheads, rank, n_state).
    MIMO_V, MIMO_Z : torch.Tensor
        Up-projections, shape (nheads, rank, headdim).
    MIMO_Out : {torch.Tensor, None}
        Down-projection, shape (nheads, rank, headdim). None leaves the
        rank axis unreduced.
    Angles : torch.Tensor
        Shape (batch, seqlen, nheads, n_angles).
    D : {torch.Tensor, None}
        Skip weight, shape (nheads,).
    Z : {torch.Tensor, None}
        Gate, shape (batch, seqlen, nheads, headdim).
    chunk_size : int
        Chunk length.
    rotary_dim_divisor : int
        Accepted for signature compatibility; unused, because the
        interleaved layout is used for both modes.
    dtype : torch.dtype
        Accepted for signature compatibility; the output follows V.
    return_state : bool, default=False
        Unsupported; only prefill is needed for classification.
    cu_seqlens : {torch.Tensor, None}, default=None
        Unsupported.

    Returns
    -------
    out : torch.Tensor
        Shape (batch, seqlen, nheads, headdim) when MIMO_Out is given,
        otherwise (batch, seqlen, rank, nheads, headdim).
    """
    if return_state:
        raise NotImplementedError(
            'return_state is not supported by the PyTorch backend: only '
            'the prefill path is needed for classification.')
    if cu_seqlens is not None:
        raise NotImplementedError(
            'cu_seqlens is not supported by the PyTorch backend.')

    return mamba3_recurrence(
        Q=Q, K=K, V=V, ADT=ADT, DT=DT, Trap=Trap, Q_bias=Q_bias,
        K_bias=K_bias, Angles=Angles, MIMO_V=MIMO_V, MIMO_Z=MIMO_Z,
        MIMO_Out=MIMO_Out, D=D, Z=Z, chunk_size=chunk_size,
        layout='interleaved')
# -----------------------------------------------------------------------------


def install_torch_backend(verbose=True, siso=True):
    """Point mamba_ssm.modules.mamba3 at these implementations.

    Parameters
    ----------
    verbose : bool, default=True
        Print what was replaced.
    siso : bool, default=True
        Also route the SISO path through PyTorch. The upstream Triton
        SISO kernel does run on non-Hopper hardware, so this is a choice:
        it keeps both Mamba-3 modes on one validated core, so that a
        SISO-versus-MIMO comparison is not confounded by a change of
        kernel or of arithmetic precision.

    Returns
    -------
    replaced : bool
        True once the entry points are installed.

    Notes
    -----
    Upstream is not forked; the module attributes are rebound, which
    keeps the swap reversible and easy to audit.
    """
    import mamba_ssm.modules.mamba3 as mamba3

    previous = getattr(mamba3, 'mamba3_mimo_combined', None)
    mamba3.mamba3_mimo_combined = mamba3_mimo_combined_torch
    if verbose:
        state = 'None' if previous is None else 'the TileLang kernel'
        print(f'mamba3_mimo_combined: replaced {state} with the PyTorch '
              f'backend')

    if siso:
        mamba3.mamba3_siso_combined = mamba3_siso_combined_torch
        if verbose:
            print('mamba3_siso_combined: replaced the Triton kernel with '
                  'the PyTorch backend')

    return True
