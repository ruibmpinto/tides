"""Exact PyTorch transcription of the Mamba-3 Triton SISO kernel.

This is a line-for-line port of mamba_ssm/ops/triton/mamba3/
mamba3_siso_fwd.py (its Phase 1 preprocessing and Phase 2 chunked main
loop) plus the angle cumsum that mamba3_siso_combined applies to the raw
angles before calling it. The goal is exact agreement with that kernel,
not a general implementation: where the kernel and the TileLang MIMO
kernel disagree, this file follows the SISO kernel.

Known points where the two upstream kernels differ, and this file follows
SISO:

* Rotary layout. SISO reshapes to (..., headdim_qk // 2, 2) and splits,
  pairing channel 2i with 2i+1. The MIMO kernel instead pairs n with
  headdim_qk // 2 + n. The two are related by a fixed permutation of
  state channels.
* Rotary width. SISO rotates all headdim_qk // 2 pairs, taking the angle
  as zero -- hence no rotation -- for pairs beyond headdim_angles. The
  MIMO kernel rotates only headdim_qk // rotary_dim_divisor pairs.

Everything else follows the kernel directly: biases are added before the
rotary; the same-step term is formed from the biased but unrotated Q and
K and scaled by gamma; K is scaled by gamma + shifted_gamma before both
the intra-chunk product and the state update; and the intra-chunk mask is
strictly lower triangular.

Functions
---------
angle_cumsum
    Cumulative tanh(angle) * pi * dt, reduced modulo 2*pi.
apply_rotary_interleaved
    Rotate channel pairs (2i, 2i+1), zero-padding missing angles.
trap_scales
    Per-step gamma and gamma + shifted_gamma.
mamba3_siso_recurrence
    Chunked prefill matching mamba3_siso_combined.
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
        Raw angle rates, shape (batch, seqlen, nheads, headdim_angles).
    dt : torch.Tensor
        Time deltas, shape (batch, nheads, seqlen).

    Returns
    -------
    cumulative : torch.Tensor
        Accumulated phase, same shape as angles.

    Notes
    -----
    Mirrors angle_dt_fwd: the raw angle is squashed by tanh and scaled by
    pi, multiplied by dt, accumulated over the whole sequence, and
    reduced mod 2*pi. The kernel uses a PTX tanh approximation, so exact
    agreement is limited by that.
    """
    scaled = torch.tanh(angles) * math.pi * dt.permute(0, 2, 1).unsqueeze(-1)
    cumulative = torch.cumsum(scaled, dim=1)
    two_pi = 2.0 * math.pi
    return cumulative - two_pi * torch.floor(cumulative / two_pi)
# -----------------------------------------------------------------------------


def apply_rotary_interleaved(tensor, cumulative):
    """Rotate channel pairs (2i, 2i+1), zero-padding missing angles.

    Parameters
    ----------
    tensor : torch.Tensor
        Shape (batch, seqlen, nheads, headdim_qk).
    cumulative : torch.Tensor
        Accumulated angles, shape (batch, seqlen, nheads,
        headdim_angles), where headdim_angles <= headdim_qk // 2.

    Returns
    -------
    rotated : torch.Tensor
        Same shape as the input.

    Notes
    -----
    The kernel loads the angle block masked with other=0.0 over
    headdim_qk // 2 columns, so pairs past headdim_angles rotate by a
    zero angle, i.e. not at all. Padding here reproduces that.
    """
    n_state = tensor.shape[-1]
    n_pairs = n_state // 2
    n_angles = cumulative.shape[-1]

    if n_angles < n_pairs:
        pad = torch.zeros(*cumulative.shape[:-1], n_pairs - n_angles,
                          device=cumulative.device,
                          dtype=cumulative.dtype)
        cumulative = torch.cat([cumulative, pad], dim=-1)

    cos = torch.cos(cumulative)
    sin = torch.sin(cumulative)

    pairs = tensor.reshape(*tensor.shape[:-1], n_pairs, 2)
    first = pairs[..., 0]
    second = pairs[..., 1]

    rotated = torch.stack([cos * first - sin * second,
                           sin * first + cos * second], dim=-1)
    return rotated.reshape(tensor.shape)
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
        dt * sigmoid(trap).
    scale : torch.Tensor
        gamma plus dt_{t+1} * (1 - sigmoid(trap_{t+1})), the shifted term
        being zero at the final step.

    Notes
    -----
    Transcribes the kernel's shifted_gamma = dt_shifted * (1 -
    trap_shifted), gamma = dt * trap, scale = shifted_gamma + gamma. The
    shift crosses chunk boundaries, so it is computed on the whole
    sequence.
    """
    sigmoid_trap = torch.sigmoid(trap)
    gamma = dt * sigmoid_trap

    shifted = dt * (1.0 - sigmoid_trap)
    shifted_gamma = torch.zeros_like(gamma)
    shifted_gamma[..., :-1] = shifted[..., 1:]

    return gamma, gamma + shifted_gamma
# -----------------------------------------------------------------------------


def mamba3_siso_recurrence(Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles,
                           D=None, Z=None, chunk_size=64):
    """Chunked prefill matching mamba3_siso_combined.

    Parameters
    ----------
    Q : torch.Tensor
        Queries, shape (batch, seqlen, nheads_qk, headdim_qk).
    K : torch.Tensor
        Keys, same shape as Q.
    V : torch.Tensor
        Values, shape (batch, seqlen, nheads, headdim_v).
    ADT : torch.Tensor
        A * dt, shape (batch, nheads, seqlen).
    DT : torch.Tensor
        dt, same shape.
    Trap : torch.Tensor
        Trapezoidal factor before the sigmoid, same shape.
    Q_bias : torch.Tensor
        Shape (nheads, headdim_qk).
    K_bias : torch.Tensor
        Shape (nheads, headdim_qk).
    Angles : torch.Tensor
        Raw angle rates, shape (batch, seqlen, nheads, headdim_angles).
    D : {torch.Tensor, None}, default=None
        Skip weight, shape (nheads,).
    Z : {torch.Tensor, None}, default=None
        Gate, shape (batch, seqlen, nheads, headdim_v).
    chunk_size : int, default=64
        Chunk length. The result does not depend on it.

    Returns
    -------
    out : torch.Tensor
        Shape (batch, seqlen, nheads, headdim_v).

    Notes
    -----
    Accumulation is promoted to at least float32, and never downcast, so
    a float64 caller stays in float64.
    """
    batch, seqlen, nheads_qk, n_state = Q.shape
    nheads, headdim = V.shape[2], V.shape[3]

    if seqlen % chunk_size != 0:
        raise RuntimeError(
            f'seqlen ({seqlen}) must be a multiple of chunk_size '
            f'({chunk_size}).')
    nchunks = seqlen // chunk_size

    acc = torch.promote_types(Q.dtype, torch.float32)
    dt = DT.to(acc)
    adt = ADT.to(acc)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Phase 1: biases, same-step term, rotary, trapezoidal scaling of K.
    gamma, scale = trap_scales(dt, Trap.to(acc))

    repeat = nheads // nheads_qk
    q_pre = Q.to(acc).repeat_interleave(repeat, dim=2) + Q_bias.to(acc)
    k_pre = K.to(acc).repeat_interleave(repeat, dim=2) + K_bias.to(acc)

    # Same-step term, from the biased but unrotated Q and K.
    qk_dot = (q_pre * k_pre).sum(dim=-1) * gamma.permute(0, 2, 1)

    cumulative = angle_cumsum(Angles.to(acc), dt)
    q_rot = apply_rotary_interleaved(q_pre, cumulative)
    k_rot = apply_rotary_interleaved(k_pre, cumulative)
    k_rot = k_rot * scale.permute(0, 2, 1).unsqueeze(-1)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Phase 2: chunked main loop.
    v_all = V.to(acc)
    shape = (batch, nchunks, chunk_size)
    q_c = q_rot.reshape(*shape, nheads, n_state)
    k_c = k_rot.reshape(*shape, nheads, n_state)
    v_c = v_all.reshape(*shape, nheads, headdim)
    qk_c = qk_dot.reshape(*shape, nheads)
    adt_c = adt.reshape(batch, nheads, nchunks, chunk_size)

    da_cs = torch.cumsum(adt_c, dim=-1)
    da_cs_last = da_cs[..., -1]
    da_cs_rev = da_cs_last.unsqueeze(-1) - da_cs

    causal = torch.tril(
        torch.ones(chunk_size, chunk_size, device=Q.device,
                   dtype=torch.bool), diagonal=-1)

    # Intra-chunk term, identical for every chunk, so computed at once.
    decay = torch.clamp(da_cs.unsqueeze(-1) - da_cs.unsqueeze(-2),
                        max=0.0)
    s_block = torch.einsum('bcihn,bcjhn->bhcij', q_c, k_c)
    s_block = s_block * torch.exp(decay) * causal
    o_intra = torch.einsum('bhcij,bcjhp->bcihp', s_block, v_c)

    # Same-step and skip terms.
    skip = qk_c if D is None else qk_c + D.to(acc)
    o_diag = skip.unsqueeze(-1) * v_c

    # Inter-chunk term: the state recurrence is sequential over chunks.
    v_scaled = v_c * torch.exp(da_cs_rev).permute(0, 2, 3, 1).unsqueeze(-1)
    chunk_delta = torch.einsum('bcihp,bcihn->bchpn', v_scaled, k_c)
    chunk_decay = torch.exp(da_cs_last).permute(0, 2, 1)

    states = []
    state = torch.zeros(batch, nheads, headdim, n_state, device=Q.device,
                        dtype=acc)
    for chunk in range(nchunks):
        states.append(state)
        state = state * chunk_decay[:, chunk][..., None, None] \
            + chunk_delta[:, chunk]
    states = torch.stack(states, dim=1)

    o_inter = torch.einsum('bcihn,bchpn->bcihp', q_c, states)
    o_inter = o_inter * torch.exp(da_cs).permute(0, 2, 3, 1).unsqueeze(-1)

    out = (o_inter + o_intra + o_diag).reshape(batch, seqlen, nheads,
                                               headdim)

    if Z is not None:
        out = out * F.silu(Z.to(acc))

    return out.to(V.dtype)
