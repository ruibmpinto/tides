# Mamba-3 on EigenWorms.
#
# Shared axes take the same values as the TIDES EigenWorms grid so the
# baseline gets the same search space and budget the proposed model did.
# Layers are kept low and hidden_dim capped at 64: EigenWorms overfits
# quickly at this sequence length (236 sequences, seq_len 17984).
#
# Axes TIDES searches that Mamba-3 does not possess, and which are
# therefore absent here: bidir, discretisation, learn_lambda, bc_rank,
# encoder_depth, lambda_encoder_depth, conj_sym, clip_eigs, proj_norm,
# and the s5_init_blocks / ssm_dim_multiplier pair. TIDES also searches
# discretisation over {zoh, bilinear}, whereas Mamba-3's
# exponential-trapezoidal coefficient is learned per step rather than
# chosen, so there is nothing to sweep. TIDES therefore
# searches a strictly richer space on axes with no Mamba-3 counterpart --
# an asymmetry in TIDES's favour that belongs in the write-up rather than
# being papered over.
#
# expand plays roughly the role of the TIDES ff_mult. d_state mirrors the
# range of the TIDES ssm_size; that quantity is derived there as
# 2 * s5_init_blocks * ssm_dim_multiplier, a parameterisation specific to
# HiPPO block initialisation, so only the range carries across.
#
# Mamba3 asserts d_inner = expand * hidden_dim is divisible by headdim.
# An invalid pair is reduced to the largest valid divisor by
# models.largest_valid_headdim rather than pruning the trial, which keeps
# the search space static for multivariate TPE. With expand=1 and
# hidden_dim=8 that leaves headdim=8 as the only option.
# MIMO is not searched: this grid specifies no mimo_mode, so every trial
# runs SISO (the default in models.build_ssm_layer). Both modes go
# through the PyTorch backend in mamba3_recurrence.py; chunk_size is not
# an axis, being invariant in the result and a throughput knob only.
NUM_EPOCHS = None
EARLY_STOPPING_PATIENCE = None
N_STARTUP_TRIALS = None

SPARSE_GRID = {
    "lr":            ("float_log", 1e-6, 1e-3),
    "weight_decay":  ("categorical", [0.0, 0.001, 0.01, 0.1]),
    "hidden_dim":    ("categorical", [8, 16, 32, 64]),
    "num_layers":    ("categorical", [1, 2, 3]),
    "d_state":       ("categorical", [16, 32, 64]),
    "drop_rate":     ("categorical", [0.0, 0.05, 0.10, 0.15, 0.20]),
    "batch_size":    ("categorical", [5, 10, 20]),
    "epoch":         ("categorical", [200, 400]),
    "expand":        ("categorical", [1, 2]),
    "headdim":       ("categorical", [8, 16, 32, 64]),
    # Mamba-3 specific: the fraction of the state dimension carrying the
    # rotary phase. 0.5 gives rotary_dim_divisor 4, 1.0 gives 2.
    "rope_fraction": ("categorical", [0.5, 1.0]),
}

GRID = SPARSE_GRID
