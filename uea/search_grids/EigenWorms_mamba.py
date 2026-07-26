# Mamba-1 control on EigenWorms.
#
# Not one of the models the reviewer asked about. Swept as a control: it
# already has a published EigenWorms entry (70.9 +/- 15.8, Walker et al.)
# obtained under a different protocol, so re-running it here separates an
# architecture effect from a protocol effect in the Mamba-2 and Mamba-3
# columns. It also acts as the correctness gate on data and harness.
#
# Shared axes take the same values as the TIDES EigenWorms grid so the
# baselines get the same search space and budget the proposed model did.
#
# 5 classes, 236 sequences after de-duplication, seq_len 17984, 6 channels.
NUM_EPOCHS = None
EARLY_STOPPING_PATIENCE = None
N_STARTUP_TRIALS = None

# d_conv and expand are searched rather than fixed because the published
# EigenWorms Mamba configuration uses the non-default d_conv=2, expand=1,
# so both settings have to be reachable.
SPARSE_GRID = {
    # Shared with the TIDES EigenWorms grid, same values.
    "lr":            ("float_log", 1e-6, 1e-3),
    "weight_decay":  ("categorical", [0.0, 0.001, 0.01, 0.1]),
    "hidden_dim":    ("categorical", [8, 16, 32, 64, 128]),
    "num_layers":    ("categorical", [1, 2]),
    "d_state":       ("categorical", [8, 16, 32, 64, 128]),
    "drop_rate":     ("float_step", 0.0, 0.2, 0.05),
    "batch_size":    ("categorical", [5, 10, 20]),
    "epoch":         ("categorical", [200, 400, 600]),
    # Mamba-1 specific.
    "d_conv":        ("categorical", [2, 4]),
    "expand":        ("categorical", [1, 2]),
}

GRID = SPARSE_GRID
