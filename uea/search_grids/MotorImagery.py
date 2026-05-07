NUM_EPOCHS = None
EARLY_STOPPING_PATIENCE = None
N_STARTUP_TRIALS = None

SPARSE_GRID = {
    "lr":                   ("float_log", 1e-6, 1e-3),
    "weight_decay":         ("categorical", [0.0, 0.001, 0.01, 0.1]),
    "hidden_dim":           ("categorical", [8, 16, 32, 64, 128]),
    "num_layers":           ("categorical", [1, 2, 3, 4, 5, 6, 7, 8]),
    "s5_init_blocks":       ("categorical", [2, 4, 8, 16]),
    "ssm_dim_multiplier":   ("categorical", [1, 2, 4]),
    "discretisation":       ("categorical", ["zoh", "bilinear"]),
    "drop_rate":            ("float_step", 0.0, 0.2, 0.05),
    "learn_lambda":         ("categorical", ["standard", "stable","exp"]),
    "bidir":                ("categorical", [True, False]),
    "encoder_depth":        ("categorical", [0, 1]),
    "lambda_encoder_depth": ("categorical", [0, 1]),
    "bc_rank":              ("categorical", [2, 4, 8, 16]),
    "dt_min":               ("categorical", [0.001]),
    "ff_mult":              ("categorical", [1.0]),
    "batch_size":           ("categorical", [5, 10, 20]),
    "random_percentage":    ("categorical", [1.0]),
    "epoch":                ("categorical", [200,400,600]),
    # fixed TIDES settings — not tuned
    "proj_norm":            ("categorical", ["rmsnorm"]),
    "proj_init_method":     ("categorical", ["zeros"]),
    "conj_sym":             ("categorical", [False]),
    "clip_eigs":            ("categorical", [True, False]),
}

GRID = SPARSE_GRID
