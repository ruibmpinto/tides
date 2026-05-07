"""Optuna search grid for TIDES on Physiome-ODE datasets.

Keys match the argument names consumed by
``hypersearch_physio.py::suggest`` and forwarded to
``tides_train_fn.train_tides``.
"""

NUM_STEPS = None
EARLY_STOPPING_PATIENCE = None
MAX_PARAMS = None

SEARCH_GRID = {
    "lr":              ("float_log",   1e-6,   5e-4),
    "lr_factor":       ("int",         1,      250),
    "weight_decay":    ("float_log",   1e-7,   1e-1),
    "hidden_size":     ("int_step", 16, 192, 16),
    "ssm_blocks":      ("int_step", 1, 8, 1),       
    "ssm_dim_mult":    ("int",         1, 8),       # ssm_size = 2 * ssm_blocks * mult
    "num_blocks":      ("int_step",         2, 12, 1),
    "encoder_depth":   ("categorical", [0,1,2]),
    "mode_combo":      ("categorical", [
                            "lti/lti/input_dependent",
                            "input_dependent/lti/input_dependent",
                            "input_dependent/input_dependent/input_dependent",
                       ]),
    "learn_lambda":    ("categorical", ["standard","exp","stable","softplus"]),
    "discretization":  ("categorical", ["zoh"]),
    "drop_rate":       ("float_step",  0.0, 0.3, 0.05),
    "batch_size":      ("int_step", 64, 164, 16),
    "dt_min":               ("categorical", [0.001]),
    "lambda_encoder_depth": ("categorical", [0]),
    "bc_rank":              ("categorical", [-1, 4, 8, 12, 16, 20]),
    "ff_mult":              ("categorical", [0.25, 0.5, 1, 2, 3, 4, 6, 8]),
    "bidir":               ("categorical", [True, False]),
    "conj_sym":            ("categorical", [False]),
    "clip_eigs":           ("categorical", [True, False]),
    "warmup_epochs":       ("categorical", [10]),
    "conv_kernel_size":    ("categorical", [0, 4]),
    "proj_init_method":    ("categorical", ["zeros", "random"]),
    "proj_norm":           ("categorical", [None, "rmsnorm"]),
}
