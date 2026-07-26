"""
Search grid registry for RFormer TIDES hypersearch.

Each module defines:
    GRID                    – hyperparameter spec dict
    NUM_EPOCHS              – int or None (None → use CLI --epoch)
    EARLY_STOPPING_PATIENCE – int or None
    N_STARTUP_TRIALS        – int or None (None → num_trials // 3)

Importing this package exposes:
    GRIDS                – {dataset_name: GRID}
    NUM_EPOCHS_MAP       – {dataset_name: int}  (None entries excluded)
    EARLY_STOPPING_MAP   – {dataset_name: int}  (None entries excluded)
    N_STARTUP_TRIALS_MAP – {dataset_name: int}  (None entries excluded)
    GRID_NAMES           – sorted list of registered names
"""

from . import (
    EthanolConcentration,
    EigenWorms,
    EigenWorms_mamba,
    EigenWorms_mamba2,
    EigenWorms_mamba3,
    Heartbeat,
    MotorImagery,
    SelfRegulationSCP1,
    SelfRegulationSCP2,
)

# Baseline grids are registered as "<dataset>:<model>". The unsuffixed key
# stays the TIDES grid, so existing callers are unaffected; hypersearch
# builds the suffixed key when --model names a Mamba variant.
_MODULES = {
    "TSC_EthanolConcentration": EthanolConcentration,
    "TSC_EigenWorms":           EigenWorms,
    "TSC_EigenWorms:mamba":     EigenWorms_mamba,
    "TSC_EigenWorms:mamba2":    EigenWorms_mamba2,
    "TSC_EigenWorms:mamba3":    EigenWorms_mamba3,
    "TSC_Heartbeat":            Heartbeat,
    "TSC_MotorImagery":         MotorImagery,
    "TSC_SelfRegulationSCP1":   SelfRegulationSCP1,
    "TSC_SelfRegulationSCP2":   SelfRegulationSCP2,
}

GRIDS = {name: mod.GRID for name, mod in _MODULES.items()}

NUM_EPOCHS_MAP = {
    name: mod.NUM_EPOCHS
    for name, mod in _MODULES.items()
    if mod.NUM_EPOCHS is not None
}

EARLY_STOPPING_MAP = {
    name: mod.EARLY_STOPPING_PATIENCE
    for name, mod in _MODULES.items()
    if mod.EARLY_STOPPING_PATIENCE is not None
}

N_STARTUP_TRIALS_MAP = {
    name: mod.N_STARTUP_TRIALS
    for name, mod in _MODULES.items()
    if mod.N_STARTUP_TRIALS is not None
}

GRID_NAMES = sorted(_MODULES.keys())
