"""Per-dataset Optuna search grids for TIDES on Physiome-ODE.

Each module in this package defines:
    SEARCH_GRID             - dict of hyperparameter specs
                              (consumed by ``suggest`` in
                              ``hypersearch_physio.py``)
    NUM_STEPS               - int or None (None means use CLI default)
    EARLY_STOPPING_PATIENCE - int or None (None means use CLI default)
    MAX_PARAMS              - int or None (optional parameter budget)

Importing this package gives access to:
    GRIDS               - {name: SEARCH_GRID}
    NUM_STEPS_MAP       - {name: NUM_STEPS}              (None excluded)
    EARLY_STOPPING_MAP  - {name: EARLY_STOPPING_PATIENCE} (None excluded)
    MAX_PARAMS_MAP      - {name: MAX_PARAMS}              (None excluded)
    GRID_NAMES          - sorted list of all registered names
"""

from . import Physiome_ODE

_MODULES = {
    'Physiome_ODE': Physiome_ODE,
}

GRIDS = {name: mod.SEARCH_GRID for name, mod in _MODULES.items()}
NUM_STEPS_MAP = {
    name: mod.NUM_STEPS for name, mod in _MODULES.items()
    if getattr(mod, 'NUM_STEPS', None) is not None}
EARLY_STOPPING_MAP = {
    name: mod.EARLY_STOPPING_PATIENCE for name, mod in _MODULES.items()
    if getattr(mod, 'EARLY_STOPPING_PATIENCE', None) is not None}
MAX_PARAMS_MAP = {
    name: mod.MAX_PARAMS for name, mod in _MODULES.items()
    if getattr(mod, 'MAX_PARAMS', None) is not None}

GRID_NAMES = sorted(_MODULES.keys())
