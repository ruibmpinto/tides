"""Optuna hyperparameter search for TIDES on Physiome-ODE datasets.

Minimises validation MSE using TPE. Resumable via SQLite. Results
saved incrementally to CSV and logged to wandb.

Usage:
    python hypersearch_physio.py \\
        --dataset hodgkin_huxley_1952_variant01 \\
        --fold 0 \\
        --num_trials 10 \\
        --study_name tides_physio_hodgkin \\
        --storage sqlite:///results/hypersearch_hodgkin.db \\
        --data_base_path ../data/physiome_ode

Classes
-------
StopWhenBeatSOTA
    Optuna callback to stop study when SOTA threshold is beaten.

Functions
---------
suggest
    Sample hyperparameters from the search grid.
append_csv
    Append a row to the results CSV file.
make_objective
    Build the Optuna objective closure.
main
    Parse CLI arguments and run the hyperparameter search.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import argparse
import csv
import gc
import os
import time
import traceback
# Third-party
import numpy as np
import optuna
from optuna.samplers import TPESampler
import wandb
# Local
from tides_train_fn import (
    MaxParamsExceeded, MinParamsExceeded, train_tides)
from search_grids import GRIDS
from utils import IMTS_dataset  # noqa: F401
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Anonymous'
__credits__ = ['Anonymous']
__status__ = 'Development'
# =============================================================================
#
# =============================================================================
DATASET_WANDB_PROJECT = {}

CSV_COLUMNS = [
    'trial', 'dataset', 'fold', 'status',
    'lr', 'lr_factor', 'weight_decay', 'hidden_size', 'ssm_size',
    'ssm_blocks', 'num_blocks', 'encoder_depth', 'lambda_re_mode',
    'lambda_im_mode', 'bc_mode', 'learn_lambda', 'discretization',
    'drop_rate', 'batch_size', 'dt_min', 'lambda_encoder_depth',
    'bc_rank', 'ff_mult', 'bidir', 'conj_sym', 'clip_eigs',
    'warmup_epochs', 'conv_kernel_size', 'proj_init_method',
    'proj_norm', 'num_params', 'val_loss', 'test_loss', 'test_mae',
]


# =============================================================================
def _select_grid(dataset):
    """Return the search grid that matches a dataset name.

    Dispatches against the registered grids in ``search_grids.GRIDS``.
    Falls back to ``Physiome_ODE`` for unknown dataset names.

    Parameters
    ----------
    dataset : str
        Name of the dataset passed on the CLI.

    Returns
    -------
    grid : dict
        Search grid dictionary consumed by ``suggest``.
    grid_name : str
        Key identifying the chosen grid (for logging).
    """
    if dataset in GRIDS:
        return GRIDS[dataset], dataset
    return GRIDS['Physiome_ODE'], 'Physiome_ODE'
# =============================================================================


# =============================================================================
def suggest(trial, grid):
    """Sample hyperparameters from a search grid.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object used to suggest values.
    grid : dict
        Search grid mapping parameter name to a spec tuple whose
        first element is the kind identifier.

    Returns
    -------
    params : dict
        Mapping from parameter name to sampled value.
    """
    params = {}
    for name, spec in grid.items():
        kind = spec[0]
        if kind == 'float_log':
            params[name] = trial.suggest_float(
                name, spec[1], spec[2], log=True)
        elif kind == 'float_step':
            params[name] = trial.suggest_float(
                name, spec[1], spec[2], step=spec[3])
        elif kind == 'int':
            params[name] = trial.suggest_int(name, spec[1], spec[2])
        elif kind == 'int_step':
            params[name] = trial.suggest_int(
                name, spec[1], spec[2], step=spec[3])
        elif kind == 'categorical':
            params[name] = trial.suggest_categorical(
                name, list(spec[1]))
        else:
            raise ValueError(f'Unknown spec kind: {kind!r}')
    return params
# =============================================================================


# =============================================================================
def append_csv(path, row):
    """Append a row to the results CSV, writing header if needed.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    row : dict
        Row contents keyed by column name.
    """
    exists = os.path.isfile(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(
            f, fieldnames=CSV_COLUMNS, extrasaction='ignore')
        if not exists:
            w.writeheader()
        w.writerow(row)
# =============================================================================


# =============================================================================
def safe_wandb_init(max_attempts=2, init_timeout=300, retry_sleep=30,
                    **init_kwargs):
    """Call ``wandb.init`` with an extended timeout and one retry.

    Parameters
    ----------
    max_attempts : int, default=2
        Total number of init attempts (1 = no retry).
    init_timeout : int, default=300
        Seconds, passed to ``wandb.Settings(init_timeout=...)``.
    retry_sleep : int, default=30
        Seconds to sleep before a retry.
    **init_kwargs
        Forwarded to ``wandb.init``.

    Returns
    -------
    run : {wandb.sdk.wandb_run.Run, None}
        ``None`` if every attempt raised an exception.
    """
    settings = wandb.Settings(init_timeout=init_timeout)
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            return wandb.init(settings=settings, **init_kwargs)
        except Exception as e:
            last_err = e
            print(f'    wandb.init attempt {attempt}/{max_attempts} '
                  f'failed: {type(e).__name__}: {e}')
            if attempt < max_attempts:
                time.sleep(retry_sleep)
    print(f'    wandb.init giving up after {max_attempts} attempts '
          f'(last error: {type(last_err).__name__})')
    return None
# =============================================================================


# =============================================================================
def make_objective(dataset, fold, data_base_path, epochs,
                   early_stop_patience, results_csv, saved_models_dir,
                   study_name, search_grid, wandb_project,
                   min_params=None, max_params=None):
    """Build the Optuna objective closure.

    Parameters
    ----------
    dataset : str
        Name of the Physiome-ODE dataset.
    fold : int
        Fold index.
    data_base_path : str
        Base path containing the dataset splits.
    epochs : int
        Maximum number of training epochs.
    early_stop_patience : int
        Patience for early stopping.
    results_csv : str
        Path to the results CSV file.
    saved_models_dir : str
        Directory where trained models are cached.
    study_name : str
        Name of the Optuna study.
    search_grid : dict
        Search grid dictionary consumed by ``suggest`` (selected per
        dataset via ``_select_grid``).
    wandb_project : str
        Wandb project name to log trials under.
    min_params : {int, None}, default=None
        Skip trials whose model falls below this parameter count
        (pruned as ``param_skip``).
    max_params : {int, None}, default=None
        Skip trials whose model exceeds this parameter count
        (pruned as ``param_skip``).

    Returns
    -------
    objective : callable
        Objective function taking an optuna.Trial and returning
        the validation loss.
    """
    def objective(trial):
        """Evaluate a single Optuna trial.

        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial object.

        Returns
        -------
        val_loss : float
            Validation loss for the trial.
        """
        p = suggest(trial, search_grid)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Parse mode combination
        _parts = p['mode_combo'].split('/')
        if len(_parts) == 3:
            lambda_re_mode, lambda_im_mode, bc_mode = _parts
        else:  # backward compat: 2-part "lambda/bc"
            lambda_re_mode, bc_mode = _parts
            lambda_im_mode = lambda_re_mode
        ssm_size = 2 * p['ssm_blocks'] * p['ssm_dim_mult']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Log trial configuration
        print(f'\n--- Trial {trial.number} ---')
        print(
            f"  hidden={p['hidden_size']}  ssm_size={ssm_size}"
            f"  blocks={p['num_blocks']}"
            f"  mode={p['mode_combo']}  lr={p['lr']:.2e}"
            f"  lr_factor={p['lr_factor']}"
            f"  wd={p['weight_decay']:.2e}  bs={p['batch_size']}"
            f"  drop={p['drop_rate']:.2f}"
            f"  disc={p['discretization']}"
            f"  learn_lambda={p['learn_lambda']}"
            f"  bidir={p['bidir']}  dt_min={p['dt_min']}"
            f"  conj_sym={p['conj_sym']}"
            f"  clip_eigs={p['clip_eigs']}"
            f"  warmup={p['warmup_epochs']}"
            f"  conv_k={p.get('conv_kernel_size', 0)}"
            f"  proj_init="
            f"{p.get('proj_init_method', 'zeros')!r}"
            f"  proj_norm={p.get('proj_norm')!r}")
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build base CSV row
        base_row = {
            'trial': trial.number,
            'dataset': dataset,
            'fold': fold,
            'lr': p['lr'],
            'lr_factor': p['lr_factor'],
            'weight_decay': p['weight_decay'],
            'hidden_size': p['hidden_size'],
            'ssm_size': ssm_size,
            'ssm_blocks': p['ssm_blocks'],
            'num_blocks': p['num_blocks'],
            'encoder_depth': p['encoder_depth'],
            'lambda_re_mode': lambda_re_mode,
            'lambda_im_mode': lambda_im_mode,
            'bc_mode': bc_mode,
            'learn_lambda': p['learn_lambda'],
            'discretization': p['discretization'],
            'drop_rate': p['drop_rate'],
            'batch_size': p['batch_size'],
            'dt_min': p['dt_min'],
            'lambda_encoder_depth': p['lambda_encoder_depth'],
            'bc_rank': p['bc_rank'],
            'ff_mult': p['ff_mult'],
            'bidir': p['bidir'],
            'conj_sym': p['conj_sym'],
            'clip_eigs': p['clip_eigs'],
            'warmup_epochs': p['warmup_epochs'],
            'conv_kernel_size': p.get('conv_kernel_size', 0),
            'proj_init_method': p.get('proj_init_method', 'zeros'),
            'proj_norm': p.get('proj_norm'),
        }
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize wandb run for this trial (retry on transient
        # network failures; prune trial if init ultimately fails).
        wandb_config = {
            **base_row,
            'study': study_name,
            'phase': 'hypersearch',
            'mode_combo': p['mode_combo'],
        }
        wandb_run = safe_wandb_init(
            project=wandb_project,
            name=f'{study_name}_t{trial.number}_f{fold}',
            group=study_name,
            config=wandb_config,
            tags=[dataset, study_name, 'hypersearch',
                  f'fold{fold}'],
            reinit='finish_previous',
        )
        if wandb_run is None:
            append_csv(
                results_csv,
                {**base_row, 'status': 'wandb_init_failed'})
            trial.set_user_attr('no_count', True)
            gc.collect()
            raise optuna.TrialPruned()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Train and record outcome
        try:
            result = train_tides(
                dataset=dataset,
                fold=fold,
                data_base_path=data_base_path,
                hidden_size=p['hidden_size'],
                ssm_size=ssm_size,
                ssm_blocks=p['ssm_blocks'],
                num_blocks=p['num_blocks'],
                encoder_depth=p['encoder_depth'],
                lambda_re_mode=lambda_re_mode,
                lambda_im_mode=lambda_im_mode,
                bc_mode=bc_mode,
                lambda_encoder_depth=p['lambda_encoder_depth'],
                bc_rank=p['bc_rank'],
                ff_mult=p['ff_mult'],
                bidir=p['bidir'],
                learn_lambda=p['learn_lambda'],
                discretization=p['discretization'],
                drop_rate=p['drop_rate'],
                dt_min=p['dt_min'],
                conj_sym=p['conj_sym'],
                clip_eigs=p['clip_eigs'],
                conv_kernel_size=p.get('conv_kernel_size', 0),
                proj_init_method=p.get(
                    'proj_init_method', 'zeros'),
                proj_norm=p.get('proj_norm'),
                lr=p['lr'],
                lr_factor=p['lr_factor'],
                warmup_epochs=p['warmup_epochs'],
                weight_decay=p['weight_decay'],
                batch_size=p['batch_size'],
                epochs=epochs,
                early_stop_patience=early_stop_patience,
                seed=fold,
                saved_models_dir=saved_models_dir,
                verbose=True,
                min_params=min_params,
                max_params=max_params,
                wandb_run=wandb_run,
            )

            val_loss = result['val_loss']
            print(
                f'  → val_loss={val_loss:.4f}'
                f"  test_loss={result['test_loss']:.4f}"
                f"  params={result['num_params']:,}")

            append_csv(results_csv, {
                **base_row,
                'num_params': result['num_params'],
                'val_loss':   val_loss,
                'test_loss':  result['test_loss'],
                'test_mae':   result['test_mae'],
                'status':     'ok',
            })
            wandb.log({
                'val_metric': val_loss,
                'val_loss': val_loss,
                'test_loss': result['test_loss'],
                'test_mae': result['test_mae'],
                'n_params': result['num_params'],
                'status_code': 0,
            })
            wandb.finish()
            gc.collect()
            return val_loss

        except MaxParamsExceeded as e:
            print(
                f'  Param limit exceeded ({e.n_params:,} > '
                f'{max_params:,}) — pruning (not counted)')
            append_csv(
                results_csv, {**base_row, 'num_params': e.n_params,
                              'status': 'param_skip'})
            wandb.log({'status_code': 4, 'n_params': e.n_params})
            wandb.finish()
            trial.set_user_attr('no_count', True)
            gc.collect()
            raise optuna.TrialPruned()

        except MinParamsExceeded as e:
            print(
                f'  Param limit not met ({e.n_params:,} < '
                f'{min_params:,}) — pruning (not counted)')
            append_csv(
                results_csv, {**base_row, 'num_params': e.n_params,
                              'status': 'param_skip'})
            wandb.log({'status_code': 5, 'n_params': e.n_params})
            wandb.finish()
            trial.set_user_attr('no_count', True)
            gc.collect()
            raise optuna.TrialPruned()

        except ValueError as e:
            if 'NaN' in str(e):
                print(f'  NaN loss — pruning')
                append_csv(
                    results_csv, {**base_row, 'status': 'nan'})
                wandb.log({'status_code': 1})
                wandb.finish()
                gc.collect()
                raise optuna.TrialPruned()
            traceback.print_exc()
            append_csv(
                results_csv, {**base_row, 'status': 'error'})
            wandb.log({'status_code': 2})
            wandb.finish()
            gc.collect()
            raise optuna.TrialPruned()

        except KeyboardInterrupt:
            print(
                f'    Interrupted (wandb stop?) — pruning trial '
                f'and continuing study')
            append_csv(
                results_csv, {**base_row, 'status': 'interrupted'})
            wandb.log({'status_code': 3})
            wandb.finish(exit_code=1)
            gc.collect()
            raise optuna.TrialPruned()

        except Exception:
            traceback.print_exc()
            append_csv(
                results_csv, {**base_row, 'status': 'error'})
            wandb.log({'status_code': 2})
            wandb.finish()
            gc.collect()
            raise optuna.TrialPruned()

    return objective
# =============================================================================


SOTA = {
    'dupont_1991a': 0.951,
    'dupont_1991b': 0.622,
    'dupont_1992b': 0.718,
    'borghans_dupont_goldbeter_1997a': 0.709,
    'hynne_dano_sorensen_2001': 0.619,
    'wolf_heinrich_2000': 0.645,
    'wolf_passarge_somsen_snoep_heinrich_westerhoff_2000': 0.784,
    'wolf_sohn_heinrich_kuriyama_2001': 0.073,
    'shorten_ocallaghan_davidson_soboleva_2007': 0.055,
    'shorten_ocallaghan_davidson_soboleva_2007_variant01': 0.037,
    'vanbeek_2007': 0.242,
    'phillips_2007': 0.131,
    'vilar_kueh_barkai_leibler_2002': 0.344,
    'wang_2006': 0.103,
    'nygren_fiset_firek_clark_lindblad_clark_giles_1998': 0.344,
    'lenbury_ruktamatakul_amornsamarnkul_2001_a': 0.039,
    'purvis_smith_koizumi_butera_2007': 0.106,
    'iribe_kohl_noble_2006': 0.037,
    'wodarz_hamer_2007_b': 0.113,
    'bagci_2008a': 0.029,
    'nelson_murray_perelson_2000_general': 0.007,
    'M_blood_flow_parent': 0.003,
    'pulmonary_O2_parent': 0.008,
    'aslanidi_2009': 0.022,
    'calzone_thieffry_tyson_novak_2007': 0.078,
    'huang_ferrell_1996': 0.052,
    'mackenzie_1996': 0.019,
    'maldonado_2006': 0.018,
    'guyton_pulmonary_oxygen_uptake_2008': 0.004,
    'gupta_aslakson_gurbaxani_vernon_2007_a': 0.018,
    'gupta_aslakson_gurbaxani_vernon_2007_b': 0.444,
    'karagiannis_popel_2004': 0.034,
    'li_1996_simple_from_paper': 0.084,
}


# =============================================================================
class StopWhenBeatSOTA:
    """Optuna callback stopping the study once SOTA is beaten.

    Attributes
    ----------
    sota_val : float
        State-of-the-art threshold value for the dataset.

    Methods
    -------
    __call__(self, study, trial)
        Stop the study when the best value beats sota_val.
    """
    def __init__(self, sota_val):
        """Constructor.

        Parameters
        ----------
        sota_val : float
            State-of-the-art threshold value.
        """
        self.sota_val = sota_val
    # -------------------------------------------------------------------------
    def __call__(self, study, trial):
        """Stop the study when the best value beats the threshold.

        Parameters
        ----------
        study : optuna.Study
            Active Optuna study.
        trial : optuna.trial.FrozenTrial
            Frozen trial that just completed.
        """
        if study.best_value < self.sota_val:
            print(
                f'\n*** SOTA beaten! '
                f'best_val={study.best_value:.4f} '
                f'< {self.sota_val:.3f} ***')
            study.stop()
# =============================================================================


# =============================================================================
def main():
    """Parse CLI arguments and run the hyperparameter search.

    Notes
    -----
    Reads CLI flags, creates (or resumes) an Optuna study stored in
    SQLite, runs the requested number of trials, appends results to
    a CSV, and prints a summary of the best trial.
    """
    parser = argparse.ArgumentParser(
        description='Optuna hypersearch for TIDES on Physiome-ODE')
    parser.add_argument('--dataset',        required=True,   type=str)
    parser.add_argument('--fold',           default=0,       type=int)
    parser.add_argument(
        '--data_base_path', default='../data/physiome_ode', type=str)
    parser.add_argument('--num_trials',     default=10,      type=int)
    parser.add_argument('--num_startup_trials',    default=200,      type=int)
    parser.add_argument('--epochs',         default=30,      type=int)
    parser.add_argument('--early_stop',     default=5,       type=int)
    parser.add_argument('--study_name',     default=None,    type=str)
    parser.add_argument('--storage',        default=None,    type=str)
    parser.add_argument(
        '--results_dir', default='results/hypersearch', type=str)
    parser.add_argument(
        '--seed', default=0, type=int, help='TPE sampler seed')
    parser.add_argument(
        '--min_params', type=int, default=None,
        help='Skip trials whose model is below this parameter '
             'count (graceful prune)')
    parser.add_argument(
        '--max_params', type=int, default=None,
        help='Skip trials whose model exceeds this parameter '
             'count (graceful prune)')
    parser.add_argument(
        '--wandb_project', default=None, type=str,
        help='Wandb project name. Defaults to '
             'DATASET_WANDB_PROJECT[dataset] or '
             "'tides-{dataset}' if not mapped.")
    args = parser.parse_args()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Authenticate with wandb once before launching trials
    wandb.login()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Prepare output directories and paths
    os.makedirs(args.results_dir, exist_ok=True)
    saved_models_dir = os.path.join(args.results_dir, 'tmp_models')
    os.makedirs(saved_models_dir, exist_ok=True)

    study_name = (args.study_name
                  or f'tides_physio_{args.dataset}_f{args.fold}')
    storage = (args.storage
               or f'sqlite:///{args.results_dir}/{study_name}.db')
    results_csv = os.path.join(
        args.results_dir, f'{study_name}.csv')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Select storage backend. A string without ``://`` is treated as a
    # filesystem path and backed by JournalStorage (append-only log,
    # NFS-safe, scales to many parallel workers). RDB URLs (sqlite/
    # postgresql/mysql) are passed through unchanged.
    if '://' not in storage:
        from optuna.storages import JournalStorage
        try:
            from optuna.storages.journal import JournalFileBackend
        except ImportError:
            from optuna.storages import \
                JournalFileStorage as JournalFileBackend
        storage = JournalStorage(JournalFileBackend(storage))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Select the per-dataset search grid
    search_grid, grid_name = _select_grid(args.dataset)
    print(f"Using search grid '{grid_name}' for dataset "
          f"'{args.dataset}'")
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create or resume the Optuna study
    sampler = TPESampler(
        seed=args.seed, multivariate=True,
        n_startup_trials=max(5, args.num_startup_trials))
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='minimize',
        sampler=sampler,
        load_if_exists=True,
    )

    wandb_project = (
        args.wandb_project
        or DATASET_WANDB_PROJECT.get(
            args.dataset, f'tides-{args.dataset}'))
    print(f"Wandb project: '{wandb_project}'")

    objective = make_objective(
        dataset=args.dataset,
        fold=args.fold,
        data_base_path=args.data_base_path,
        epochs=args.epochs,
        early_stop_patience=args.early_stop,
        results_csv=results_csv,
        saved_models_dir=saved_models_dir,
        study_name=study_name,
        search_grid=search_grid,
        wandb_project=wandb_project,
        min_params=args.min_params,
        max_params=args.max_params,
    )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Configure callbacks and run the optimisation
    already_done = len(
        [t for t in study.trials
         if t.state == optuna.trial.TrialState.COMPLETE])
    print(
        f"Study '{study_name}': {already_done} done so far, "
        f'this worker running {args.num_trials} trial(s)')

    callbacks = []
    sota_val = SOTA.get(args.dataset)
    if sota_val is not None:
        print(
            f'SOTA target for {args.dataset}: {sota_val} '
            f'— will stop early if beaten')
        callbacks.append(StopWhenBeatSOTA(sota_val))

    # Trials pruned for parameter-count violations set the
    # ``no_count`` user attr and do not count towards the budget.
    n_counted = 0
    while n_counted < args.num_trials:
        study.optimize(objective, n_trials=1, callbacks=callbacks)
        last_trial = study.trials[-1]
        if last_trial.user_attrs.get('no_count'):
            print(
                f'  ↳ Trial {last_trial.number} skipped — not counted '
                f'({n_counted}/{args.num_trials} real trials so far)')
        else:
            n_counted += 1
            print(
                f'  ↳ Trial {last_trial.number} counted '
                f'({n_counted}/{args.num_trials} real trials)')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Print final summary
    print(f'\nSearch complete.')
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE]
    if completed:
        print(f'Best trial : #{study.best_trial.number}')
        print(f'Best val   : {study.best_value:.4f}')
        print(f'Best params: {study.best_params}')
    else:
        print('No completed trials.')
    print(f'Results CSV: {results_csv}')
# =============================================================================


if __name__ == "__main__":
    main()