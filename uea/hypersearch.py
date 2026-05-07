"""
Optuna-based hyperparameter search for TIDES on UEA datasets (RFormer setup).

Uses multivariate TPE to maximise validation accuracy. Supports resumable
studies via SQLite. Results are logged to wandb and saved to a CSV file.

Fixed settings (not tuned):
    lambda_re_mode = "input_dependent"
    lambda_im_mode = "lti"
    bc_mode        = "input_dependent"
    conj_sym       = False
    clip_eigs      = False
    proj_norm      = "rmsnorm"

Usage:
    python hypersearch.py --datasets TSC_SelfRegulationSCP1
                          --data_dir /path/to/data
                          --output_dir outputs/hypersearch
                          --epoch 100
                          --num_trials 100
                          --num_seeds 1
                          [--storage sqlite:///hypersearch.db]
                          [--study_name tides-anon]
                          [--wandb_project tides-anon]
"""

import argparse
import csv
import json
import os
import sys
import traceback

import numpy as np
import optuna
import torch
import wandb

# Hypersearch imports train_trial from the sibling main.py
sys.path.insert(0, os.path.dirname(__file__))
from main import train_trial
import search_grids as _sg

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_SEEDS = [42, 43, 44, 45, 46]

SEARCH_GRIDS         = _sg.GRIDS
DATASET_NUM_EPOCHS   = _sg.NUM_EPOCHS_MAP
DATASET_EARLY_STOP   = _sg.EARLY_STOPPING_MAP
DATASET_N_STARTUP    = _sg.N_STARTUP_TRIALS_MAP

CSV_COLUMNS = [
    "trial", "dataset", "seed",
    "lr", "weight_decay",
    "hidden_dim", "num_layers", "s5_init_blocks", "ssm_dim_multiplier", "ssm_size",
    "discretisation", "drop_rate", "learn_lambda",
    "bidir", "encoder_depth", "lambda_encoder_depth", "bc_rank",
    "dt_min", "ff_mult", "batch_size", "random_percentage", "epoch", "clip_eigs",
    "val_metric", "best_test_metric", "final_test_metric", "status",
]

# ---------------------------------------------------------------------------
# Grid suggestion
# ---------------------------------------------------------------------------

def suggest_from_grid(trial: optuna.Trial, grid: dict) -> dict:
    params = {}
    for name, spec in grid.items():
        kind = spec[0]
        if kind == "float_log":
            params[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
        elif kind == "float_step":
            params[name] = trial.suggest_float(name, spec[1], spec[2], step=spec[3])
        elif kind == "float":
            params[name] = trial.suggest_float(name, spec[1], spec[2])
        elif kind == "int":
            params[name] = trial.suggest_int(name, spec[1], spec[2])
        elif kind == "int_step":
            params[name] = trial.suggest_int(name, spec[1], spec[2], step=spec[3])
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec[1])
        else:
            raise ValueError(f"Unknown grid spec kind: {kind!r}")
    return params


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def append_result(results_file, row):
    file_exists = os.path.isfile(results_file)
    with open(results_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Objective factory
# ---------------------------------------------------------------------------

def make_objective(datasets, device, early_stop_patience, seeds, results_file,
                   wandb_project, use_random_drop, data_dir=None):
    primary_grid = SEARCH_GRIDS[datasets[0]]
    if len(datasets) > 1:
        for ds in datasets[1:]:
            if SEARCH_GRIDS.get(ds) != primary_grid:
                print(f"  WARN: grid for {ds!r} differs from {datasets[0]!r}; "
                      f"using {datasets[0]!r}'s grid for the shared Optuna space.")

    def objective(trial: optuna.Trial) -> float:
        p = suggest_from_grid(trial, primary_grid)

        lr                 = p["lr"]
        weight_decay       = p["weight_decay"]
        hidden_dim         = p["hidden_dim"]
        num_blocks         = p["num_layers"]
        ssm_blocks         = p["s5_init_blocks"]
        ssm_dim_multiplier = p["ssm_dim_multiplier"]
        ssm_size           = 2 * ssm_blocks * ssm_dim_multiplier
        discretisation     = p["discretisation"]
        drop_rate          = p["drop_rate"]
        learn_lambda       = p["learn_lambda"]
        bidir              = p["bidir"]
        encoder_depth      = p["encoder_depth"]
        lambda_encoder_depth = p["lambda_encoder_depth"]
        bc_rank            = p["bc_rank"]
        dt_min             = p["dt_min"]
        ff_mult            = p["ff_mult"]
        batch_size         = p["batch_size"]
        random_percentage  = p["random_percentage"]
        epoch              = p["epoch"]
        clip_eigs          = p["clip_eigs"]

        print(
            f"\n--- Trial {trial.number} ---\n"
            f"  lr={lr:.2e}  wd={weight_decay}  hidden={hidden_dim}  layers={num_blocks}  "
            f"ssm_size={ssm_size}(blocks={ssm_blocks}×2×{ssm_dim_multiplier})  "
            f"disc={discretisation!r}  drop={drop_rate:.1f}  learn_lambda={learn_lambda!r}  "
            f"bidir={bidir}  enc={encoder_depth}  lam_enc={lambda_encoder_depth}  "
            f"bc_rank={bc_rank}  dt_min={dt_min}  ff_mult={ff_mult}  "
            f"batch={batch_size}  keep={random_percentage}  clip_eigs={clip_eigs}"
        )

        val_metrics: list[float] = []

        for dataset in datasets:
            for s in seeds:
                wandb_config = {
                    "trial": trial.number, "dataset": dataset, "seed": s,
                    "lr": lr, "weight_decay": weight_decay,
                    "hidden_dim": hidden_dim, "num_layers": num_blocks,
                    "ssm_size": ssm_size, "s5_init_blocks": ssm_blocks,
                    "ssm_dim_multiplier": ssm_dim_multiplier,
                    "discretisation": discretisation, "drop_rate": drop_rate,
                    "learn_lambda": learn_lambda, "bidir": bidir,
                    "encoder_depth": encoder_depth,
                    "lambda_encoder_depth": lambda_encoder_depth,
                    "bc_rank": bc_rank, "dt_min": dt_min, "ff_mult": ff_mult,
                    "batch_size": batch_size, "random_percentage": random_percentage,
                    "epoch": epoch,
                    "clip_eigs": clip_eigs,
                    "early_stop_patience": early_stop_patience,
                    "lambda_re_mode": "input_dependent",
                    "lambda_im_mode": "lti",
                    "bc_mode": "input_dependent",
                }

                base_row = {
                    "trial": trial.number, "dataset": dataset, "seed": s,
                    "lr": lr, "weight_decay": weight_decay,
                    "hidden_dim": hidden_dim, "num_layers": num_blocks,
                    "s5_init_blocks": ssm_blocks, "ssm_dim_multiplier": ssm_dim_multiplier,
                    "ssm_size": ssm_size,
                    "discretisation": discretisation, "drop_rate": drop_rate,
                    "learn_lambda": learn_lambda, "bidir": bidir,
                    "encoder_depth": encoder_depth,
                    "lambda_encoder_depth": lambda_encoder_depth,
                    "bc_rank": bc_rank, "dt_min": dt_min, "ff_mult": ff_mult,
                    "batch_size": batch_size, "random_percentage": random_percentage,
                    "epoch": epoch, "clip_eigs": clip_eigs,
                }

                print(f"  RUN  {dataset} seed={s}  epoch={epoch}")
                run = wandb.init(
                    project=wandb_project,
                    name=f"{dataset}_t{trial.number}_s{s}",
                    config=wandb_config,
                    tags=[dataset, "run"],
                    reinit=True,
                    mode="offline",
                    settings=wandb.Settings(init_timeout=300),
                )

                try:
                    best_val, best_test, final_acc = train_trial(
                        dataset=dataset,
                        seed=s,
                        device=device,
                        data_dir=data_dir,
                        lr=lr,
                        weight_decay=weight_decay,
                        epoch=epoch,
                        early_stop_patience=early_stop_patience,
                        batch_size=batch_size,
                        use_random_drop=use_random_drop,
                        random_percentage=random_percentage,
                        hidden_dim=hidden_dim,
                        ssm_size=ssm_size,
                        ssm_blocks=ssm_blocks,
                        num_blocks=num_blocks,
                        discretization=discretisation,
                        drop_rate=drop_rate,
                        learn_lambda=learn_lambda,
                        bidir=bidir,
                        encoder_depth=encoder_depth,
                        lambda_encoder_depth=lambda_encoder_depth,
                        bc_rank=bc_rank,
                        dt_min=dt_min,
                        ff_mult=ff_mult,
                        clip_eigs=clip_eigs,
                    )

                    val_metrics.append(best_val)
                    print(f"    → best_val={best_val:.4f}  best_test={best_test:.4f}  final={final_acc:.4f}")
                    append_result(results_file, {
                        **base_row,
                        "val_metric": best_val,
                        "best_test_metric": best_test,
                        "final_test_metric": final_acc,
                        "status": "ok",
                    })
                    wandb.log({
                        "val_metric": best_val,
                        "best_test_metric": best_test,
                        "final_test_metric": final_acc,
                        "status_code": 0,
                    })
                    wandb.finish()

                except KeyboardInterrupt:
                    print("    Interrupted — pruning trial")
                    append_result(results_file, {**base_row,
                                                 "val_metric": "", "best_test_metric": "",
                                                 "final_test_metric": "", "status": "interrupted"})
                    wandb.log({"status_code": 3})
                    wandb.finish(exit_code=1)
                    raise optuna.TrialPruned()

                except Exception as e:
                    err_str = f"{type(e).__name__}: {e}"
                    is_oom = "out of memory" in err_str.lower()
                    label = "oom" if is_oom else "error"
                    print(f"    {label.upper()}:")
                    traceback.print_exc()
                    append_result(results_file, {**base_row,
                                                 "val_metric": "", "best_test_metric": "",
                                                 "final_test_metric": "", "status": label})
                    wandb.log({"status_code": 2})
                    wandb.finish()
                    raise optuna.TrialPruned()

        if not val_metrics:
            raise optuna.TrialPruned()

        agg = float(np.mean(val_metrics))
        trial.set_user_attr("val_metrics", val_metrics)
        trial.set_user_attr("mean_val_metric", agg)
        return agg

    return objective


# ---------------------------------------------------------------------------
# Main search loop
# ---------------------------------------------------------------------------

def run_hypersearch(
    datasets, device, early_stop_patience, num_trials, num_seeds, results_file,
    seed, wandb_project, study_name, storage, use_random_drop,
    n_startup_trials=None, data_dir=None,
):
    wandb.login(key="acb80b5313c2f79b59f413b1186edcb4749c9cae")

    seeds = REPO_SEEDS[:num_seeds]

    if n_startup_trials is None:
        n_startup_trials = DATASET_N_STARTUP.get(datasets[0], num_trials // 3)

    sampler = optuna.samplers.TPESampler(
        seed=seed, multivariate=True,
        n_startup_trials=n_startup_trials, constant_liar=True,
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )

    objective = make_objective(
        datasets=datasets,
        device=device,
        early_stop_patience=early_stop_patience,
        seeds=seeds,
        results_file=results_file,
        wandb_project=wandb_project,
        use_random_drop=use_random_drop,
        data_dir=data_dir,
    )

    print(f"Optuna study '{study_name}': {num_trials} trials, direction=maximize")
    print(f"Datasets: {datasets}  Seeds: {seeds}")
    print(f"Results → {results_file}\n")

    n_counted = 0
    while n_counted < num_trials:
        study.optimize(objective, n_trials=1)
        last = study.trials[-1]
        if last.state == optuna.trial.TrialState.COMPLETE:
            n_counted += 1
            print(f"  ↳ Trial {last.number} counted ({n_counted}/{num_trials})\n")
        else:
            print(f"  ↳ Trial {last.number} pruned/failed — not counted "
                  f"({n_counted}/{num_trials} so far)\n")

    print("\nSearch complete.")
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed:
        print(f"Best trial : #{study.best_trial.number}")
        print(f"Best value : {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")
    else:
        print("No completed trials.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search for TIDES on UEA datasets (RFormer)"
    )
    parser.add_argument("--datasets", nargs="+", default=["TSC_SelfRegulationSCP1"],
                        help="Dataset(s) to search over (must share the same grid)")
    parser.add_argument("--data_dir", default="data_dir",
                        help="Root data directory (passed to get_dataset_preprocess)")
    parser.add_argument("--early_stop_patience", type=int, default=30,
                        help="Early stopping patience in epochs (0 = disabled)")
    parser.add_argument("--num_trials", type=int, default=100,
                        help="Number of counted Optuna trials")
    parser.add_argument("--n_startup_trials", type=int, default=None,
                        help="Random trials before TPE starts. Defaults to num_trials // 3.")
    parser.add_argument("--num_seeds", type=int, default=1,
                        choices=range(1, len(REPO_SEEDS) + 1),
                        help="Number of seeds from REPO_SEEDS to average over")
    parser.add_argument("--results_file", default="hypersearch_results.csv")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for the TPE sampler")
    parser.add_argument("--wandb_project", default="tides-anon")
    parser.add_argument("--study_name", default="tides-anon",
                        help="Optuna study name (used with --storage to resume)")
    parser.add_argument("--storage", default=None,
                        help="Optuna storage URL, e.g. sqlite:///hypersearch.db")
    parser.add_argument("--no_random_drop", action="store_true",
                        help="Disable random dropping (use full time series)")
    parser.add_argument("--device", default=None,
                        help="PyTorch device override (default: cuda if available, else cpu)")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    run_hypersearch(
        datasets=args.datasets,
        device=device,
        early_stop_patience=args.early_stop_patience,
        num_trials=args.num_trials,
        num_seeds=args.num_seeds,
        results_file=args.results_file,
        seed=args.seed,
        wandb_project=args.wandb_project,
        study_name=args.study_name,
        storage=args.storage,
        use_random_drop=not args.no_random_drop,
        n_startup_trials=args.n_startup_trials,
        data_dir=args.data_dir,
    )
