"""
For a completed Optuna hypersearch study, pick the top-K trials (by fold-0 val
MSE) and run 5-fold final evaluation for each.

Outputs a ranked table + CSV, and logs everything to wandb project tides-anon.

Usage:
    python run_top_k_final.py \\
        --dataset hodgkin_huxley_1952_variant01 \\
        --storage sqlite:///results/hypersearch.db \\
        --study_name tides_physio_hypersearch_hodgkin_huxley_1952_variant01_f0 \\
        --top_k 5

Results are saved to:
    <results_dir>/<dataset>/top_k_results.csv
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import optuna
import wandb
from optuna.trial import TrialState

from tides_train_fn import train_tides
from utils import IMTS_dataset  # noqa: F401 — pickle deserialization

WANDB_PROJECT = "tides-anon"

CSV_COLUMNS = [
    "rank", "trial_number", "dataset", "val_score_fold0",
    "hidden_size", "ssm_size", "ssm_blocks", "num_blocks", "encoder_depth",
    "lambda_re_mode", "lambda_im_mode", "bc_mode", "learn_lambda", "discretization",
    "drop_rate", "batch_size", "dt_min", "lr", "lr_factor", "weight_decay",
    "lambda_encoder_depth", "bc_rank",
    "bidir", "conj_sym", "clip_eigs", "warmup_epochs",
    "fold0_test", "fold1_test", "fold2_test", "fold3_test", "fold4_test",
    "test_mean", "test_std",
    "fold0_val", "fold1_val", "fold2_val", "fold3_val", "fold4_val",
    "val_mean",
    "num_params",
]


def extract_hp(params: dict) -> dict:
    _parts = params["mode_combo"].split("/")
    if len(_parts) == 3:
        lambda_re_mode, lambda_im_mode, bc_mode = _parts
    else:
        lambda_re_mode, bc_mode = _parts
        lambda_im_mode = lambda_re_mode
    ssm_size = 2 * params["ssm_blocks"] * params["ssm_dim_mult"]
    return {
        "hidden_size":          params["hidden_size"],
        "ssm_size":             ssm_size,
        "ssm_blocks":           params["ssm_blocks"],
        "num_blocks":           params["num_blocks"],
        "encoder_depth":        params.get("encoder_depth", 1),
        "lambda_re_mode":       lambda_re_mode,
        "lambda_im_mode":       lambda_im_mode,
        "bc_mode":              bc_mode,
        "lambda_encoder_depth": params.get("lambda_encoder_depth", 1),
        "bc_rank":              params.get("bc_rank", 8),
        "learn_lambda":         params["learn_lambda"],
        "discretization":       params["discretization"],
        "drop_rate":            params["drop_rate"],
        "dt_min":               params["dt_min"],
        "lr":                   params["lr"],
        "lr_factor":            params.get("lr_factor", 1),
        "weight_decay":         params["weight_decay"],
        "batch_size":           params["batch_size"],
        "bidir":                params.get("bidir", False),
        "conj_sym":             params.get("conj_sym", True),
        "clip_eigs":            params.get("clip_eigs", False),
        "warmup_epochs":        params.get("warmup_epochs", 0),
        "conv_kernel_size":     params.get("conv_kernel_size", 0),
        "proj_init_method":     params.get("proj_init_method", "zeros"),
        "proj_norm":            params.get("proj_norm"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",        required=True,  type=str)
    parser.add_argument("--storage",        required=True,  type=str)
    parser.add_argument("--study_name",     required=True,  type=str)
    parser.add_argument("--top_k",          default=3,      type=int)
    parser.add_argument("--data_base_path", default="../data/physiome_ode", type=str)
    parser.add_argument("--epochs",         default=45,     type=int)
    parser.add_argument("--early_stop",     default=10,     type=int)
    parser.add_argument("--results_dir",    default="results/tides_physio_hypersearch", type=str)
    args = parser.parse_args()

    out_dir = os.path.join(args.results_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    saved_models_dir = os.path.join(out_dir, "models")
    os.makedirs(saved_models_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "top_k_results.csv")

    # ── Load study ────────────────────────────────────────────────────────────
    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not completed:
        print(f"ERROR: no completed trials in study '{args.study_name}'", file=sys.stderr)
        sys.exit(1)

    top_trials = sorted(completed, key=lambda t: t.value)[: args.top_k]
    print(f"\nDataset : {args.dataset}")
    print(f"Study   : {args.study_name}  ({len(completed)} completed trials)")
    print(f"Top-{args.top_k} val scores: {[f'{t.value:.4f}' for t in top_trials]}")

    csv_rows = []

    for rank, trial in enumerate(top_trials, start=1):
        hp = extract_hp(trial.params)
        print(f"\n{'='*60}")
        print(f"Rank {rank} | Trial #{trial.number} | fold-0 val={trial.value:.4f}")
        print(f"  {hp}")

        fold_test  = []
        fold_val   = []
        num_params = None

        for fold in range(5):
            print(f"  -- fold {fold} --")

            result = train_tides(
                dataset=args.dataset,
                fold=fold,
                data_base_path=args.data_base_path,
                epochs=args.epochs,
                early_stop_patience=args.early_stop,
                seed=fold,
                saved_models_dir=saved_models_dir,
                verbose=True,
                **hp,
            )

            fold_test.append(result["test_loss"])
            fold_val.append(result["val_loss"])
            if num_params is None:
                num_params = result["num_params"]

            print(f"    fold {fold}: val={result['val_loss']:.4f}  "
                  f"test={result['test_loss']:.4f}  mae={result['test_mae']:.4f}  "
                  f"params={result['num_params']:,}  early={result['stopped_early']}")

        test_mean = float(np.mean(fold_test))
        test_std  = float(np.std(fold_test))
        val_mean  = float(np.mean(fold_val))

        row = {
            "rank":             rank,
            "trial_number":     trial.number,
            "dataset":          args.dataset,
            "val_score_fold0":  trial.value,
            **{k: hp[k] for k in ["hidden_size", "ssm_size", "ssm_blocks",
                                   "num_blocks", "encoder_depth",
                                   "lambda_re_mode", "lambda_im_mode", "bc_mode", "learn_lambda",
                                   "discretization", "drop_rate", "batch_size",
                                   "dt_min", "lr", "lr_factor", "weight_decay",
                                   "lambda_encoder_depth", "bc_rank",
                                   "bidir", "conj_sym", "clip_eigs", "warmup_epochs"]},
            "fold0_test": fold_test[0], "fold1_test": fold_test[1],
            "fold2_test": fold_test[2], "fold3_test": fold_test[3],
            "fold4_test": fold_test[4],
            "test_mean":  test_mean,
            "test_std":   test_std,
            "fold0_val":  fold_val[0], "fold1_val": fold_val[1],
            "fold2_val":  fold_val[2], "fold3_val": fold_val[3],
            "fold4_val":  fold_val[4],
            "val_mean":   val_mean,
            "num_params": num_params,
        }
        csv_rows.append(row)

        # Write CSV incrementally
        write_header = not os.path.isfile(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            if write_header:
                w.writeheader()
            w.writerow(row)

    # ── Final leaderboard table ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Dataset: {args.dataset}")
    print(f"{'Rank':>4}  {'Trial':>6}  {'val0':>6}  {'test_mean':>10}  {'test_std':>9}  {'params':>8}")
    print("-" * 70)
    best_row = min(csv_rows, key=lambda r: r["test_mean"])
    for r in csv_rows:
        marker = " ◀ best" if r is best_row else ""
        print(f"  {r['rank']:>2}  #{r['trial_number']:>5}  "
              f"{r['val_score_fold0']:>6.4f}  "
              f"{r['test_mean']:>10.4f}  ±{r['test_std']:>7.4f}  "
              f"{r['num_params']:>8,}{marker}")

    print(f"\nBest config  test_loss: {best_row['test_mean']:.4f} ± {best_row['test_std']:.4f}")
    print(f"Results CSV : {csv_path}")
    print()
    print("Leaderboard reference (hodgkin_huxley_1952_variant01 / HOD01):")
    print("  GRU-ODE-Bayes 0.851±0.118  LinODENet 0.441±0.043  CRU 0.409±0.049")
    print("  Neural Flows  0.701±0.077  GraFITi   0.493±0.046  GraFITi-C 0.609±0.056")


if __name__ == "__main__":
    main()
