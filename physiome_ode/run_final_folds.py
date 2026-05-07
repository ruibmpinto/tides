"""
Run all 5 folds with the best hyperparams from an Optuna hypersearch study.
Outputs mean ± std test MSE directly comparable to the Physiome-ODE leaderboard.

Usage (after hypersearch_physio.py has completed):

    python run_final_folds.py \\
        --dataset hodgkin_huxley_1952_variant01 \\
        --storage sqlite:///results/hypersearch.db \\
        --study_name tides_physio_hodgkin_huxley_1952_variant01_f0

Or override hyperparams manually:

    python run_final_folds.py \\
        --dataset hodgkin_huxley_1952_variant01 \\
        --hidden_size 32 --ssm_blocks 2 --ssm_dim_mult 2 --num_blocks 3 \\
        --mode_combo input_dependent/lti/input_dependent --lr 5e-4 --weight_decay 1e-4 \\
        --batch_size 32 --drop_rate 0.0 --learn_lambda standard \\
        --discretization zoh --dt_min 0.001

Leaderboard reference (raw test MSE, mean ± std over 5 folds):
    hodgkin_huxley_1952_variant01 (HOD01):
        GRU-ODE-Bayes 0.851±0.118 | LinODENet 0.441±0.043 | CRU 0.409±0.049
        Neural Flows  0.701±0.077 | GraFITi   0.493±0.046 | GraFITi-C 0.609±0.056
"""

import argparse
import sys

import numpy as np
import wandb

from tides_train_fn import train_tides
from utils import IMTS_dataset  # noqa: F401 — pickle deserialization

WANDB_PROJECT = "tides-anon"


def load_best_params_from_optuna(storage: str, study_name: str) -> dict:
    import optuna
    study = optuna.load_study(study_name=study_name, storage=storage)
    p = study.best_params
    _parts = p["mode_combo"].split("/")
    if len(_parts) == 3:
        lambda_re_mode, lambda_im_mode, bc_mode = _parts
    else:
        lambda_re_mode, bc_mode = _parts
        lambda_im_mode = lambda_re_mode
    ssm_size = 2 * p["ssm_blocks"] * p["ssm_dim_mult"]
    return {
        "hidden_size":          p["hidden_size"],
        "ssm_size":             ssm_size,
        "ssm_blocks":           p["ssm_blocks"],
        "num_blocks":           p["num_blocks"],
        "encoder_depth":        p["encoder_depth"],
        "lambda_re_mode":       lambda_re_mode,
        "lambda_im_mode":       lambda_im_mode,
        "bc_mode":              bc_mode,
        "lambda_encoder_depth": p.get("lambda_encoder_depth", 1),
        "bc_rank":              p.get("bc_rank", 8),
        "learn_lambda":         p["learn_lambda"],
        "discretization":       p["discretization"],
        "drop_rate":            p["drop_rate"],
        "dt_min":               p["dt_min"],
        "conv_kernel_size":     p.get("conv_kernel_size", 0),
        "proj_init_method":     p.get("proj_init_method", "zeros"),
        "proj_norm":            p.get("proj_norm"),
        "lr":                   p["lr"],
        "weight_decay":         p["weight_decay"],
        "batch_size":           p["batch_size"],
    }


def main():
    parser = argparse.ArgumentParser(description="Run 5 folds with best TIDES config")
    parser.add_argument("--dataset",        required=True,  type=str)
    parser.add_argument("--data_base_path", default="../data/physiome_ode", type=str)
    parser.add_argument("--epochs",         default=45,    type=int)
    parser.add_argument("--early_stop",     default=10,    type=int)
    parser.add_argument("--saved_models_dir", default="saved_models", type=str)

    # Load from Optuna study
    parser.add_argument("--storage",    default=None, type=str, help="Optuna storage URL")
    parser.add_argument("--study_name", default=None, type=str, help="Optuna study name")

    # Or specify manually
    parser.add_argument("--hidden_size",           default=None, type=int)
    parser.add_argument("--ssm_blocks",            default=None, type=int)
    parser.add_argument("--ssm_dim_mult",          default=None, type=int)
    parser.add_argument("--num_blocks",            default=None, type=int)
    parser.add_argument("--encoder_depth",         default=1,    type=int)
    parser.add_argument("--mode_combo",            default="input_dependent/lti/input_dependent", type=str)
    parser.add_argument("--learn_lambda",          default="standard",      type=str)
    parser.add_argument("--discretization",        default="zoh",           type=str)
    parser.add_argument("--drop_rate",             default=0.0,  type=float)
    parser.add_argument("--dt_min",                default=0.001, type=float)
    parser.add_argument("--lr",                    default=1e-3,  type=float)
    parser.add_argument("--weight_decay",          default=1e-3,  type=float)
    parser.add_argument("--batch_size",            default=32,    type=int)
    parser.add_argument("--lambda_encoder_depth",  default=1,     type=int)
    parser.add_argument("--bc_rank",               default=8,     type=int)
    args = parser.parse_args()

    # ── Get hyperparams ───────────────────────────────────────────────────────
    if args.storage and args.study_name:
        print(f"Loading best params from study '{args.study_name}'...")
        hp = load_best_params_from_optuna(args.storage, args.study_name)
    else:
        if args.hidden_size is None or args.ssm_blocks is None or args.ssm_dim_mult is None or args.num_blocks is None:
            print("ERROR: provide either --storage+--study_name or all of "
                  "--hidden_size --ssm_blocks --ssm_dim_mult --num_blocks", file=sys.stderr)
            sys.exit(1)
        _parts = args.mode_combo.split("/")
        if len(_parts) == 3:
            lambda_re_mode, lambda_im_mode, bc_mode = _parts
        else:
            lambda_re_mode, bc_mode = _parts
            lambda_im_mode = lambda_re_mode
        hp = {
            "hidden_size":          args.hidden_size,
            "ssm_size":             2 * args.ssm_blocks * args.ssm_dim_mult,
            "ssm_blocks":           args.ssm_blocks,
            "num_blocks":           args.num_blocks,
            "encoder_depth":        args.encoder_depth,
            "lambda_re_mode":       lambda_re_mode,
            "lambda_im_mode":       lambda_im_mode,
            "bc_mode":              bc_mode,
            "lambda_encoder_depth": args.lambda_encoder_depth,
            "bc_rank":              args.bc_rank,
            "learn_lambda":         args.learn_lambda,
            "discretization":       args.discretization,
            "drop_rate":            args.drop_rate,
            "dt_min":               args.dt_min,
            "lr":                   args.lr,
            "weight_decay":         args.weight_decay,
            "batch_size":           args.batch_size,
        }

    wandb.login()

    print(f"\nDataset : {args.dataset}")
    print(f"Epochs  : {args.epochs}  early_stop={args.early_stop}")
    print(f"Config  : hidden={hp['hidden_size']}  ssm_size={hp['ssm_size']}"
          f"  blocks={hp['num_blocks']}  mode={hp['lambda_re_mode']}/{hp['lambda_im_mode']}/{hp['bc_mode']}"
          f"  lr={hp['lr']:.2e}  wd={hp['weight_decay']:.2e}  bs={hp['batch_size']}"
          f"  drop={hp['drop_rate']:.2f}  disc={hp['discretization']}"
          f"  learn_lambda={hp['learn_lambda']}  dt_min={hp['dt_min']}")
    print()

    # ── Run all 5 folds ───────────────────────────────────────────────────────
    val_losses  = []
    test_losses = []
    test_maes   = []

    for fold in range(5):
        print(f"── Fold {fold} ──────────────────────────────────")
        run = wandb.init(
            project=WANDB_PROJECT,
            name=f"{args.dataset}_final_f{fold}",
            config={**hp, "dataset": args.dataset, "fold": fold,
                    "epochs": args.epochs, "phase": "final_eval"},
            tags=[args.dataset, "final_eval"],
            reinit=True,
        )
        result = train_tides(
            dataset=args.dataset,
            fold=fold,
            data_base_path=args.data_base_path,
            epochs=args.epochs,
            early_stop_patience=args.early_stop,
            seed=fold,
            saved_models_dir=args.saved_models_dir,
            verbose=True,
            wandb_run=run,
            **hp,
        )
        wandb.log({
            "val_loss":  result["val_loss"],
            "test_loss": result["test_loss"],
            "test_mae":  result["test_mae"],
            "fold":      fold,
        })
        wandb.finish()

        val_losses.append(result["val_loss"])
        test_losses.append(result["test_loss"])
        test_maes.append(result["test_mae"])
        print(f"  fold {fold}: val={result['val_loss']:.4f}  "
              f"test_loss={result['test_loss']:.4f}  test_mae={result['test_mae']:.4f}  "
              f"params={result['num_params']:,}  "
              f"stopped_early={result['stopped_early']}")

    # ── Summary ───────────────────────────────────────────────────────────────
    # Log aggregate summary as a separate wandb run
    summary_run = wandb.init(
        project=WANDB_PROJECT,
        name=f"{args.dataset}_final_summary",
        config={**hp, "dataset": args.dataset, "phase": "final_summary"},
        tags=[args.dataset, "final_summary"],
        reinit=True,
    )
    wandb.log({
        "test_loss_mean": float(np.mean(test_losses)),
        "test_loss_std":  float(np.std(test_losses)),
        "test_mae_mean":  float(np.mean(test_maes)),
        "test_mae_std":   float(np.std(test_maes)),
        "val_loss_mean":  float(np.mean(val_losses)),
    })
    wandb.finish()

    print()
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"best_val_loss: {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")
    print(f"test_loss:     {np.mean(test_losses):.4f} ± {np.std(test_losses):.4f}   ← leaderboard number")
    print(f"test_mae:      {np.mean(test_maes):.4f} ± {np.std(test_maes):.4f}")
    print(f"Number of trainable parameters: {result['num_params']:,}")
    print()
    print("Leaderboard reference (hodgkin_huxley_1952_variant01 / HOD01):")
    print("  GRU-ODE-Bayes 0.851±0.118  LinODENet 0.441±0.043  CRU 0.409±0.049")
    print("  Neural Flows  0.701±0.077  GraFITi   0.493±0.046  GraFITi-C 0.609±0.056")


if __name__ == "__main__":
    main()
