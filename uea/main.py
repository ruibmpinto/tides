#!/usr/bin/env python3
import argparse
import os
import random
import sys
import time
import pprint
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from model_classification import DecoderTransformer
from lstm_classification import LSTM_Classification
from utils import get_dataset_preprocess, get_dataset, ComputeModelParams
from sig_utils import ComputeSignatures

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tides import TIDESClassifier, step_scale_from_indices



def parse_args():
    """Parses command-line arguments and loads defaults from a YAML config file."""
    prelim = argparse.ArgumentParser(add_help=False)
    prelim.add_argument(
        '--config', '-c', type=str,
        help='Path to a YAML config file with default argument values'
    )
    args, remaining_argv = prelim.parse_known_args()

    parser = argparse.ArgumentParser(
        description="Train a Transformer or LSTM classifier on time series data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[prelim]
    )
    # Data and seeds
    parser.add_argument("--dataset", type=str, default="TSC_SelfRegulationSCP1", help="Dataset to use")
    parser.add_argument("--n_seeds", type=int, default=4, help="Number of random seeds to try")

    # Model and training
    parser.add_argument("--model", choices=["transformer", "lstm", "tides"], default="transformer", help="Model type")
    parser.add_argument("--epoch", type=int, default=300, help="Max training epochs")
    parser.add_argument("--early_stop_patience", type=int, default=15, help="Early stopping patience in epochs (0 = disabled)")
    parser.add_argument("--batch_size", type=int, default=20, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=-1, help="Evaluation batch size (-1 to use training batch_size)")
    parser.add_argument("--lr", type=float, default=0.00040788, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for optimizer")

    # Signature options
    parser.add_argument("--use_signatures", action="store_true", help="Enable signature features")
    parser.add_argument("--online_signature_calc", action="store_true", help="Compute signatures online per batch")
    parser.add_argument("--sig_win_len", type=int, default=50, help="Signature window length")
    parser.add_argument("--sig_level", type=int, default=2, help="Signature level")
    parser.add_argument("--num_windows", type=int, default=100, help="Number of windows for offline signatures")

    # Signature geometry
    parser.add_argument("--global_backward", action="store_true", help="Use global backward signature")
    parser.add_argument("--global_forward", action="store_true", help="Use global forward signature")
    parser.add_argument("--local_tight", action="store_true", help="Use local tight signature")
    parser.add_argument("--local_wide", action="store_true", help="Use local wide signature")
    parser.add_argument("--local_width", type=float, default=50.0, help="Local window width")

    # Data irregularity and time channel
    parser.add_argument("--irreg", action="store_true", help="Use irregular time intervals")
    parser.add_argument("--univariate", action="store_true", help="Use univariate signature")
    parser.add_argument("--add_time", action="store_true", help="Append time channel to inputs")

    # Random drop
    parser.add_argument("--use_random_drop", action="store_true", help="Enable random dropping of time points")
    parser.add_argument("--random_percentage", type=float, default=0.7, help="Fraction of points to keep when randomly dropping")

    # Transformer-specific
    parser.add_argument("--n_head", type=int, default=3, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of Transformer layers")
    parser.add_argument("--embedded_dim", type=int, default=10, help="Embedding dimension size")

    # Dropout probabilities
    parser.add_argument("--embd_pdrop", type=float, default=0.1, help="Embedding dropout probability")
    parser.add_argument("--attn_pdrop", type=float, default=0.1, help="Attention dropout probability")
    parser.add_argument("--resid_pdrop", type=float, default=0.1, help="Residual dropout probability")

    # TIDES-specific hyperparameters (used when --model tides)
    parser.add_argument("--tides_hidden", type=int, default=64, help="TIDES hidden dimension")
    parser.add_argument("--tides_ssm_size", type=int, default=64, help="TIDES total SSM state size")
    parser.add_argument("--tides_ssm_blocks", type=int, default=2, help="TIDES HiPPO diagonal blocks")
    parser.add_argument("--tides_num_blocks", type=int, default=2, help="TIDES number of SSM blocks")
    parser.add_argument("--tides_lambda_re_mode", choices=["lti", "input_dependent"], default="lti", help="TIDES Lambda real mode")
    parser.add_argument("--tides_lambda_im_mode", choices=["lti", "input_dependent"], default="lti", help="TIDES Lambda imaginary mode")
    parser.add_argument("--tides_bc_mode", choices=["lti", "input_dependent"], default="lti", help="TIDES B/C mode")
    parser.add_argument("--tides_bc_rank", type=int, default=8, help="TIDES B/C low-rank factor (0=full, <0=diagonal)")
    parser.add_argument("--tides_drop_rate", type=float, default=0.05, help="TIDES dropout rate inside blocks")
    parser.add_argument("--tides_learn_lambda", choices=["standard", "exp", "stable", "softplus"], default="standard", help="TIDES Lambda reparameterization")
    parser.add_argument("--tides_encoder_depth", type=int, default=1, help="TIDES GLU layers in input encoder")
    parser.add_argument("--tides_ff_mult", type=float, default=1.0, help="TIDES feed-forward expansion in GLU")
    parser.add_argument("--tides_dt_min", type=float, default=0.001, help="TIDES minimum log-step init value")
    parser.add_argument("--tides_dt_max", type=float, default=0.1, help="TIDES maximum log-step init value")
    parser.add_argument("--tides_discretization", choices=["zoh", "bilinear"], default="zoh", help="TIDES discretization method")
    parser.add_argument("--tides_clip_eigs", action="store_true", help="TIDES clip eigenvalues to left half-plane")
    parser.add_argument("--tides_bidir", action="store_true", help="TIDES bidirectional SSM")
    parser.add_argument("--tides_lambda_encoder_depth", type=int, default=0, help="TIDES GLU layers in lambda encoder")
    parser.add_argument("--tides_proj_norm", type=str, default="rmsnorm", help="TIDES projection norm (rmsnorm or none)")

    # Convergence criteria
    parser.add_argument("--epochs_for_convergence", type=int, default=10000, help="Epoch window for convergence check")
    parser.add_argument("--accuracy_for_convergence", type=float, default=0.6, help="Accuracy threshold for convergence")
    parser.add_argument("--std_for_convergence", type=float, default=0.05, help="Std deviation fraction for convergence")

    # Data splits and input size
    parser.add_argument("--test_size", type=float, default=0.3, help="Fraction of data for test set")
    parser.add_argument("--val_size", type=float, default=0.5, help="Fraction of test set for validation")
    parser.add_argument("--input_size", type=int, default=5, help="Number of input features")

    # Partition & query
    parser.add_argument("--v_partition", type=float, default=0.1, help="Validation partition ratio")
    parser.add_argument("--q_len", type=int, default=1, help="Query length for model")

    # Early stopping and subsequence
    parser.add_argument("--early_stop_ep", type=int, default=500, help="Epochs before early stopping")
    parser.add_argument("--sub_len", type=int, default=1, help="Subsequence length for data")

    # Warmup proportion
    parser.add_argument("--warmup_proportion", type=float, default=-1, help="Warmup proportion for LR schedule")

    # Optimizer and checkpoints
    parser.add_argument("--optimizer", type=str, default="Adam", help="Optimizer to use")
    parser.add_argument("--continue_training", action="store_true", help="Continue training from checkpoint")
    parser.add_argument("--save_all_epochs", action="store_true", help="Save model after every epoch")
    parser.add_argument("--pretrained_model_path", type=str, default="", help="Path to pretrained model")
    parser.add_argument("--downsampling", action="store_true", help="Downsample data during processing")
    parser.add_argument("--zero_shot_downsample", action="store_true", help="Apply zero-shot downsampling approach")

    # Misc flags
    parser.add_argument("--overlap", action="store_true", help="Overlap data windows")
    parser.add_argument("--scale_att", action="store_true", help="Scale attention weights")
    parser.add_argument("--sparse", action="store_true", help="Use sparse connections in model")

    # Load YAML defaults if provided
    if args.config:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)

    # Final parse (remaining_argv allows CLI overrides of YAML)
    return parser.parse_args(remaining_argv)


def compute_signature_inputs(inputs, t, indices_keep, config, device):
    """Computes signatures for a batch of inputs."""
    if config.add_time:
        inputs = torch.cat([t.repeat(inputs.size(0), 1, 1), inputs], dim=2)
    selected = inputs[:, indices_keep, :]
    x = np.linspace(0, inputs.size(1), inputs.size(1))[indices_keep]
    return ComputeSignatures(selected, x, config, device)


def calculate_accuracy(config, model, loader, num_classes, seq_len_orig, indices_keep, device,
                        step_scale=None):
    """Calculates model accuracy and avg cross-entropy loss on a given data loader."""
    model.eval()
    correct = total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    error_dist = {}
    t = torch.linspace(0, seq_len_orig, seq_len_orig).view(-1, 1).to(device) if config.add_time else None

    with torch.no_grad():
        for batch in loader:
            inputs = batch['input'].to(device)
            labels = batch['label'].to(device)
            if config.online_signature_calc:
                inputs = compute_signature_inputs(inputs, t, indices_keep, config, device)

            if step_scale is not None:
                inputs = inputs[:, indices_keep, :]
                outputs = model(inputs, step_scale=step_scale)
            else:
                outputs = model(inputs)
            total_loss += criterion(outputs, labels).item()
            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            wrong_preds = labels[preds != labels]
            if wrong_preds.numel() > 0:
                counts = torch.bincount(wrong_preds, minlength=num_classes)
                for i in range(len(counts)):
                    if counts[i] > 0:
                        error_dist[i] = error_dist.get(i, 0) + int(counts[i])

    return correct / total, total_loss / len(loader), error_dist


def create_model(config, num_features, seq_len, num_samples, num_classes, device):
    """Creates and returns the appropriate model based on config."""
    if config.model == 'transformer':
        return DecoderTransformer(
            config, input_dim=num_features, n_head=config.n_head,
            layer=config.num_layers, seq_num=num_samples, n_embd=config.embedded_dim,
            win_len=seq_len, num_classes=num_classes
        ).to(device)
    elif config.model == 'lstm':
        return LSTM_Classification(
            input_size=num_features, hidden_size=10,
            num_layers=100, batch_first=True, num_classes=num_classes
        ).to(device)
    elif config.model == 'tides':
        return TIDESClassifier(
            d_input=num_features,
            num_classes=num_classes,
            d_hidden=config.tides_hidden,
            ssm_size=config.tides_ssm_size,
            ssm_blocks=config.tides_ssm_blocks,
            num_blocks=config.tides_num_blocks,
            lambda_re_mode=config.tides_lambda_re_mode,
            lambda_im_mode=config.tides_lambda_im_mode,
            bc_mode=config.tides_bc_mode,
            bc_rank=config.tides_bc_rank,
            drop_rate=config.tides_drop_rate,
            learn_lambda=config.tides_learn_lambda,
            encoder_depth=config.tides_encoder_depth,
            lambda_encoder_depth=config.tides_lambda_encoder_depth,
            ff_mult=config.tides_ff_mult,
            dt_min=config.tides_dt_min,
            dt_max=config.tides_dt_max,
            discretization=config.tides_discretization,
            clip_eigs=config.tides_clip_eigs,
            bidir=config.tides_bidir,
            proj_norm=config.tides_proj_norm if config.tides_proj_norm != "none" else None,
        ).to(device)
    else:
        raise ValueError(f"Unsupported model: {config.model}")


def train_one_seed(config, seed, device):
    """Main training and evaluation loop for a single random seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    start_time = time.time()
    print("\n" + "="*50)
    print(f"Starting Training for Seed: {seed}")
    print("="*50)
    print("Configuration:")
    pprint.pprint(vars(config))

    if not config.online_signature_calc:
        # FIX 2: Standardized variable names to match their usage below.
        train_loader, val_loader, test_loader, seq_len_orig, num_classes, num_samples, num_features = \
            get_dataset_preprocess(config, seed, device)
        seq_len = seq_len_orig
    else:
        train_loader, val_loader, test_loader, seq_len_orig, num_classes, num_samples, num_features = \
            get_dataset(config, seed)
        num_features, seq_len = ComputeModelParams(seq_len_orig, num_features, config)

    print(f"\nDataset Info: Classes={num_classes}, Samples={num_samples}, Features={num_features}, SeqLen={seq_len}")

    indices = list(range(seq_len_orig))
    if config.use_random_drop:
        keep_indices = sorted(random.sample(indices, int(config.random_percentage * seq_len_orig)))
        if 0 not in keep_indices:
            keep_indices.insert(0, 0)
    else:
        keep_indices = indices

    # For TIDES: precompute per-timestep step sizes from the kept indices.
    # step_scale[i] = gap between kept index i and i-1 on the original grid.
    # When no drop is applied (all indices kept) every gap = 1, matching the
    # default uniform step_scale = 1.0 used by other models.
    tides_step_scale = (
        step_scale_from_indices(keep_indices, device=device)
        if config.model == "tides" else None
    )

    model = create_model(config, num_features, seq_len, num_samples, num_classes, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, config.optimizer)(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    t_for_sig = torch.linspace(0, seq_len_orig, seq_len_orig).view(-1, 1).to(device) if config.add_time else None

    best_val_loss = float("inf")
    best_val_acc = -1.0
    best_test_at_val = -1.0
    no_improve = 0
    patience = getattr(config, 'early_stop_patience', 15)
    for epoch in range(config.epoch):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        for batch in train_loader:
            inputs, labels = batch['input'].to(device), batch['label'].to(device)
            if config.online_signature_calc:
                inputs = compute_signature_inputs(inputs, t_for_sig, keep_indices, config, device)

            if tides_step_scale is not None:
                inputs = inputs[:, keep_indices, :]
                outputs = model(inputs, step_scale=tides_step_scale)
            else:
                outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_correct += (outputs.argmax(dim=1) == labels).sum().item()
            epoch_total += labels.size(0)

        avg_loss = epoch_loss / len(train_loader)
        train_acc = epoch_correct / epoch_total
        val_acc, val_loss, _ = calculate_accuracy(config, model, val_loader, num_classes, seq_len_orig, keep_indices, device,
                                                   step_scale=tides_step_scale)
        test_acc, test_loss, _ = calculate_accuracy(config, model, test_loader, num_classes, seq_len_orig, keep_indices, device,
                                                     step_scale=tides_step_scale)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_test_at_val = test_acc
            no_improve = 0
        else:
            no_improve += 1

        print(f"Epoch {epoch+1:03}/{config.epoch} | Loss: {avg_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Acc: {val_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}% | "
              f"Best Val Loss: {best_val_loss:.4f} | No improve: {no_improve}/{patience}")

        if wandb.run is not None:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
                "best_test_at_val": best_test_at_val,
            })

        if patience > 0 and no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} (patience={patience})")
            break

    print(f"\nTotal training time: {time.time() - start_time:.2f}s")
    final_acc, _, err_dist = calculate_accuracy(config, model, test_loader, num_classes, seq_len_orig, keep_indices, device,
                                             step_scale=tides_step_scale)
    return best_val_acc, best_test_at_val, final_acc


def train_trial(
    dataset,
    seed,
    device,
    *,
    data_dir=None,
    lr=1e-3,
    weight_decay=0.0,
    epoch=300,
    early_stop_patience=15,
    batch_size=20,
    use_random_drop=True,
    random_percentage=0.7,
    hidden_dim=64,
    ssm_size=64,
    ssm_blocks=8,
    num_blocks=2,
    lambda_re_mode="input_dependent",
    lambda_im_mode="lti",
    bc_mode="input_dependent",
    bidir=False,
    learn_lambda="standard",
    encoder_depth=1,
    lambda_encoder_depth=0,
    bc_rank=8,
    drop_rate=0.05,
    dt_min=0.001,
    dt_max=0.1,
    discretization="zoh",
    ff_mult=1.0,
    clip_eigs=False,
    proj_norm="rmsnorm",
):
    """Run one TIDES trial; returns (best_val_acc, best_test_at_val, final_acc)."""
    import types
    config = types.SimpleNamespace(
        dataset=dataset,
        data_dir=data_dir,
        model="tides",
        epoch=epoch,
        early_stop_patience=early_stop_patience,
        batch_size=batch_size,
        eval_batch_size=-1,
        lr=lr,
        weight_decay=weight_decay,
        optimizer="Adam",
        use_random_drop=use_random_drop,
        random_percentage=random_percentage,
        # signature / misc (unused for tides runs)
        use_signatures=False,
        online_signature_calc=False,
        add_time=False,
        univariate=False,
        global_backward=False,
        global_forward=False,
        local_tight=False,
        local_wide=False,
        local_width=50.0,
        sig_win_len=50,
        sig_level=2,
        num_windows=100,
        irreg=False,
        # transformer params (unused)
        n_head=3,
        num_layers=num_blocks,
        embedded_dim=10,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        # data splits
        test_size=0.3,
        val_size=0.5,
        input_size=5,
        v_partition=0.1,
        q_len=1,
        early_stop_ep=500,
        sub_len=1,
        warmup_proportion=-1,
        downsampling=False,
        zero_shot_downsample=False,
        overlap=False,
        scale_att=False,
        sparse=False,
        # convergence (unused)
        epochs_for_convergence=10000,
        accuracy_for_convergence=0.6,
        std_for_convergence=0.05,
        # TIDES
        tides_hidden=hidden_dim,
        tides_ssm_size=ssm_size,
        tides_ssm_blocks=ssm_blocks,
        tides_num_blocks=num_blocks,
        tides_lambda_re_mode=lambda_re_mode,
        tides_lambda_im_mode=lambda_im_mode,
        tides_bc_mode=bc_mode,
        tides_bc_rank=bc_rank,
        tides_drop_rate=drop_rate,
        tides_learn_lambda=learn_lambda,
        tides_encoder_depth=encoder_depth,
        tides_lambda_encoder_depth=lambda_encoder_depth,
        tides_ff_mult=ff_mult,
        tides_dt_min=dt_min,
        tides_dt_max=dt_max,
        tides_discretization=discretization,
        tides_clip_eigs=clip_eigs,
        tides_bidir=bidir,
        tides_proj_norm=proj_norm,
    )
    return train_one_seed(config, seed, device)


def main():
    """Main execution function."""
    config = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

        
    seeds = [42 + i for i in range(config.n_seeds)]
    results = []
    
    for s in seeds:
        _, _, acc = train_one_seed(config, s, device)
        print(f"\nSeed {s} -> Final Test Accuracy: {acc*100:.2f}%\n")
        results.append(acc)
        
    if len(results) > 1:
        mean_acc = np.mean(results)
        std_acc = np.std(results)
        print(f"\nAverage Accuracy over {len(results)} seeds: {mean_acc*100:.2f}% (± {std_acc*100:.2f}%)")


if __name__ == "__main__":
    main()