"""Smoke test of the baseline wiring into the UEA pipeline.

Runs two epochs of each variant through train_trial, which exercises the
whole path: aeon loading from extract_path, de-duplication, the time
channel, the 70/15/15 split, model construction with the divisibility
reduction, the training loop and evaluation. Cheap enough to catch
integration mistakes before any sweep job is submitted.

Run on a GPU node from the uea directory:
    python mamba_baselines/smoke_sweep_wiring.py

Functions
---------
main
    Train each variant for two epochs and report the outcome.
"""

#
#                                                                       Modules
# =============================================================================
# Standard
import os
import sys
import time
import traceback
# Third-party
import torch
# Local
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import train_trial

#
# =============================================================================


def main():
    """Train each variant for two epochs and report the outcome."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = os.environ.get(
        'UEA_DATA_DIR',
        '/cluster/scratch/rbarreira/ssm_input_dependent/uea_data')
    dataset = 'TSC_EigenWorms'

    configurations = (
        ('mamba', {'hidden_dim': 16, 'num_layers': 1, 'd_state': 16,
                   'drop_rate': 0.05, 'd_conv': 2, 'expand': 1}),
        ('mamba2', {'hidden_dim': 16, 'num_layers': 1, 'd_state': 16,
                    'drop_rate': 0.05, 'headdim': 64}),
        ('mamba3', {'hidden_dim': 16, 'num_layers': 1, 'd_state': 16,
                    'drop_rate': 0.05, 'headdim': 64,
                    'mimo_mode': 'siso', 'rope_fraction': 0.5}),
        ('mamba3', {'hidden_dim': 16, 'num_layers': 1, 'd_state': 16,
                    'drop_rate': 0.05, 'headdim': 32,
                    'mimo_mode': 'mimo_rank4', 'rope_fraction': 0.5}),
    )

    failures = 0
    for model, params in configurations:
        label = f'{model}/{params.get("mimo_mode", "-")}'
        print(f'--- {label} ---')
        # headdim 64 with hidden_dim 16 and expand 2 gives d_inner 32, so
        # the reduction to a valid divisor is exercised here on purpose.
        try:
            start = time.time()
            best_val, best_test, final_acc = train_trial(
                dataset=dataset, seed=42, device=device, model=model,
                mamba_params=params, add_time=True, data_dir=data_dir,
                lr=1e-3, weight_decay=0.0, epoch=2,
                early_stop_patience=0, batch_size=10,
                use_random_drop=False, random_percentage=1.0)
            print(f'  val={best_val:.4f} test@val={best_test:.4f} '
                  f'final={final_acc:.4f} in {time.time() - start:.0f}s')
        except Exception as exc:
            print(f'  FAILED {type(exc).__name__}: {exc}')
            traceback.print_exc()
            failures += 1
        finally:
            torch.cuda.empty_cache()
        print('')

    print(f'=== {len(configurations) - failures}/'
          f'{len(configurations)} configurations ran ===')
    if failures:
        sys.exit(1)


if __name__ == '__main__':
    main()
