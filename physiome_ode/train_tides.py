"""Training script for TIDES on the Physiome-ODE benchmark.

Follows the structure of train_grafiti.py but uses tides_collate and
TIDESForecastingModel instead of GrATiF.

Functions
---------
MSE
    Masked mean squared error between target and prediction tensors.
MAE
    Masked mean absolute error between target and prediction tensors.
RMSE
    Masked root mean squared error between target and prediction
    tensors.
predict_fn
    Run the model on a batch and return (y_true, y_pred, mask).
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import argparse
import logging
import os
import random
import sys
import time
import warnings
from random import SystemRandom
# Third-party
import numpy as np
import torch
from torch import Tensor, jit
from torch.optim import AdamW
# Local
from tides.tides_collate import tides_collate
from tides.tides_forecasting import TIDESForecastingModel
from utils import get_data_loaders, IMTS_dataset  # noqa: F401
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Anonymous'
__credits__ = ['Anonymous']
__status__ = 'Development'
# =============================================================================
#
# =============================================================================
parser = argparse.ArgumentParser(
    description='Training Script for TIDES on the Physiome-ODE benchmark.')
parser.add_argument(
    '-q', '--quiet', default=False, const=True, help='quiet mode',
    nargs='?')
parser.add_argument(
    '-r', '--run_id', default=None, type=str, help='run_id')
parser.add_argument(
    '-e', '--epochs', default=45, type=int, help='maximum epochs')
parser.add_argument(
    '-es', '--early-stop', default=10, type=int,
    help='early stop patience')
parser.add_argument(
    '-f', '--fold', default=0, type=int, help='fold number')
parser.add_argument(
    '-bs', '--batch-size', default=32, type=int, help='batch size')
parser.add_argument(
    '-lr', '--learn-rate', default=0.001, type=float,
    help='learning rate')
parser.add_argument(
    '-b', '--betas', default=(0.9, 0.999), type=float, nargs=2,
    help='adam betas')
parser.add_argument(
    '-wd', '--weight-decay', default=0.001, type=float,
    help='weight decay')
parser.add_argument(
    '-hs', '--hidden-size', default=64, type=int,
    help='TIDES hidden dim')
parser.add_argument(
    '-sd', '--ssm-size', default=32, type=int, help='SSM state dim')
parser.add_argument(
    '-sb', '--ssm-blocks', default=4, type=int,
    help='SSM sub-blocks per layer')
parser.add_argument(
    '-nl', '--nlayers', default=4, type=int,
    help='number of TIDES blocks')
parser.add_argument(
    '-ed', '--encoder-depth', default=1, type=int,
    help='GLU encoder depth')
parser.add_argument(
    '-lrm', '--lambda-re-mode', default='input_dependent', type=str,
    help='lti|input_dependent')
parser.add_argument(
    '-lim', '--lambda-im-mode', default='lti', type=str,
    help='lti|input_dependent')
parser.add_argument(
    '-bm', '--bc-mode', default='input_dependent', type=str,
    help='lti|input_dependent')
parser.add_argument(
    '-led', '--lambda-enc-depth', default=0, type=int,
    help='lambda encoder depth')
parser.add_argument(
    '-bcr', '--bc-rank', default=8, type=int,
    help='BC low-rank head bottleneck rank')
parser.add_argument(
    '-ll', '--learn-lambda', default='standard', type=str,
    help='standard|exp|stable|softplus')
parser.add_argument(
    '-disc', '--discretization', default='zoh', type=str,
    help='zoh|bilinear')
parser.add_argument(
    '-do', '--drop-rate', default=0.0, type=float,
    help='dropout probability')
parser.add_argument(
    '-dtmin', '--dt-min', default=0.001, type=float,
    help='min log_step')
parser.add_argument(
    '-dtmax', '--dt-max', default=0.1, type=float,
    help='max log_step')
parser.add_argument(
    '-fh', '--forc-time', default=0.5, type=float,
    help='forecast horizon [0,1]')
parser.add_argument(
    '-ot', '--observation-time', default=0.5, type=float,
    help='conditioning range [0,1]')
parser.add_argument(
    '-dset', '--dataset', required=True, type=str,
    help='name of the dataset')
parser.add_argument(
    '-ck', '--conv-kernel-size', default=0, type=int,
    help='causal conv kernel size (0=off)')
parser.add_argument(
    '-pi', '--proj-init-method', default='zeros', type=str,
    help='zeros|random')
parser.add_argument(
    '-pn', '--proj-norm', default=None, type=str,
    help='None|rmsnorm')
parser.add_argument(
    '-n', '--note', default='', type=str, help='optional note')

ARGS = parser.parse_args()
print(' '.join(sys.argv))
experiment_id = int(SystemRandom().random() * 10000000)
print(ARGS, experiment_id)

torch.manual_seed(ARGS.fold)
random.seed(ARGS.fold)
np.random.seed(ARGS.fold)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

model_path = (
    'saved_models/TIDES_' + ARGS.dataset + '_' + str(experiment_id)
    + '.h5'
)
os.makedirs('saved_models', exist_ok=True)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

warnings.filterwarnings(
    action='ignore', category=UserWarning, module='torch')
logging.basicConfig(level=logging.WARN)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data
TRAIN_LOADER, VALID_LOADER, TEST_LOADER = get_data_loaders(
    fold=ARGS.fold,
    path=f'../data/physiome_ode/{ARGS.dataset}/',
    batch_size=ARGS.batch_size,
    collate_fn=tides_collate,
)
EVAL_LOADERS = {
    'train': TRAIN_LOADER,
    'valid': VALID_LOADER,
    'test': TEST_LOADER,
}


# =============================================================================
# Loss / metrics
# =============================================================================
def MSE(y, yhat, mask):
    """Masked mean squared error.

    Parameters
    ----------
    y : torch.Tensor
        Ground-truth tensor.
    yhat : torch.Tensor
        Predicted tensor with same shape as y.
    mask : torch.Tensor
        Boolean mask selecting valid positions.

    Returns
    -------
    err : torch.Tensor
        Scalar mean squared error over masked entries.
    """
    err = torch.mean((y[mask] - yhat[mask]) ** 2)
    return err
# =============================================================================


# =============================================================================
def MAE(y, yhat, mask):
    """Masked mean absolute error.

    Parameters
    ----------
    y : torch.Tensor
        Ground-truth tensor.
    yhat : torch.Tensor
        Predicted tensor with same shape as y.
    mask : torch.Tensor
        Boolean mask selecting valid positions.

    Returns
    -------
    err : torch.Tensor
        Scalar mean absolute error over masked entries.
    """
    return torch.mean(torch.abs(y[mask] - yhat[mask]))
# =============================================================================


# =============================================================================
def RMSE(y, yhat, mask):
    """Masked root mean squared error.

    Parameters
    ----------
    y : torch.Tensor
        Ground-truth tensor.
    yhat : torch.Tensor
        Predicted tensor with same shape as y.
    mask : torch.Tensor
        Float/boolean mask selecting valid positions.

    Returns
    -------
    err : torch.Tensor
        Scalar root mean squared error averaged over the batch.
    """
    err = torch.sqrt(
        torch.sum(mask * (y - yhat) ** 2, 1) / (torch.sum(mask, 1)))
    return torch.mean(err)
# =============================================================================


METRICS = {
    'RMSE': jit.script(RMSE),
    'MSE':  jit.script(MSE),
    'MAE':  jit.script(MAE),
}
LOSS = jit.script(MSE)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Model

# Infer d_input from first batch
batch = next(iter(TRAIN_LOADER))
d_input = batch.values.shape[-1]

MODEL_CONFIG = {
    'd_input':              d_input,
    'd_hidden':             ARGS.hidden_size,
    'ssm_size':             ARGS.ssm_size,
    'ssm_blocks':           ARGS.ssm_blocks,
    'num_blocks':           ARGS.nlayers,
    'encoder_depth':        ARGS.encoder_depth,
    'lambda_re_mode':       ARGS.lambda_re_mode,
    'lambda_im_mode':       ARGS.lambda_im_mode,
    'bc_mode':              ARGS.bc_mode,
    'lambda_encoder_depth': ARGS.lambda_enc_depth,
    'bc_rank':              ARGS.bc_rank,
    'learn_lambda':         ARGS.learn_lambda,
    'discretization':       ARGS.discretization,
    'drop_rate':            ARGS.drop_rate,
    'dt_min':               ARGS.dt_min,
    'dt_max':               ARGS.dt_max,
    'conv_kernel_size':     ARGS.conv_kernel_size,
    'proj_init_method':     ARGS.proj_init_method,
    'proj_norm':            ARGS.proj_norm,
}

MODEL = TIDESForecastingModel(**MODEL_CONFIG).to(DEVICE)


# =============================================================================
def predict_fn(model, batch):
    """Run the model on a batch and return targets, predictions, mask.

    Parameters
    ----------
    model : torch.nn.Module
        The TIDES forecasting model.
    batch : tuple
        Tuple of tensors ``(values, step_scale, target_mask, y_vals)``
        as produced by ``tides_collate``.

    Returns
    -------
    y_vals : torch.Tensor
        Ground-truth tensor on DEVICE with shape (B, L, D).
    yhat : torch.Tensor
        Predicted tensor on DEVICE with shape (B, L, D).
    target_mask : torch.Tensor
        Mask tensor on DEVICE with shape (B, L, D).
    """
    values, step_scale, target_mask, y_vals = (
        t.to(DEVICE) for t in batch)
    # model returns (B, L, D); we flatten target positions for scalar
    # loss
    yhat = model(values, step_scale)  # (B, L, D)
    return y_vals, yhat, target_mask  # all (B, L, D)
# =============================================================================


# Sanity check forward pass
MODEL.zero_grad(set_to_none=True)
Y, YHAT, MASK = predict_fn(MODEL, batch)
R = LOSS(Y, YHAT, MASK)
assert torch.isfinite(R).item(), 'Model collapsed on first forward pass!'
MODEL.zero_grad(set_to_none=True)

print(
    f'd_input={d_input}, params='
    f'{sum(p.numel() for p in MODEL.parameters() if p.requires_grad):,}'
)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Optimizer / scheduler
OPTIMIZER = AdamW(
    MODEL.parameters(),
    lr=ARGS.learn_rate,
    betas=ARGS.betas,
    weight_decay=ARGS.weight_decay,
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    OPTIMIZER, 'min', patience=10, factor=0.5, min_lr=1e-5,
)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Training loop
ovr_start_time = time.time()
es = False
best_val_loss = 1e10
total_num_batches = 0
epoch_times = []
early_stop = 0

for epoch in range(1, ARGS.epochs + 1):
    MODEL.train()
    loss_list = []
    start_time = time.time()

    for batch in TRAIN_LOADER:
        total_num_batches += 1
        OPTIMIZER.zero_grad()
        Y, YHAT, MASK = predict_fn(MODEL, batch)
        R = LOSS(Y, YHAT, MASK)
        assert torch.isfinite(R).item(), 'Model Collapsed!'
        loss_list.append([R])
        R.backward()
        OPTIMIZER.step()

    epoch_time = time.time()
    train_loss = torch.mean(torch.Tensor(loss_list))

    MODEL.eval()
    loss_list = []
    count = 0
    with torch.no_grad():
        for batch in VALID_LOADER:
            total_num_batches += 1
            Y, YHAT, MASK = predict_fn(MODEL, batch)
            R = LOSS(Y, YHAT, MASK)
            loss_list.append([R * MASK.sum()])
            count += MASK.sum()

    val_loss = torch.sum(torch.Tensor(loss_list).to(DEVICE) / count)
    epoch_times.append(epoch_time - start_time)
    print(
        f'{epoch:3d}  Train: {train_loss.item():.4f}  '
        f'Val: {val_loss.item():.4f}'
        f'  Time: {epoch_times[-1]:.2f}s'
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(
            {
                'args':                 ARGS,
                'epoch':                epoch,
                'model_config':         MODEL_CONFIG,
                'state_dict':           MODEL.state_dict(),
                'optimizer_state_dict': OPTIMIZER.state_dict(),
                'loss':                 train_loss,
            },
            model_path,
        )
        early_stop = 0
    else:
        early_stop += 1

    if early_stop == ARGS.early_stop:
        print(
            f'Early stopping — no improvement for '
            f'{ARGS.early_stop} epochs'
        )
        es = True

    scheduler.step(val_loss)

    if (epoch == ARGS.epochs) or es:
        print(f'tot_train_time: {time.time() - ovr_start_time:.1f}s')
        chp = torch.load(model_path, weights_only=False)
        MODEL.load_state_dict(chp['state_dict'])

        MODEL.eval()
        loss_list = []
        mae_list = []
        count = 0
        with torch.no_grad():
            inf_start = time.time()
            for batch in TEST_LOADER:
                total_num_batches += 1
                Y, YHAT, MASK = predict_fn(MODEL, batch)
                R = LOSS(Y, YHAT, MASK)
                assert torch.isfinite(R).item(), 'Model Collapsed!'
                loss_list.append([R * MASK.sum()])
                mae_list.append([MAE(Y, YHAT, MASK) * MASK.sum()])
                count += MASK.sum()

        print(f'inference_time: {time.time() - inf_start:.2f}s')
        print(f'avg_epoch_time: {np.mean(epoch_times):.2f}s')
        test_loss = torch.sum(
            torch.Tensor(loss_list).to(DEVICE) / count)
        test_mae = torch.sum(
            torch.Tensor(mae_list).to(DEVICE) / count)
        num_params = sum(
            p.numel() for p in MODEL.parameters()
            if p.requires_grad)
        print(
            f'best_val_loss: {best_val_loss.item():.4f}  '
            f'test_loss: {test_loss.item():.4f}'
        )
        print(f'test_mae: {test_mae.item():.4f}')
        print(f'Number of trainable parameters: {num_params:,}')
        break