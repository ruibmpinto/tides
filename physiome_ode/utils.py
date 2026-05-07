"""Utilities for irregular multivariate time series (IMTS) datasets.

Dataset adapter and data loader helpers used by the Physiome-ODE
training loop. Provides the synthetic IMTS dataset builder and
utilities to persist and load train/valid/test splits.

Classes
-------
Sample
    Named tuple representing a single sample of the data.
IMTS_dataset
    Synthetic IMTS dataset with Bernoulli masking and noise injection.

Functions
---------
create_dataloaders
    Build and persist train/valid/test IMTS datasets to disk.
get_data_loaders
    Load persisted datasets and wrap them in PyTorch DataLoaders.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import glob
import random
from typing import Any, NamedTuple
# Third-party
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence as pad
from torch.utils.data import DataLoader, Dataset
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'kloetergens (kloetergens@ismll.de)'
__credits__ = ['kloetergens', 'ChristianK37', ]
__status__ = 'Development'
# =============================================================================
#
# =============================================================================


# =============================================================================
class Sample(NamedTuple):
    """A single sample of the data.

    Attributes
    ----------
    key : int
        Sample identifier.
    inputs : tuple
        Tuple of input tensors for the model.
    targets : torch.Tensor
        Target tensor associated with the sample.
    """
    key: int
    inputs: tuple
    targets: Tensor
# =============================================================================


# =============================================================================
class IMTS_dataset(Dataset):
    """Synthetic IMTS dataset with Bernoulli masking and noise injection.

    Loads a collection of parquet time-series files, splits them into
    an observation window and a forecasting window, normalizes values,
    adds Gaussian noise, and randomly drops entries according to a
    sparsity level.

    Attributes
    ----------
    T : torch.Tensor
        Observation timestamps (batch, n_obs).
    TY : torch.Tensor
        Forecasting timestamps (batch, n_pred).
    X : torch.Tensor
        Observation values (batch, n_obs, n_features), NaN where
        unobserved.
    Y : torch.Tensor
        Forecasting values (batch, n_pred, n_features), NaN where
        unobserved.

    Methods
    -------
    __len__(self)
        Return the number of samples.
    __getitem__(self, idx)
        Return the Sample at position idx.
    """
    def __init__(self, files, ot, fh, fold, sparsity=0.9, noise=0.05):
        """Constructor.

        Parameters
        ----------
        files : list
            Parquet file paths, one per sample.
        ot : float
            Observation time fraction of the maximum time.
        fh : float
            Forecasting horizon fraction of the maximum time.
        fold : int
            Fold index used to seed the PyTorch RNG.
        sparsity : float, default=0.9
            Probability with which an entry is masked out.
        noise : float, default=0.05
            Magnitude of the Gaussian noise added to values.
        """
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load files and compute split windows
        torch.manual_seed(fold)
        T = []
        X = []
        TY = []
        Y = []
        T_max = max(pd.read_parquet(files[0])['t'])
        value_columns = list(pd.read_parquet(files[0]).columns)
        value_columns.remove('t')
        observation_time = T_max * ot
        forecasting_horizon = observation_time + (T_max * fh)
        for f in files:
            raw_TS = pd.read_parquet(f)
            T.append(
                raw_TS['t'].loc[raw_TS['t'] <= observation_time].values)
            X.append(
                raw_TS[value_columns].loc[
                    raw_TS['t'] <= observation_time].values)
            TY.append(
                raw_TS['t']
                .loc[
                    (raw_TS['t'] > observation_time)
                    & (raw_TS['t'] < forecasting_horizon)
                ]
                .values
            )
            Y.append(
                raw_TS[value_columns]
                .loc[
                    (raw_TS['t'] > observation_time)
                    & (raw_TS['t'] < forecasting_horizon)
                ]
                .values
            )

        T = torch.tensor(np.stack(T, axis=0)).type(torch.float32)
        X = torch.tensor(np.stack(X, axis=0)).type(torch.float32)
        TY = torch.tensor(np.stack(TY, axis=0)).type(torch.float32)
        Y = torch.tensor(np.stack(Y, axis=0)).type(torch.float32)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Normalize
        T = T / T_max
        TY = TY / T_max
        std_V = torch.std(X.reshape(-1, X.shape[-1]), dim=0)
        mean_V = torch.mean(X.reshape(-1, X.shape[-1]), dim=0)
        X = (X - mean_V) / std_V
        Y = (Y - mean_V) / std_V
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Apply noise
        X += torch.randn(X.shape) * 0.05
        Y += torch.randn(Y.shape) * 0.05
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Apply masking
        M = (torch.rand(X.shape) > sparsity).type(torch.bool)
        MY = (torch.rand(Y.shape) > sparsity).type(torch.bool)
        T_MASK = torch.sum(M, axis=-1) > 0
        TY_MASK = torch.sum(MY, axis=-1) > 0

        T = pad(
            [T[i, T_MASK[i]] for i in range(X.shape[0])],
            batch_first=True)
        TY = pad(
            [TY[i, TY_MASK[i]] for i in range(X.shape[0])],
            batch_first=True)
        X = pad(
            [X[i, T_MASK[i], :] for i in range(X.shape[0])],
            batch_first=True)
        Y = pad(
            [Y[i, TY_MASK[i], :] for i in range(X.shape[0])],
            batch_first=True)
        M = pad(
            [M[i, T_MASK[i], :] for i in range(X.shape[0])],
            batch_first=True)
        MY = pad(
            [MY[i, TY_MASK[i], :] for i in range(X.shape[0])],
            batch_first=True)
        X[~M] = torch.nan
        Y[~MY] = torch.nan

        self.T = T
        self.TY = TY
        self.X = X
        self.Y = Y
    # -------------------------------------------------------------------------
    def __len__(self):
        """Return the number of samples.

        Returns
        -------
        n_samples : int
            Number of samples in the dataset.
        """
        return self.X.shape[0]
    # -------------------------------------------------------------------------
    def __getitem__(self, idx):
        """Return the Sample at position idx.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        sample : Sample
            Sample with inputs (T, X, TY) and targets Y.
        """
        return Sample(
            key=idx,
            inputs=(self.T[idx], self.X[idx], self.TY[idx]),
            targets=self.Y[idx],
        )
# =============================================================================


# =============================================================================
def create_dataloaders(
        model, fold, observation_time, forecasting_horizon, sparsity):
    """Build and persist train/valid/test IMTS datasets to disk.

    Parameters
    ----------
    model : str
        Model name used to locate the source parquet directory.
    fold : int
        Fold index used to seed the shuffle and the torch RNG.
    observation_time : float
        Observation time fraction of the maximum time.
    forecasting_horizon : float
        Forecasting horizon fraction of the maximum time.
    sparsity : float
        Probability with which an entry is masked out.
    """
    path = f'benchmark_datasets/{model}'
    files = glob.glob(f'{path}/*.parquet')
    random.seed(fold)
    random.shuffle(files)
    train_dataset = IMTS_dataset(
        files=files[: int(len(files) * 0.7)],
        ot=observation_time,
        fh=forecasting_horizon,
        sparsity=sparsity,
        fold=fold,
    )
    valid_dataset = IMTS_dataset(
        files=files[int(len(files) * 0.7): int(len(files) * 0.8)],
        ot=observation_time,
        fh=forecasting_horizon,
        sparsity=sparsity,
        fold=fold,
    )
    test_dataset = IMTS_dataset(
        files=files[int(len(files) * 0.8):],
        ot=observation_time,
        fh=forecasting_horizon,
        sparsity=sparsity,
        fold=fold,
    )
    out_path = f'IMTS_benchmark_datasets/{sparsity}/{model}/{fold}'
    torch.save(train_dataset, f'{out_path}/train.pt')
    torch.save(valid_dataset, f'{out_path}/valid.pt')
    torch.save(test_dataset, f'{out_path}/test.pt')
# =============================================================================


# =============================================================================
def get_data_loaders(path, fold, batch_size, collate_fn):
    """Load persisted datasets and wrap them in PyTorch DataLoaders.

    Parameters
    ----------
    path : str
        Base directory containing the fold subdirectory.
    fold : int
        Fold index subdirectory.
    batch_size : int
        Batch size used for the train and valid loaders.
    collate_fn : callable
        Collate function forwarded to each DataLoader.

    Returns
    -------
    TRAIN_LOADER : torch.utils.data.DataLoader
        DataLoader for the training split.
    VALID_LOADER : torch.utils.data.DataLoader
        DataLoader for the validation split.
    TEST_LOADER : torch.utils.data.DataLoader
        DataLoader for the test split (batch_size=32).
    """
    train_dataset = torch.load(
        f'{path}/{fold}/train.pt', weights_only=False)
    valid_dataset = torch.load(
        f'{path}/{fold}/valid.pt', weights_only=False)
    test_dataset = torch.load(
        f'{path}/{fold}/test.pt', weights_only=False)

    TRAIN_LOADER = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn
    )
    VALID_LOADER = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn
    )
    TEST_LOADER = DataLoader(
        test_dataset, batch_size=32, shuffle=False,
        collate_fn=collate_fn
    )
    return TRAIN_LOADER, VALID_LOADER, TEST_LOADER
# =============================================================================
