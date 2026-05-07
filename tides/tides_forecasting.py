"""TIDES forecasting model for the Physiome-ODE benchmark.

Wraps the core TIDES encoder with an output projection head.
Consumes TIDESBatch produced by tides_collate and returns per-position predictions.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .tides import TIDES


class TIDESForecastingModel(nn.Module):
    """TIDES encoder + linear output head for irregularly-sampled forecasting.

    Args:
        d_input:              Number of input/output channels (D).
        d_hidden:             Hidden dimension for TIDES blocks (H).
        ssm_size:             SSM state dimension (N before conj_sym halving).
        ssm_blocks:           Number of parallel SSM sub-blocks per layer.
        num_blocks:           Number of stacked TIDESBlock layers.
        encoder_depth:        Number of GLU residual layers in the input encoder.
        lambda_re_mode:       "lti" | "input_dependent"
        lambda_im_mode:       "lti" | "input_dependent"
        bc_mode:              "lti" | "input_dependent"
        lambda_encoder_depth: GLU depth for the lambda input-projection.

        learn_lambda:         "standard" | "exp" | "stable"
        discretization:       "zoh" | "bilinear"
        drop_rate:            Dropout probability.
        dt_min, dt_max:       Range for log_step initialization.
    """

    def __init__(
        self,
        d_input: int,
        d_hidden: int = 64,
        ssm_size: int = 32,
        ssm_blocks: int = 4,
        num_blocks: int = 4,
        encoder_depth: int = 1,
        lambda_re_mode: str = "input_dependent",
        lambda_im_mode: str = "lti",
        bc_mode: str = "input_dependent",
        lambda_encoder_depth: int = 0,
        bc_rank: int = 8,
        ff_mult: float = 1.0,
        learn_lambda: str = "standard",
        discretization: str = "zoh",
        drop_rate: float = 0.0,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        bidir: bool = False,
        conj_sym: bool = True,
        clip_eigs: bool = False,
        conv_kernel_size: int = 0,
        proj_init_method: str = "zeros",
        proj_norm: str = None,
    ):
        super().__init__()
        self.tides = TIDES(
            d_input=d_input,
            d_hidden=d_hidden,
            ssm_size=ssm_size,
            ssm_blocks=ssm_blocks,
            num_blocks=num_blocks,
            encoder_depth=encoder_depth,
            lambda_re_mode=lambda_re_mode,
            lambda_im_mode=lambda_im_mode,
            bc_mode=bc_mode,
            lambda_encoder_depth=lambda_encoder_depth,
            bc_rank=bc_rank,
            ff_mult=ff_mult,
            learn_lambda=learn_lambda,
            discretization=discretization,
            drop_rate=drop_rate,
            dt_min=dt_min,
            dt_max=dt_max,
            bidir=bidir,
            conj_sym=conj_sym,
            clip_eigs=clip_eigs,
            conv_kernel_size=conv_kernel_size,
            proj_init_method=proj_init_method,
            proj_norm=proj_norm,
        )
        self.output_proj = nn.Linear(d_hidden, d_input)

    def forward(self, values: Tensor, step_scale: Tensor) -> Tensor:
        """Forward pass.

        Args:
            values:     (B, L, D) — merged timeline values (zeros at target positions)
            step_scale: (B, L)    — per-step delta_t

        Returns:
            (B, L, D) — predictions at every position (extract target positions for loss)
        """
        h = self.tides(values, step_scale=step_scale)   # (B, L, d_hidden)
        return self.output_proj(h)                      # (B, L, D)
