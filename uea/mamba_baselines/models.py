"""Mamba-1, Mamba-2 and Mamba-3 classifiers sharing one block structure.

The block is the one used for the published Mamba and S6 EigenWorms
baselines in the Log-NCDE repository -- LayerNorm, SSM layer, GELU,
dropout, GLU, dropout, residual -- with mean pooling and a linear head.
Keeping that block fixed means the only thing differing between the three
models is the SSM layer itself.

Mamba-3 runs through the PyTorch backend in mamba3_recurrence.py: its
MIMO path must, because the upstream TileLang kernel needs a Hopper GPU,
and its SISO path is routed there too so that a SISO-versus-MIMO
comparison is not confounded by a change of kernel or of arithmetic
precision.

Classes
-------
GatedLinearUnit
    Gated linear unit, as in the published baseline block.
MambaVariantBlock
    Normalisation, SSM layer, activation, GLU and residual.
MambaVariantClassifier
    Encoder, stacked blocks, mean pooling and linear head.

Functions
---------
largest_valid_headdim
    Largest divisor of d_inner not exceeding a requested head dimension.
build_ssm_layer
    Construct the SSM layer for one of the three model variants.
"""

#
#                                                                       Modules
# =============================================================================
# Standard
import os
import sys
# Third-party
import torch
# Local
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mamba3_recurrence import install_torch_backend

#
# =============================================================================


def largest_valid_headdim(d_inner, requested):
    """Largest divisor of d_inner not exceeding a requested head dim.

    Parameters
    ----------
    d_inner : int
        Inner width, expand * hidden_dim.
    requested : int
        Head dimension suggested by the search.

    Returns
    -------
    headdim : int
        Largest divisor of d_inner that is <= requested, falling back to
        1 when no larger divisor exists.

    Notes
    -----
    Mamba2 and Mamba3 both assert d_inner % headdim == 0. Reducing an
    invalid suggestion is preferred to pruning the trial: it keeps the
    Optuna search space static, which multivariate TPE handles much
    better than a conditional space.
    """
    headdim = min(requested, d_inner)
    while headdim > 1 and d_inner % headdim != 0:
        headdim -= 1
    return max(headdim, 1)
# -----------------------------------------------------------------------------


def build_ssm_layer(model_name, hidden_dim, d_state, params):
    """Construct the SSM layer for one of the three model variants.

    Parameters
    ----------
    model_name : {'mamba', 'mamba2', 'mamba3'}
        Which variant to build.
    hidden_dim : int
        Model width.
    d_state : int
        SSM state dimension.
    params : dict
        Variant-specific settings. Recognised keys: 'd_conv', 'expand',
        'headdim', 'mimo_mode', 'rope_fraction', 'chunk_size'.

    Returns
    -------
    layer : torch.nn.Module
        The SSM layer.
    resolved : dict
        Settings actually used, after divisibility reduction and any
        argument coupling. Recorded per trial so the sweep log reflects
        what ran rather than what was requested.
    """
    if model_name == 'mamba':
        from mamba_ssm.modules.mamba_simple import Mamba

        d_conv = int(params.get('d_conv', 4))
        expand = int(params.get('expand', 2))
        layer = Mamba(d_model=hidden_dim, d_state=d_state, d_conv=d_conv,
                      expand=expand)
        return layer, {'d_conv': d_conv, 'expand': expand}

    if model_name == 'mamba2':
        from mamba_ssm.modules.mamba2 import Mamba2

        expand = int(params.get('expand', 2))
        headdim = largest_valid_headdim(expand * hidden_dim,
                                        int(params.get('headdim', 64)))
        layer = Mamba2(d_model=hidden_dim, d_state=d_state, d_conv=4,
                       expand=expand, headdim=headdim, ngroups=1)
        return layer, {'expand': expand, 'headdim': headdim, 'd_conv': 4}

    if model_name == 'mamba3':
        from mamba_ssm.modules.mamba3 import Mamba3

        install_torch_backend(verbose=False, siso=True)

        expand = int(params.get('expand', 2))
        headdim = largest_valid_headdim(expand * hidden_dim,
                                        int(params.get('headdim', 64)))
        mimo_mode = params.get('mimo_mode', 'siso')
        is_mimo = mimo_mode != 'siso'
        mimo_rank = int(mimo_mode.replace('mimo_rank', '')) if is_mimo \
            else 4
        rope_fraction = float(params.get('rope_fraction', 0.5))

        # chunk_size only affects throughput, not the result (check C1),
        # so it is not searched. 64 // rank is the upstream advice.
        chunk_size = int(params.get('chunk_size', 64 // mimo_rank))

        # Upstream couples these: the fused pre-gate path asserts that
        # MIMO and the output-projection norm are both enabled.
        layer = Mamba3(d_model=hidden_dim, d_state=d_state, expand=expand,
                       headdim=headdim, ngroups=1,
                       rope_fraction=rope_fraction, is_mimo=is_mimo,
                       mimo_rank=mimo_rank, is_outproj_norm=is_mimo,
                       fuse_pregate_headwise_norm=is_mimo,
                       chunk_size=chunk_size)
        return layer, {'expand': expand, 'headdim': headdim,
                       'is_mimo': is_mimo,
                       'mimo_rank': mimo_rank if is_mimo else 1,
                       'rope_fraction': rope_fraction,
                       'chunk_size': chunk_size}

    raise RuntimeError(f'Unknown model {model_name!r}.')
# -----------------------------------------------------------------------------


# =============================================================================
class GatedLinearUnit(torch.nn.Module):
    """Gated linear unit, as in the published baseline block.

    Attributes
    ----------
    linear : torch.nn.Linear
        Projection to twice the input width.
    """
    def __init__(self, input_dim):
        """Constructor.

        Parameters
        ----------
        input_dim : int
            Input and output width.
        """
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, input_dim * 2)
    # -------------------------------------------------------------------------

    def forward(self, x):
        """Apply the gated linear unit.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, seqlen, input_dim).

        Returns
        -------
        out : torch.Tensor
            Same shape as the input.
        """
        out = self.linear(x)
        width = x.shape[2]
        return out[:, :, :width] * torch.sigmoid(out[:, :, width:])
# =============================================================================


# =============================================================================
class MambaVariantBlock(torch.nn.Module):
    """Normalisation, SSM layer, activation, GLU and residual.

    Attributes
    ----------
    norm : torch.nn.LayerNorm
        Pre-normalisation.
    ssm : torch.nn.Module
        The SSM layer for the chosen variant.
    glu : GatedLinearUnit
        Gated linear unit.
    activation : torch.nn.GELU
        Activation.
    dropout : torch.nn.Dropout
        Dropout applied after the activation and after the GLU.
    resolved : dict
        Settings actually used by the SSM layer.
    """
    def __init__(self, model_name, hidden_dim, d_state, drop_rate,
                 params):
        """Constructor.

        Parameters
        ----------
        model_name : {'mamba', 'mamba2', 'mamba3'}
            Which variant to build.
        hidden_dim : int
            Model width.
        d_state : int
            SSM state dimension.
        drop_rate : float
            Dropout probability. Searched, whereas the published baseline
            block hard-codes 0.1, because the grid this sweep mirrors
            searches its dropout.
        params : dict
            Variant-specific settings.
        """
        super().__init__()
        self.norm = torch.nn.LayerNorm(hidden_dim)
        self.ssm, self.resolved = build_ssm_layer(model_name, hidden_dim,
                                                  d_state, params)
        self.glu = GatedLinearUnit(hidden_dim)
        self.activation = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(drop_rate)
    # -------------------------------------------------------------------------

    def forward(self, x):
        """Apply the block.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, seqlen, hidden_dim).

        Returns
        -------
        out : torch.Tensor
            Same shape as the input.
        """
        skip = x
        x = self.norm(x)
        x = self.ssm(x)
        x = self.dropout(self.activation(x))
        x = self.glu(x)
        x = self.dropout(x)
        return x + skip
# =============================================================================


# =============================================================================
class MambaVariantClassifier(torch.nn.Module):
    """Encoder, stacked blocks, mean pooling and linear head.

    Attributes
    ----------
    linear_encoder : torch.nn.Linear
        Input projection.
    blocks : torch.nn.Sequential
        Stacked MambaVariantBlock instances.
    linear_decoder : torch.nn.Linear
        Classification head.
    resolved : dict
        Settings actually used by the first block.

    Notes
    -----
    The head returns logits, unlike the published block which applies a
    softmax, because the training loop uses cross entropy and that
    applies its own log softmax. Applying softmax twice would flatten the
    gradients.
    """
    def __init__(self, model_name, input_dim, output_dim, hidden_dim,
                 d_state, num_layers, drop_rate, params):
        """Constructor.

        Parameters
        ----------
        model_name : {'mamba', 'mamba2', 'mamba3'}
            Which variant to build.
        input_dim : int
            Number of input channels.
        output_dim : int
            Number of classes.
        hidden_dim : int
            Model width.
        d_state : int
            SSM state dimension.
        num_layers : int
            Number of blocks.
        drop_rate : float
            Dropout probability.
        params : dict
            Variant-specific settings.
        """
        super().__init__()
        self.linear_encoder = torch.nn.Linear(input_dim, hidden_dim)
        blocks = [MambaVariantBlock(model_name, hidden_dim, d_state,
                                    drop_rate, params)
                  for _ in range(num_layers)]
        self.blocks = torch.nn.Sequential(*blocks)
        self.linear_decoder = torch.nn.Linear(hidden_dim, output_dim)
        self.resolved = blocks[0].resolved
    # -------------------------------------------------------------------------

    def forward(self, x):
        """Classify a batch of sequences.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, seqlen, input_dim).

        Returns
        -------
        logits : torch.Tensor
            Shape (batch, output_dim).
        """
        x = self.linear_encoder(x)
        x = self.blocks(x)
        return self.linear_decoder(torch.mean(x, dim=1))
# =============================================================================
