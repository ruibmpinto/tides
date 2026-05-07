"""Toy forward-pass smoke tests for TIDES PyTorch feature combinations.

Notes
-----
Iterates over a collection of configuration dictionaries and runs a
single forward pass through the TIDES model for each, asserting that
the output tensor has the expected shape and contains only finite
values.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import sys
# Third-party
import torch
# Local
from tides.tides import TIDES
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Anonymous'
__credits__ = ['Anonymous']
__status__ = 'Development'
# =============================================================================
#
# =============================================================================
BATCH, L, D_INPUT, D_HIDDEN = 2, 16, 4, 16
SSM_SIZE, SSM_BLOCKS, NUM_BLOCKS = 32, 2, 2

COMMON = dict(
    d_input=D_INPUT,
    d_hidden=D_HIDDEN,
    ssm_size=SSM_SIZE,
    ssm_blocks=SSM_BLOCKS,
    num_blocks=NUM_BLOCKS,
    dt_min=0.001,
    dt_max=0.1,
    drop_rate=0.0,
    encoder_depth=1,
    lambda_encoder_depth=0,
    ff_mult=1.0,
)

CONFIGS = {
    # 1. Fully LTI baseline
    'lti_baseline': dict(
        lambda_re_mode='lti', lambda_im_mode='lti', bc_mode='lti',
        learn_lambda='standard', conj_sym=True,
    ),
    # 2. Input-dependent lambda_re, softplus
    'id_lambda_re_softplus': dict(
        lambda_re_mode='input_dependent', lambda_im_mode='lti',
        bc_mode='lti',
        learn_lambda='softplus', conj_sym=True,
    ),
    # 3. ID lambda both, exp
    'id_lambda_both_exp': dict(
        lambda_re_mode='input_dependent',
        lambda_im_mode='input_dependent', bc_mode='lti',
        learn_lambda='exp', conj_sym=False,
    ),
    # 4. ID BC diag
    'id_bc_diag': dict(
        lambda_re_mode='lti', lambda_im_mode='lti',
        bc_mode='input_dependent',
        learn_lambda='standard', bc_rank=-1, conj_sym=True,
    ),
    # 5. ID BC low-rank
    'id_bc_lowrank': dict(
        lambda_re_mode='lti', lambda_im_mode='lti',
        bc_mode='input_dependent',
        learn_lambda='standard', bc_rank=4, conj_sym=False,
    ),
    # 6. Full ID + conv + rmsnorm + random init + softplus
    'full_id_conv_norm_random': dict(
        lambda_re_mode='input_dependent',
        lambda_im_mode='input_dependent',
        bc_mode='input_dependent', learn_lambda='softplus',
        bc_rank=-1, conv_kernel_size=4,
        proj_init_method='random', proj_norm='rmsnorm',
        conj_sym=True,
    ),
    # 7. Full ID + conv + zeros + no norm
    'full_id_conv_zeros': dict(
        lambda_re_mode='input_dependent',
        lambda_im_mode='input_dependent',
        bc_mode='input_dependent', learn_lambda='softplus',
        bc_rank=-1, conv_kernel_size=4,
        proj_init_method='zeros', proj_norm=None, conj_sym=False,
    ),
    # 8. ID lambda_re + ID BC lowrank + conv + rmsnorm
    'id_lambda_bc_lowrank_conv_norm': dict(
        lambda_re_mode='input_dependent', lambda_im_mode='lti',
        bc_mode='input_dependent', learn_lambda='softplus',
        bc_rank=4, conv_kernel_size=3,
        proj_init_method='zeros', proj_norm='rmsnorm',
        conj_sym=True,
    ),
    # 9. Stable + bidir
    'stable_bidir': dict(
        lambda_re_mode='input_dependent', lambda_im_mode='lti',
        bc_mode='lti',
        learn_lambda='stable', bidir=True, conj_sym=True,
    ),
    # 10. With step_scale (irregular sampling)
    'full_id_step_scale': dict(
        lambda_re_mode='input_dependent', lambda_im_mode='lti',
        bc_mode='input_dependent', learn_lambda='softplus',
        bc_rank=-1, conv_kernel_size=4,
        proj_init_method='random', proj_norm='rmsnorm',
        conj_sym=True,
    ),
}


passed, failed = 0, 0
for name, cfg in CONFIGS.items():
    try:
        torch.manual_seed(42)
        model = TIDES(**{**COMMON, **cfg})
        model.eval()

        x = torch.randn(BATCH, L, D_INPUT)

        with torch.no_grad():
            if name == 'full_id_step_scale':
                step_scale = torch.rand(BATCH, L) * 0.1 + 0.01
                out = model(x, step_scale=step_scale)
            else:
                out = model(x)

        expected = (BATCH, L, D_HIDDEN)
        assert out.shape == expected, \
            f'shape {out.shape} != expected {expected}'
        assert torch.isfinite(out).all(), 'output contains NaN/Inf'

        print(
            f'  PASS  {name:40s}  shape={tuple(out.shape)}  '
            f'range=[{out.min():.4f}, {out.max():.4f}]')
        passed += 1
    except Exception as e:
        print(f'  FAIL  {name:40s}  {type(e).__name__}: {e}')
        failed += 1

print(f'\n{passed} passed, {failed} failed out of {passed + failed}')
sys.exit(1 if failed else 0)