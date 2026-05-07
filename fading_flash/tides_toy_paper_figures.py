#!/usr/bin/env python3
"""Generate NeurIPS camera-ready figures for the TIDES paper.

Usage:
    python tides_toy_paper_figures.py          # use cache if available
    python tides_toy_paper_figures.py --retrain  # force retrain
"""

import argparse
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─── Color-blind safe palette (Okabe-Ito) ─────────────────────────────────────
COLORS = {
    'S5':              '#0072B2',   # blue
    'Mamba surrogate': '#CC0000',   # red
    'TIDES':           '#009E73',   # bluish green
    'TIDES (Λ-only)':  '#17BECF',  # turquoise
    'Real Mamba':      '#9467BD',   # purple
}
MARKERS = {
    'S5':              '^',
    'Mamba surrogate': 's',
    'TIDES':           'o',
    'TIDES (Λ-only)':  '*',
    'Real Mamba':      'D',
}
DISPLAY_NAMES = {
    'Mamba surrogate': r'Mamba$_{\mathrm{S}}$',
}
ZONE_FILL  = ['#cde6f7', '#fbe5c8', '#f7c8c8']   # background zone overlays
ZONE_LABEL = ['Slow (λ=1.0)', 'Medium (λ=1.5)', 'Fast (λ=2.0)']

# ─── Global matplotlib style ──────────────────────────────────────────────────
_FS = 9   # base font size — ≥8pt at final NeurIPS column width
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.25,
    'font.size': _FS,
    'axes.labelsize': _FS,
    'axes.titlesize': _FS,
    'legend.fontsize': _FS - 1,
    'xtick.labelsize': _FS - 1,
    'ytick.labelsize': _FS - 1,
    'pdf.fonttype': 42,   # embed fonts as Type 1 (not bitmap) in PDF
    'ps.fonttype': 42,
})

# ─── Task constants ───────────────────────────────────────────────────────────
RATE_LEVELS = torch.tensor([1.0, 1.5, 2.0])
N_RATES = 3
L = 40
D_IN, D_OUT = 4, 1
DT_GRID = np.array([0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0])

CACHE_DIR = Path('./cache')
CACHE_DIR.mkdir(exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# DATA GENERATORS  (copied verbatim from fading_flash_experiment.py)
# ═════════════════════════════════════════════════════════════════════════════

def make_one_sequence(n_flashes=3, dt_val=1.0, n_zones=3, seed=None):
    if seed is not None:
        np.random.seed(seed); torch.manual_seed(seed)
    pixels = torch.zeros(L)
    positions = np.random.choice(L, size=n_flashes, replace=False)
    pixels[positions] = 1.0
    if n_zones > 1:
        boundaries = sorted(np.random.choice(range(4, L - 4), size=n_zones - 1, replace=False))
    else:
        boundaries = []
    zone_spans = [0] + list(boundaries) + [L]
    rate_idx = torch.zeros(L, dtype=torch.long)
    prev = -1
    for i in range(n_zones):
        r = np.random.randint(N_RATES)
        while r == prev:
            r = np.random.randint(N_RATES)
        rate_idx[zone_spans[i]:zone_spans[i+1]] = r
        prev = r
    rates = RATE_LEVELS[rate_idx]
    alpha = torch.exp(-rates * dt_val)
    beta  = (1 - alpha) / rates
    y, h = torch.zeros(L), torch.tensor(0.0)
    for j in range(L):
        h = alpha[j] * h + beta[j] * pixels[j]
        y[j] = h
    return pixels, rate_idx, y


def sample_batch(batch_size, dt_range=(0.5, 1.5)):
    pixels = torch.zeros(batch_size, L)
    rate_idx = torch.zeros(batch_size, L, dtype=torch.long)
    for b in range(batch_size):
        n_px = np.random.randint(2, 5)
        pos = np.random.choice(L, size=n_px, replace=False)
        pixels[b, pos] = 1.0
        n_s = np.random.randint(2, 4)
        sw = sorted(np.random.choice(range(4, L-4), size=n_s-1, replace=False)) if n_s > 1 else []
        bnd = [0] + list(sw) + [L]
        prev = -1
        for s in range(n_s):
            r = np.random.randint(N_RATES)
            while r == prev: r = np.random.randint(N_RATES)
            rate_idx[b, bnd[s]:bnd[s+1]] = r
            prev = r
    rate_oh = F.one_hot(rate_idx, num_classes=N_RATES).float()
    x = torch.cat([pixels.unsqueeze(-1), rate_oh], dim=-1)
    dt = torch.empty(batch_size).uniform_(*dt_range)
    rates = RATE_LEVELS[rate_idx]
    alpha = torch.exp(-rates * dt.unsqueeze(-1))
    beta  = (1 - alpha) / rates
    y = torch.zeros(batch_size, L); h = torch.zeros(batch_size)
    for j in range(L):
        h = alpha[:, j] * h + beta[:, j] * pixels[:, j]
        y[:, j] = h
    return x, dt, y.unsqueeze(-1)


# ═════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS  (copied verbatim; IDS5 full → TIDES, IDS5 restricted → TIDES (Λ-only))
# ═════════════════════════════════════════════════════════════════════════════

def discretize_zoh_real(lambda_re, B, step):
    lambda_bar = torch.exp(lambda_re * step)
    scale = (lambda_bar - 1.0) / lambda_re
    return lambda_bar, scale.unsqueeze(-1) * B


class MinimalSSM(nn.Module):
    def __init__(self, d_in, d_out, P,
                 lambda_re_mode='lti', bc_mode='lti', step_mode='external'):
        super().__init__()
        self.d_in, self.d_out, self.P = d_in, d_out, P
        self.lambda_re_mode = lambda_re_mode
        self.bc_mode        = bc_mode
        self.step_mode      = step_mode

        init_decays = torch.linspace(0.5, 3.0, P)
        self.Lambda_re = nn.Parameter(-init_decays)
        self.B = nn.Parameter(torch.randn(P, d_in) / math.sqrt(d_in))
        self.C = nn.Parameter(torch.randn(d_out, P) / math.sqrt(P))
        self.D = nn.Parameter(torch.zeros(d_out, d_in))

        if step_mode == 'external':
            self.log_step = nn.Parameter(
                torch.empty(P).uniform_(math.log(0.1), math.log(1.0)))
        else:
            self.register_buffer('log_step', torch.zeros(P))

        if lambda_re_mode == 'input_dependent':
            self.lambda_re_proj = nn.Linear(d_in, P)
            with torch.no_grad():
                self.lambda_re_proj.weight.zero_()
                self.lambda_re_proj.bias.copy_(self.Lambda_re.data)
        if bc_mode == 'input_dependent':
            self.b_mod_proj = nn.Linear(d_in, P)
            self.c_mod_proj = nn.Linear(d_in, P)
            with torch.no_grad():
                for proj in [self.b_mod_proj, self.c_mod_proj]:
                    proj.weight.zero_(); proj.bias.fill_(1.0)
        if step_mode == 'input_dependent':
            self.step_proj = nn.Linear(d_in + 1, P)
            with torch.no_grad():
                self.step_proj.weight.zero_()
                self.step_proj.bias.fill_(0.5413)

    def forward(self, x, step_scale):
        B_, L_, _ = x.shape; P = self.P
        if self.lambda_re_mode == 'input_dependent':
            lambda_re = self.lambda_re_proj(x)
        else:
            lambda_re = self.Lambda_re.view(1, 1, P).expand(B_, L_, P)
        if self.bc_mode == 'input_dependent':
            b_mod = self.b_mod_proj(x); c_mod = self.c_mod_proj(x)
            B = b_mod.unsqueeze(-1) * self.B.view(1, 1, P, self.d_in)
            C = self.C.view(1, 1, self.d_out, P) * c_mod.unsqueeze(-2)
        else:
            B = self.B.view(1, 1, P, self.d_in).expand(B_, L_, P, self.d_in)
            C = self.C.view(1, 1, self.d_out, P).expand(B_, L_, self.d_out, P)
        if self.step_mode == 'input_dependent':
            dt_feat = step_scale.view(B_, 1, 1).expand(B_, L_, 1)
            x_dt = torch.cat([x, dt_feat], dim=-1)
            step = F.softplus(self.step_proj(x_dt))
        else:
            step_base = torch.exp(self.log_step)
            step = step_scale.view(B_, 1, 1) * step_base.view(1, 1, P)
            step = step.expand(B_, L_, P)
        lambda_bar, B_bar = discretize_zoh_real(lambda_re, B, step)
        h = torch.zeros(B_, P, device=x.device)
        ys = []
        for j in range(L_):
            Bu = torch.einsum('bpi,bi->bp', B_bar[:, j], x[:, j])
            h = lambda_bar[:, j] * h + Bu
            y_j = (torch.einsum('bop,bp->bo', C[:, j], h)
                   + torch.einsum('oi,bi->bo', self.D, x[:, j]))
            ys.append(y_j)
        return torch.stack(ys, dim=1)


class ModelWithEncoder(nn.Module):
    def __init__(self, H, P, lambda_re_mode, bc_mode, step_mode):
        super().__init__()
        self.encoder = nn.Linear(D_IN, H)
        self.ssm = MinimalSSM(H, D_OUT, P, lambda_re_mode, bc_mode, step_mode)
        self.P              = P
        self.lambda_re_mode = lambda_re_mode
        self.bc_mode        = bc_mode
        self.step_mode      = step_mode

    def forward(self, x, step_scale):
        return self.ssm(self.encoder(x), step_scale)


def make_S5():
    return ModelWithEncoder(H=15, P=3,
                            lambda_re_mode='lti', bc_mode='lti', step_mode='external')

def make_TIDES():                          # was IDS5 full
    return ModelWithEncoder(H=7, P=3,
                            lambda_re_mode='input_dependent', bc_mode='input_dependent',
                            step_mode='external')

def make_Mamba():
    return ModelWithEncoder(H=7, P=3,
                            lambda_re_mode='lti', bc_mode='input_dependent',
                            step_mode='input_dependent')

def make_TIDES_lambda():                   # was IDS5 restricted
    return ModelWithEncoder(H=11, P=3,
                            lambda_re_mode='input_dependent', bc_mode='lti',
                            step_mode='external')


class RealMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=3, d_conv=3, expand=1, dt_rank='auto'):
        super().__init__()
        self.d_model = d_model; self.d_state = d_state; self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        if dt_rank == 'auto':
            dt_rank = max(1, math.ceil(d_model / 16))
        self.dt_rank = dt_rank
        self.in_proj  = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d   = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                  groups=self.d_inner, padding=d_conv - 1, bias=True)
        self.x_proj   = nn.Linear(self.d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj  = nn.Linear(dt_rank, self.d_inner, bias=True)
        dt_min, dt_max = 0.001, 0.1
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
                       + math.log(dt_min))
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D     = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        B, L_, _ = x.shape; n = self.d_state
        xz = self.in_proj(x); x_path, res = xz.chunk(2, dim=-1)
        x_path = self.conv1d(x_path.transpose(1, 2))[:, :, :L_].transpose(1, 2)
        x_path = F.silu(x_path)
        A = -torch.exp(self.A_log); d_in = A.shape[0]
        x_dbl = self.x_proj(x_path)
        delta, B_inp, C_inp = torch.split(x_dbl, [self.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        deltaA  = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        deltaB_u = delta.unsqueeze(-1) * B_inp.unsqueeze(-2) * x_path.unsqueeze(-1)
        h = torch.zeros(B, d_in, n, dtype=x_path.dtype, device=x_path.device)
        ys = []
        for j in range(L_):
            h = deltaA[:, j] * h + deltaB_u[:, j]
            y_j = torch.einsum('bdn,bn->bd', h, C_inp[:, j]) + self.D * x_path[:, j]
            ys.append(y_j)
        return torch.stack(ys, dim=1) * F.silu(res)


class RealMambaForTask(nn.Module):
    def __init__(self, d_model=4, d_state=3, d_conv=3, expand=1):
        super().__init__()
        self.input_proj = nn.Linear(D_IN + 1, d_model)
        self.norm       = nn.LayerNorm(d_model)
        self.mixer      = RealMambaBlock(d_model, d_state=d_state, d_conv=d_conv,
                                         expand=expand, dt_rank=1)
        self.output_proj = nn.Linear(d_model, D_OUT)

    def forward(self, x, dt):
        B, L_, _ = x.shape
        dt_feat = dt.view(B, 1, 1).expand(B, L_, 1)
        xin = torch.cat([x, dt_feat], dim=-1)
        h = self.input_proj(xin)
        h = h + self.mixer(self.norm(h))
        return self.output_proj(h)


def make_Real_Mamba():
    return RealMambaForTask(d_model=4, d_state=3, d_conv=3, expand=1)


MODEL_FACTORIES = {
    'S5':              make_S5,
    'Mamba surrogate': make_Mamba,
    'TIDES':           make_TIDES,
    'TIDES (Λ-only)':  make_TIDES_lambda,
    'Real Mamba':      make_Real_Mamba,
}

# ═════════════════════════════════════════════════════════════════════════════
# TRAINING & EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

def train_one(model, name, n_steps=3000, lr=3e-3, sigma_in=0.0, sigma_out=0.0):
    device = next(model.parameters()).device
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for step in range(n_steps):
        x, dt, y = sample_batch(32)
        x, dt, y = x.to(device), dt.to(device), y.to(device)
        if sigma_in  > 0: x = x + sigma_in  * torch.randn_like(x)
        if sigma_out > 0: y = y + sigma_out * torch.randn_like(y)
        pred = model(x, dt)
        loss = F.mse_loss(pred, y)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if (step + 1) % 1000 == 0:
            print(f'  [{name:22s}] step {step+1}/{n_steps}'
                  f'  loss={np.mean(losses[-100:]):.5f}')
    return losses


@torch.no_grad()
def eval_at_dt(model, dt_val, n_batches=6, batch_size=64):
    device = next(model.parameters()).device
    model.eval(); losses = []
    for _ in range(n_batches):
        x, _, y = sample_batch(batch_size, dt_range=(dt_val, dt_val))
        x, y = x.to(device), y.to(device)
        pred = model(x, torch.full((batch_size,), dt_val, device=device))
        losses.append(F.mse_loss(pred, y).item())
    model.train()
    return float(np.mean(losses))


@torch.no_grad()
def target_var(dt_val, n=10, bs=128):
    ys = [sample_batch(bs, dt_range=(dt_val, dt_val))[2] for _ in range(n)]
    return torch.cat(ys).var().item()


# ═════════════════════════════════════════════════════════════════════════════
# PROBE / ANALYSIS UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def effective_lambda_per_zone(model, rate_c, dt_val):
    """Effective continuous-time λ for the best-matching state in `model` at `dt_val`."""
    step_base = torch.exp(model.ssm.log_step.data).numpy()
    x = torch.zeros(D_IN); x[1 + rate_c] = 1.0
    with torch.no_grad():
        h = model.encoder(x)
        lam_re = (model.ssm.lambda_re_proj(h).numpy()
                  if model.lambda_re_mode == 'input_dependent'
                  else model.ssm.Lambda_re.data.numpy())
        if model.step_mode == 'input_dependent':
            x_dt = torch.cat([h, torch.tensor([dt_val])])
            step = F.softplus(model.ssm.step_proj(x_dt)).numpy()
        else:
            step = step_base * dt_val
    lam_effs = -(lam_re * step) / dt_val
    target   = RATE_LEVELS[rate_c].item()
    return lam_effs[np.argmin(np.abs(lam_effs - target))]


def get_states(model, rate_c, dt_val=1.0, pixel_pos=5):
    """Run impulse-in-zone-c, return (L, P) state matrix. MinimalSSM models only."""
    pixels = torch.zeros(1, L); pixels[0, pixel_pos] = 1.0
    rate_idx_c = torch.full((1, L), rate_c, dtype=torch.long)
    rate_oh = F.one_hot(rate_idx_c, num_classes=N_RATES).float()
    x = torch.cat([pixels.unsqueeze(-1), rate_oh], dim=-1)
    with torch.no_grad():
        x_enc = model.encoder(x)
        ssm = model.ssm; B_, L_, _ = x_enc.shape; P = ssm.P
        lam_re = (ssm.lambda_re_proj(x_enc)
                  if ssm.lambda_re_mode == 'input_dependent'
                  else ssm.Lambda_re.view(1, 1, P).expand(B_, L_, P))
        if ssm.bc_mode == 'input_dependent':
            b_mod = ssm.b_mod_proj(x_enc)
            B = b_mod.unsqueeze(-1) * ssm.B.view(1, 1, P, ssm.d_in)
        else:
            B = ssm.B.view(1, 1, P, ssm.d_in).expand(B_, L_, P, ssm.d_in)
        if ssm.step_mode == 'input_dependent':
            dt_feat = torch.full((B_, L_, 1), dt_val)
            step = F.softplus(ssm.step_proj(torch.cat([x_enc, dt_feat], dim=-1)))
        else:
            step = (torch.tensor(dt_val).view(1, 1, 1) *
                    torch.exp(ssm.log_step).view(1, 1, P)).expand(B_, L_, P)
        lambda_bar = torch.exp(lam_re * step)
        B_bar = ((lambda_bar - 1.0) / lam_re).unsqueeze(-1) * B
        h = torch.zeros(B_, P); states = []
        for j in range(L_):
            h = lambda_bar[:, j] * h + torch.einsum('bpi,bi->bp', B_bar[:, j], x_enc[:, j])
            states.append(h.clone())
    return torch.stack(states, dim=1)[0].numpy()   # (L, P)


def gt_response(rate_c, dt_val=1.0, pixel_pos=5):
    lam   = RATE_LEVELS[rate_c].item()
    alpha = np.exp(-lam * dt_val); beta = (1 - alpha) / lam
    y = np.zeros(L); h = 0.0
    for j in range(L):
        h = alpha * h + (beta if j == pixel_pos else 0); y[j] = h
    return y


def r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0


def canonical_best_idx(model, rate_c):
    step_base = torch.exp(model.ssm.log_step.data).numpy()
    if model.lambda_re_mode == 'input_dependent':
        x = torch.zeros(D_IN); x[1 + rate_c] = 1.0
        with torch.no_grad():
            h = model.encoder(x)
            lam_re = model.ssm.lambda_re_proj(h).numpy()
    else:
        lam_re = model.ssm.Lambda_re.data.numpy()
    lam_eff = lam_re * step_base
    target  = -RATE_LEVELS[rate_c].item()
    return np.argmin(np.abs(lam_eff - target)), lam_eff


def compute_recon_stats(recon_models):
    canonical_err = {}; canonical_idx = {}
    onecomp = {};       full_pz = {};      shared = {}
    for name, m in recon_models.items():
        idxs, errs = [], []
        for c in range(N_RATES):
            bidx, lam_eff = canonical_best_idx(m, c)
            best = lam_eff[bidx]
            errs.append(100 * abs(best - (-RATE_LEVELS[c].item())) /
                        abs(-RATE_LEVELS[c].item()))
            idxs.append(bidx)
        canonical_idx[name] = idxs; canonical_err[name] = errs

        one, full = [], []
        for c in range(N_RATES):
            X   = get_states(m, c); y_gt = gt_response(c)
            X1  = X[:, [canonical_idx[name][c]]]
            w1, *_ = np.linalg.lstsq(X1, y_gt, rcond=None)
            one.append(r2(y_gt, X1 @ w1))
            wf, *_ = np.linalg.lstsq(X, y_gt, rcond=None)
            full.append(r2(y_gt, X @ wf))
        onecomp[name] = one; full_pz[name] = full

        Xs  = [get_states(m, c) for c in range(N_RATES)]
        ys  = [gt_response(c)   for c in range(N_RATES)]
        Xst = np.concatenate(Xs); yst = np.concatenate(ys)
        w, *_ = np.linalg.lstsq(Xst, yst, rcond=None)
        shared[name] = (r2(yst, Xst @ w), [r2(ys[c], Xs[c] @ w) for c in range(N_RATES)])

    return canonical_err, canonical_idx, onecomp, full_pz, shared


# ═════════════════════════════════════════════════════════════════════════════
# CACHE  (keyed on training config hash)
# ═════════════════════════════════════════════════════════════════════════════

TRAIN_CONFIG = {
    'n_steps': 3000, 'lr': 3e-3, 'seed': 0,
    'noisy_sigma_in': 0.2, 'noisy_sigma_out': 0.05,
}


def _hash(d):
    return hashlib.md5(str(sorted(d.items())).encode()).hexdigest()[:12]


def load_or_train(retrain=False):
    cache_path = CACHE_DIR / f'snapshot_{_hash(TRAIN_CONFIG)}.pt'

    if not retrain and cache_path.exists():
        print(f'Loading cache: {cache_path}')
        return torch.load(cache_path, weights_only=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training all models from scratch on {device}...')

    # ── 5 main models ──────────────────────────────────────────────────────
    models = {}
    for name, make in MODEL_FACTORIES.items():
        torch.manual_seed(0); np.random.seed(0)
        m = make().to(device); print(f'\n  Training {name}...')
        train_one(m, name)
        m.cpu().eval(); models[name] = m

    print('\nEvaluating OOD performance...')
    var_y   = [target_var(float(d)) for d in DT_GRID]
    results = {n: [eval_at_dt(m, float(d)) for d in DT_GRID]
               for n, m in models.items()}

    # ── noisy variants (TIDES and TIDES Λ-only only) ───────────────────────
    print('\nTraining noisy variants...')
    noisy_models = {}
    for name in ['TIDES', 'TIDES (Λ-only)']:
        torch.manual_seed(0); np.random.seed(0)
        m = MODEL_FACTORIES[name]().to(device); print(f'\n  Training {name} (noisy)...')
        train_one(m, f'{name} (noisy)',
                  sigma_in=TRAIN_CONFIG['noisy_sigma_in'],
                  sigma_out=TRAIN_CONFIG['noisy_sigma_out'])
        m.cpu().eval(); noisy_models[name] = m

    print('\nEvaluating noisy variants on clean data...')
    noisy_results = {n: [eval_at_dt(m, float(d)) for d in DT_GRID]
                     for n, m in noisy_models.items()}

    snap = {
        'state_dicts':       {n: m.state_dict() for n, m in models.items()},
        'noisy_state_dicts': {n: m.state_dict() for n, m in noisy_models.items()},
        'results':           results,
        'noisy_results':     noisy_results,
        'var_y':             var_y,
    }
    torch.save(snap, cache_path)
    print(f'\nCache saved: {cache_path}')
    return snap


def restore_models(snap):
    models = {}
    for name, make in MODEL_FACTORIES.items():
        m = make(); m.load_state_dict(snap['state_dicts'][name]); m.eval()
        models[name] = m
    noisy = {}
    for name in ['TIDES', 'TIDES (Λ-only)']:
        m = MODEL_FACTORIES[name]()
        m.load_state_dict(snap['noisy_state_dicts'][name]); m.eval()
        noisy[name] = m
    return models, noisy


# ═════════════════════════════════════════════════════════════════════════════
# SHARED PLOT HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _save(fig, name):
    path = f'{name}.pdf'
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def _draw_zones(ax, rate_idx, annotate=False):
    j0 = 0
    for j in range(1, L + 1):
        if j == L or rate_idx[j] != rate_idx[j-1]:
            z = rate_idx[j0].item()
            ax.axvspan(j0 - 0.5, j - 0.5, color=ZONE_FILL[z], alpha=0.7, zorder=0)
            if annotate:
                ax.text((j0 + j - 1) / 2, 0.93,
                        ZONE_LABEL[z].split()[0],      # "Slow" / "Medium" / "Fast"
                        ha='center', va='top',
                        transform=ax.get_xaxis_transform(),
                        fontsize=_FS - 1, fontweight='bold')
            j0 = j


def _shade_train(ax, label='Training Δt'):
    ax.axvspan(0.5, 1.5, color='gray', alpha=0.12, zorder=0, label=label)


def _rel_err(mse_list, var_y):
    return np.sqrt(np.array(mse_list) / np.array(var_y)) * 100


def _log_dt_ticks(ax):
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=5))
    ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=[2, 5], numticks=10))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE: fig_setup  ── single example sequence (setup illustration)
# ═════════════════════════════════════════════════════════════════════════════

def make_fig_setup():
    # Deterministic scene: three equal zones, one flash each, slow → medium → fast.
    ZONE_BOUNDS = [(0, 13), (13, 27), (27, 40)]
    FLASH_POS   = [3, 16, 30]

    rate_idx = torch.zeros(L, dtype=torch.long)
    for z, (lo, hi) in enumerate(ZONE_BOUNDS):
        rate_idx[lo:hi] = z

    pixels = torch.zeros(L)
    for p in FLASH_POS:
        pixels[p] = 1.0

    def _glow(dt_val):
        rates = RATE_LEVELS[rate_idx]
        alpha = torch.exp(-rates * dt_val)
        beta  = (1 - alpha) / rates
        y, h  = torch.zeros(L), torch.tensor(0.0)
        for j in range(L):
            h = alpha[j] * h + beta[j] * pixels[j]
            y[j] = h
        return y.numpy()

    y1  = _glow(1.0)
    y01 = _glow(0.1)

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(3.5, 3.6), sharex=True,
        gridspec_kw={'height_ratios': [1, 1.4, 1.4], 'hspace': 0.05})

    def _glow_panel(ax, y_np, label):
        _draw_zones(ax, rate_idx)
        for p in FLASH_POS:
            ax.plot(p, 0, 'v', color='#222222', ms=4, clip_on=True, zorder=3)
        ax.plot(np.arange(L), y_np, color='#222222', lw=1.5)
        ax.fill_between(np.arange(L), 0, y_np, color='#222222', alpha=0.14)
        ax.set_ylim(0, y_np.max() * 1.12)
        ax.set_yticks([])
        ax.set_ylabel(label, fontsize=_FS)
        ax.grid(False)

    # ── row 1: input flashes ──────────────────────────────────────────────────
    _draw_zones(ax1, rate_idx)
    for z, (lo, hi) in enumerate(ZONE_BOUNDS):
        mid = (lo + hi) / 2
        ax1.text(mid, 0.96, ZONE_LABEL[z].split()[0],
                 ha='center', va='top', transform=ax1.get_xaxis_transform(),
                 fontsize=_FS - 1, fontweight='bold')
        ax1.text(mid, 0.58, f'λ = {RATE_LEVELS[z].item():.1f}',
                 ha='center', va='top', transform=ax1.get_xaxis_transform(),
                 fontsize=_FS - 2, style='italic', color='#444444')
    for p in FLASH_POS:
        ax1.plot([p, p], [0, 0.82], color='#222222', lw=1.5)
        ax1.plot(p, 0.88, '*', color='#222222', ms=9, clip_on=True)
    ax1.set_ylim(0, 1.0)
    ax1.set_yticks([])
    ax1.set_ylabel('Input', fontsize=_FS)
    ax1.set_xlim(-0.5, L - 0.5)
    ax1.tick_params(axis='x', which='both', bottom=False)
    ax1.grid(False)

    # ── row 2: glow at dt = 1 ─────────────────────────────────────────────────
    _glow_panel(ax2, y1,  'Glow\n(Δt = 1)')
    ax2.tick_params(axis='x', which='both', bottom=False)

    # ── row 3: glow at dt = 0.1 ───────────────────────────────────────────────
    _glow_panel(ax3, y01, 'Glow\n(Δt = 0.1)')
    ax3.set_xlabel('Position j', fontsize=_FS)

    return fig


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE: fig_mechanism  ── hero figure: effective λ vs Δt, 3 zones × 3 models
# ═════════════════════════════════════════════════════════════════════════════

def make_fig_mechanism(models):
    main = ['S5', 'Mamba surrogate', 'TIDES']

    # Pre-compute curves
    curves = {}
    for name in main:
        m = models[name]
        for c in range(N_RATES):
            curves[(name, c)] = np.array(
                [effective_lambda_per_zone(m, c, float(d)) for d in DT_GRID])

    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.8), sharey=True)

    for c, ax in enumerate(axes):
        _shade_train(ax, label='Training Δt' if c == 0 else '_nolegend_')
        ax.axhline(RATE_LEVELS[c].item(), color='black', linestyle='--', lw=1.4,
                   label='Ground truth' if c == 0 else '_nolegend_')
        for name in main:
            ax.plot(DT_GRID, curves[(name, c)],
                    MARKERS[name] + '-', color=COLORS[name], lw=1.8, ms=5,
                    label=DISPLAY_NAMES.get(name, name) if c == 0 else '_nolegend_')
        _log_dt_ticks(ax)
        ax.set_xlabel('Test-time Δt', fontsize=_FS)
        if c == 0:
            ax.set_ylabel('Effective decay', fontsize=_FS)
        # panel label as in-axes annotation (no matplotlib title padding)
        ax.text(0.5, 1.01, ZONE_LABEL[c], transform=ax.transAxes,
                ha='center', va='bottom', fontsize=_FS, fontweight='bold')
        ax.grid(True, which='both', alpha=0.25)

    # legend to the right of the rightmost panel
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc='center left',
                    bbox_to_anchor=(1.03, 0.5),
                    fontsize=_FS - 1, framealpha=0.9, handlelength=1.8,
                    borderaxespad=0)

    plt.tight_layout(pad=0.5, w_pad=1.0)
    plt.subplots_adjust(right=0.82)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE: fig_ood_error  ── relative error vs Δt, all five models
# ═════════════════════════════════════════════════════════════════════════════

def make_fig_ood_error(results, var_y, order=None):
    if order is None:
        order = ['S5', 'Mamba surrogate', 'TIDES', 'TIDES (Λ-only)', 'Real Mamba']
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    _shade_train(ax)
    for name in order:
        ax.plot(DT_GRID, _rel_err(results[name], var_y),
                MARKERS[name] + '-', color=COLORS[name], lw=1.8, ms=6, label=DISPLAY_NAMES.get(name, name))
    _log_dt_ticks(ax)
    ax.set_yscale('log')
    ax.set_xlabel('Test-time Δt', fontsize=_FS)
    ax.set_ylabel('Relative error (%)', fontsize=_FS)
    ax.legend(loc='center left', bbox_to_anchor=(1.03, 0.5),
              fontsize=_FS - 1, framealpha=0.9, handlelength=1.8, borderaxespad=0)
    ax.grid(True, which='both', alpha=0.25)
    plt.tight_layout(pad=0.4)
    plt.subplots_adjust(right=0.70)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE: fig_real_mamba  ── Mamba surrogate vs Real Mamba only
# ═════════════════════════════════════════════════════════════════════════════

def make_fig_real_mamba(results, var_y):
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    _shade_train(ax)
    for name in ['Mamba surrogate', 'Real Mamba']:
        ax.plot(DT_GRID, _rel_err(results[name], var_y),
                MARKERS[name] + '-', color=COLORS[name], lw=1.8, ms=6, label=DISPLAY_NAMES.get(name, name))
    _log_dt_ticks(ax)
    ax.set_yscale('log')
    ax.set_xlabel('Test-time Δt', fontsize=_FS)
    ax.set_ylabel('Relative error (%)', fontsize=_FS)
    ax.legend(loc='upper right', fontsize=_FS - 1, framealpha=0.9, handlelength=1.8)
    ax.grid(True, which='both', alpha=0.25)
    plt.tight_layout(pad=0.4)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE: fig_tides_ablation  ── TIDES vs TIDES (Λ-only), clean vs noisy training
# ═════════════════════════════════════════════════════════════════════════════

def make_fig_tides_ablation(clean_results, noisy_results, var_y):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    panels = [
        (ax1, clean_results,  'Clean training'),
        (ax2, noisy_results,  'Noisy training → clean eval'),
    ]
    for ax, res, panel_title in panels:
        _shade_train(ax)
        for name in ['TIDES', 'TIDES (Λ-only)']:
            ax.plot(DT_GRID, _rel_err(res[name], var_y),
                    MARKERS[name] + '-', color=COLORS[name], lw=1.8, ms=6, label=DISPLAY_NAMES.get(name, name))
        _log_dt_ticks(ax)
        ax.set_yscale('log')
        ax.set_xlabel('Test-time Δt', fontsize=_FS)
        ax.set_ylabel('Relative error (%) — clean data', fontsize=_FS)
        ax.text(0.5, 1.01, panel_title, transform=ax.transAxes,
                ha='center', va='bottom', fontsize=_FS, fontweight='bold')
        ax.legend(loc='upper right', fontsize=_FS - 1, framealpha=0.9, handlelength=1.8)
        ax.grid(True, which='both', alpha=0.25)

    plt.tight_layout(pad=0.5, w_pad=1.2)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE: fig_reconstruction  ── four reconstruction panels for MinimalSSM models
# ═════════════════════════════════════════════════════════════════════════════

def make_fig_reconstruction(recon_models):
    canonical_err, canonical_idx, onecomp, full_pz, shared = \
        compute_recon_stats(recon_models)

    n = len(recon_models)
    name_list = list(recon_models.keys())
    w = 0.17
    offsets = np.linspace(-(n - 1) / 2 * w, (n - 1) / 2 * w, n)
    x_pos   = np.arange(3)
    xtlabels = ['Slow\n(λ=1.0)', 'Medium\n(λ=1.5)', 'Fast\n(λ=2.0)']

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))

    def _bars(ax, data_map, ylabel, ylim=None, legend_loc=None, joint_suffix=False):
        for i, name in enumerate(name_list):
            vals = (data_map[name][1] if joint_suffix else data_map[name])
            lbl  = (f'{DISPLAY_NAMES.get(name, name)}  (joint={data_map[name][0]:.3f})' if joint_suffix else DISPLAY_NAMES.get(name, name))
            ax.bar(x_pos + offsets[i], vals, w, color=COLORS[name],
                   label=lbl, edgecolor='#555555', linewidth=0.4)
        ax.set_xticks(x_pos); ax.set_xticklabels(xtlabels, fontsize=_FS - 1)
        ax.set_ylabel(ylabel, fontsize=_FS)
        if ylim: ax.set_ylim(*ylim)
        if legend_loc:
            ax.legend(loc=legend_loc, fontsize=_FS - 2, framealpha=0.9)
        ax.axhline(1.0, color='gray', lw=0.6, linestyle='--')
        ax.grid(axis='y', alpha=0.3)

    # (a) canonical Λ recovery — error bars (lower = better, no R²=1 line)
    ax = axes[0, 0]
    for i, name in enumerate(name_list):
        ax.bar(x_pos + offsets[i], canonical_err[name], w, color=COLORS[name],
               label=DISPLAY_NAMES.get(name, name), edgecolor='#555555', linewidth=0.4)
    ax.set_xticks(x_pos); ax.set_xticklabels(xtlabels, fontsize=_FS - 1)
    ax.set_ylabel('|λ_eff − λ_true| / λ_true  (%)', fontsize=_FS)
    ax.legend(loc='upper right', fontsize=_FS - 2, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    ax.text(0.02, 0.97, '(a)', transform=ax.transAxes, va='top', fontsize=_FS,
            fontweight='bold')

    # (b) 1-component R²
    ax = axes[0, 1]
    _bars(ax, onecomp, 'R²  (single canonical state)', ylim=(-0.15, 1.05),
          legend_loc='lower right')
    ax.text(0.02, 0.97, '(b)', transform=ax.transAxes, va='top', fontsize=_FS,
            fontweight='bold')

    # (c) full-basis per-zone R²
    ax = axes[1, 0]
    _bars(ax, full_pz, 'R²  (full basis, per-zone readout)', ylim=(0.96, 1.005))
    ax.text(0.02, 0.97, '(c)', transform=ax.transAxes, va='top', fontsize=_FS,
            fontweight='bold')

    # (d) shared-readout R²
    ax = axes[1, 1]
    _bars(ax, shared, 'R²  (shared readout)', ylim=(0.85, 1.01),
          legend_loc='lower left', joint_suffix=True)
    ax.text(0.02, 0.97, '(d)', transform=ax.transAxes, va='top', fontsize=_FS,
            fontweight='bold')

    plt.tight_layout(pad=0.6, h_pad=1.0, w_pad=1.0)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Generate TIDES paper figures.')
    parser.add_argument('--retrain', action='store_true',
                        help='Ignore cache and retrain all models from scratch.')
    args = parser.parse_args()

    snap = load_or_train(retrain=args.retrain)
    models, noisy_models = restore_models(snap)
    results       = snap['results']
    noisy_results = snap['noisy_results']
    var_y         = snap['var_y']

    # TIDES and TIDES Λ-only clean results (subset from full results dict)
    clean_ablation = {k: results[k] for k in ['TIDES', 'TIDES (Λ-only)']}
    # recon uses the 4 MinimalSSM-based models (all except Real Mamba)
    recon_models = {n: models[n] for n in ['S5', 'Mamba surrogate', 'TIDES', 'TIDES (Λ-only)']}

    print('\nGenerating figures...')
    _save(make_fig_setup(),                                           'fig_setup')
    _save(make_fig_mechanism(models),                                 'fig_mechanism')
    _save(make_fig_ood_error(results, var_y,
              order=['S5', 'Mamba surrogate', 'TIDES']),              'fig_ood_error_main')
    _save(make_fig_ood_error(results, var_y),                         'fig_ood_error_full')
    _save(make_fig_real_mamba(results, var_y),                        'fig_real_mamba')
    _save(make_fig_tides_ablation(clean_ablation, noisy_results, var_y),
                                                                      'fig_tides_ablation')
    _save(make_fig_reconstruction(recon_models),                      'fig_reconstruction')
    print('\nDone — 6 PDFs written to current directory.')


if __name__ == '__main__':
    main()
