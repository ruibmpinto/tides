"""
Ablation: training-step wall time and peak GPU memory for TIDES vs RFormer
across sequence lengths and channel counts.

Grid:
    seq_lens : 1_000, 10_000, 100_000, 1_000_000
    channels : 1, 10, 100

Protocol:
    - batch_size = 1
    - 2 warmup steps (discarded), then 10 measured steps averaged
    - peak GPU memory reset before each timed block
    - Each config runs in an isolated subprocess so SLURM OOM kills are
      caught gracefully by the parent and logged as OOM.

Output: ablations/results_time_memory.csv
"""

import csv
import json
import os
import subprocess
import sys
import time
import traceback
import types

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "../.."))

import torch
import torch.nn as nn
import torch.optim as optim

from tides import TIDESClassifier
from model_classification import DecoderTransformer

# ── constants ─────────────────────────────────────────────────────────────────
SEQ_LENS   = [1_000, 5_000, 10_000, 50_000, 100_000]
CHANNELS   = [1]
BATCH_SIZE = 8
NUM_CLASSES = 5
N_WARMUP   = 5
N_STEPS    = 20

# ~100k param matched configs (4 layers each)
# TIDES: 99,557 params  |  RFormer: 94,368 params
TIDES_HIDDEN     = 32
TIDES_SSM_SIZE   = 16
TIDES_BLOCKS     = 1
TIDES_NUM_BLOCKS = 4

RFORMER_EMBD   = 32
RFORMER_HEADS  = 2
RFORMER_LAYERS = 4

RESULTS_FILE = os.path.join(_HERE, "results_time_memory.csv")
CSV_COLS = ["model", "seq_len", "channels", "time_ms", "time_std_ms", "peak_mem_mb", "status"]

# ── worker (runs inside subprocess) ──────────────────────────────────────────

def _worker(model_name: str, seq_len: int, channels: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def make_tides():
        return TIDESClassifier(
            d_input=channels, num_classes=NUM_CLASSES,
            d_hidden=TIDES_HIDDEN, ssm_size=TIDES_SSM_SIZE,
            ssm_blocks=TIDES_BLOCKS, num_blocks=TIDES_NUM_BLOCKS,
            lambda_re_mode="input_dependent", lambda_im_mode="lti",
            bc_mode="input_dependent", conj_sym=False, clip_eigs=False,
            proj_norm="rmsnorm",
        ).to(device)

    def make_rformer():
        cfg = types.SimpleNamespace(embd_pdrop=0.0, attn_pdrop=0.0, resid_pdrop=0.0,
                                    sparse=False, sub_len=1, scale_att=True, q_len=1)
        return DecoderTransformer(config=cfg, input_dim=channels, n_head=RFORMER_HEADS,
                                  seq_num=1, layer=RFORMER_LAYERS, n_embd=RFORMER_EMBD,
                                  win_len=seq_len, num_classes=NUM_CLASSES).to(device)

    model = make_tides() if model_name == "TIDES" else make_rformer()
    x = torch.randn(BATCH_SIZE, seq_len, channels, device=device)
    is_tides = model_name == "TIDES"

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    labels = torch.zeros(BATCH_SIZE, dtype=torch.long, device=device)

    model.train()
    for _ in range(N_WARMUP):
        optimizer.zero_grad()
        out = model(x, step_scale=1.0) if is_tides else model(x)
        criterion(out, labels).backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    step_times = []
    for _ in range(N_STEPS):
        optimizer.zero_grad()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        out = model(x, step_scale=1.0) if is_tides else model(x)
        criterion(out, labels).backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_times.append((time.perf_counter() - t0) * 1000)

    elapsed_ms = float(torch.tensor(step_times).mean())
    std_ms = float(torch.tensor(step_times).std())
    peak_mb = (torch.cuda.max_memory_allocated(device) / 1024 / 1024
               if device.type == "cuda" else float("nan"))

    print(json.dumps({"time_ms": elapsed_ms, "time_std_ms": std_ms, "peak_mem_mb": peak_mb}))


# ── orchestrator ──────────────────────────────────────────────────────────────

def run_one(model_name: str, seq_len: int, channels: int, writer, f):
    print(f"  [{model_name}] L={seq_len:>9,}  C={channels:>3}", end="  ", flush=True)

    cmd = [sys.executable, __file__, "--worker", model_name, str(seq_len), str(channels)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            data = json.loads(result.stdout.strip().splitlines()[-1])
            t, s, m = data["time_ms"], data["time_std_ms"], data["peak_mem_mb"]
            print(f"time={t:8.1f} ± {s:.1f} ms  mem={m:8.1f} MB")
            row = dict(model=model_name, seq_len=seq_len, channels=channels,
                       time_ms=f"{t:.2f}", time_std_ms=f"{s:.2f}",
                       peak_mem_mb=f"{m:.1f}", status="ok")
        else:
            # Non-zero exit — killed by OOM or crashed
            stderr_lines = result.stderr.strip().splitlines()
            stderr_tail = stderr_lines[-1] if stderr_lines else ""
            if result.returncode == -9 or "kill" in stderr_tail.lower() or "oom" in stderr_tail.lower():
                print("OOM")
                status = "OOM"
            else:
                print(f"ERROR (rc={result.returncode})")
                for line in stderr_lines[-10:]:
                    print(f"    {line}")
                status = f"error:{result.returncode}"
            row = dict(model=model_name, seq_len=seq_len, channels=channels,
                       time_ms="", time_std_ms="", peak_mem_mb="", status=status)
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        row = dict(model=model_name, seq_len=seq_len, channels=channels,
                   time_ms="", time_std_ms="", peak_mem_mb="", status="timeout")

    writer.writerow(row)
    f.flush()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(device)}")
    print(f"Grid:   seq_lens={SEQ_LENS}  channels={CHANNELS}")
    print(f"Output: {RESULTS_FILE}\n")

    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLS)
        writer.writeheader()

        for seq_len in SEQ_LENS:
            for channels in CHANNELS:
                print(f"\nseq_len={seq_len:>9,}  channels={channels}")
                for model_name in ["TIDES", "RFormer"]:
                    run_one(model_name, seq_len, channels, writer, f)

    print(f"\nDone. Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        _, _, model_name, seq_len, channels = sys.argv
        _worker(model_name, int(seq_len), int(channels))
    else:
        main()
