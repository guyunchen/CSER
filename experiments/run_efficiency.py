import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from models.unified_ser import UnifiedSERModel

try:
    from thop import profile
except ImportError:
    profile = None


def load_results(results_dir, limit=None):
    rows = []
    for path in sorted(Path(results_dir).glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            row = json.load(f)
        if row.get("status") == "success":
            rows.append(row)
        if limit and len(rows) >= limit:
            break
    return rows


def count_params(model):
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params, trainable


@torch.no_grad()
def latency_ms(model, dummy, device, warmup, runs):
    model.eval()
    for _ in range(warmup):
        _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(runs):
        _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / runs


def measure_one(result, device, args):
    cfg = result["config"]
    model_cfg = cfg["model"]
    model = UnifiedSERModel(model_cfg).to(device)
    checkpoint = result.get("checkpoint") or ""
    checkpoint_mb = 0.0
    if checkpoint and os.path.exists(checkpoint):
        checkpoint_mb = os.path.getsize(checkpoint) / (1024 * 1024)
    dummy = torch.randn(1, args.seq_len, model_cfg["input_dim"], device=device)
    params, trainable = count_params(model)
    flops = ""
    if profile is not None:
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
    ms = latency_ms(model, dummy, device, args.warmup, args.runs)
    gpu_memory = 0.0
    if device.type == "cuda":
        gpu_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return {
        "experiment": result["experiment"],
        "group": result["group"],
        "id": result["id"],
        "seed": result["seed"],
        "protocol": result.get("protocol", ""),
        "fold_index": result.get("split", {}).get("fold_index", ""),
        "params": params / 1e6,
        "trainable_params": trainable / 1e6,
        "mflops": "" if flops == "" else flops / 1e6,
        "latency_ms": ms,
        "rtf": (ms / 1000.0) / args.audio_duration,
        "checkpoint_mb": checkpoint_mb,
        "gpu_memory_mb": gpu_memory,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(EXPERIMENT_DIR / "output" / "results"))
    parser.add_argument("--output", default=str(EXPERIMENT_DIR / "output" / "efficiency" / "efficiency.csv"))
    parser.add_argument("--seq-len", type=int, default=300)
    parser.add_argument("--audio-duration", type=float, default=3.0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = load_results(args.results, args.limit)
    rows = [measure_one(result, device, args) for result in results]
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fieldnames = ["experiment", "group", "id", "seed", "protocol", "fold_index", "params", "trainable_params", "mflops", "latency_ms", "rtf", "checkpoint_mb", "gpu_memory_mb"]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved efficiency CSV: {args.output}")


if __name__ == "__main__":
    main()
