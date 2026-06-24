import argparse
import csv
import os
import time
from copy import deepcopy

import torch

from data_utils.experiment import load_config
from modules.ls4_CTLN import LiteGLSER
from train_ls4 import DEFAULT_CONFIG

try:
    from thop import profile
except ImportError:
    profile = None


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@torch.no_grad()
def measure_latency(model, dummy_input, device, warmup, runs):
    model.eval()
    for _ in range(warmup):
        _ = model(dummy_input)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(runs):
        _ = model(dummy_input)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / runs * 1000.0


def measure_flops(model, dummy_input):
    if profile is None:
        return None
    flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
    return flops


def load_model_config(config_path):
    cfg = load_config(config_path, deepcopy(DEFAULT_CONFIG))
    model_cfg = dict(cfg["model"])
    model_cfg.pop("model_name", None)
    return model_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True, help="Config files to evaluate.")
    parser.add_argument("--labels", nargs="*", default=None, help="Optional labels matching --configs.")
    parser.add_argument("--seq-len", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--audio-duration", type=float, default=3.0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--output", default="output/efficiency/ls4_efficiency.csv")
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.configs):
        raise ValueError("--labels must have the same length as --configs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []

    for index, config_path in enumerate(args.configs):
        label = args.labels[index] if args.labels else os.path.splitext(os.path.basename(config_path))[0]
        model_cfg = load_model_config(config_path)
        model = LiteGLSER(**model_cfg).to(device)
        dummy_input = torch.randn(args.batch_size, args.seq_len, model_cfg["input_dim"], device=device)

        total_params, trainable_params = count_parameters(model)
        flops = measure_flops(model, dummy_input)
        latency_ms = measure_latency(model, dummy_input, device, args.warmup, args.runs)
        rtf = (latency_ms / 1000.0) / args.audio_duration

        row = {
            "label": label,
            "config": config_path,
            "device": str(device),
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "audio_duration": args.audio_duration,
            "noise_frontend": model_cfg.get("noise_frontend", "identity"),
            "hidden_dim": model_cfg["hidden_dim"],
            "num_da_ls4_layers": model_cfg.get("num_da_ls4_layers", model_cfg.get("num_ls4_layers")),
            "d_state": model_cfg["d_state"],
            "p_order": model_cfg["p_order"],
            "params": total_params,
            "trainable_params": trainable_params,
            "params_m": total_params / 1e6,
            "flops": "" if flops is None else flops,
            "mflops": "" if flops is None else flops / 1e6,
            "latency_ms": latency_ms,
            "rtf": rtf,
        }
        rows.append(row)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nEfficiency evaluation")
    print(f"Saved CSV: {args.output}")
    print("label               params(M)  MFLOPs     latency(ms)  RTF")
    for row in rows:
        mflops = row["mflops"]
        mflops_text = "n/a" if mflops == "" else f"{mflops:.2f}"
        print(
            f"{row['label']:<19} "
            f"{row['params_m']:.3f}      "
            f"{mflops_text:<9} "
            f"{row['latency_ms']:.2f}        "
            f"{row['rtf']:.4f}"
        )


if __name__ == "__main__":
    main()
