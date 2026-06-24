import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from data_utils.metrics import calculate_metrics
from data_utils.reader_ctln import collate_fn
from models.unified_ser import UnifiedSERModel
from training.data import build_iemocap_datasets


def load_results(results_dir, limit=None):
    rows = []
    for path in sorted(Path(results_dir).glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            row = json.load(f)
        checkpoint = row.get("checkpoint") or ""
        if row.get("status") == "success" and checkpoint and os.path.exists(checkpoint):
            rows.append(row)
        if limit and len(rows) >= limit:
            break
    return rows


def valid_mask(feats, lengths):
    positions = torch.arange(feats.size(1), device=feats.device)
    return positions.unsqueeze(0) < lengths.to(feats.device).unsqueeze(1)


def scale_noise_to_snr(feats, noise, lengths, snr):
    mask = valid_mask(feats, lengths).unsqueeze(-1)
    signal_power = feats[mask.expand_as(feats)].pow(2).mean().clamp_min(1e-8)
    noise_power = noise[mask.expand_as(noise)].pow(2).mean().clamp_min(1e-8)
    target = signal_power / (10.0 ** (snr / 10.0))
    return noise * torch.sqrt(target / noise_power)


def add_noise(feats, lengths, noise_type, snr):
    if noise_type == "clean":
        return feats
    if noise_type == "white":
        noise = torch.randn_like(feats)
    elif noise_type == "babble":
        noise = torch.zeros_like(feats)
        batch = feats.size(0)
        for shift in (1, 2, 3):
            noise = noise + torch.roll(feats, shifts=shift % batch, dims=0)
        noise = noise / 3.0
        noise = noise - noise.mean(dim=1, keepdim=True)
    else:
        raise ValueError(f"Unsupported noise_type: {noise_type}")
    scaled = scale_noise_to_snr(feats, noise, lengths, snr)
    mask = valid_mask(feats, lengths).unsqueeze(-1)
    return torch.where(mask, feats + scaled, feats)


@torch.no_grad()
def evaluate(model, loader, device, noise_type="clean", snr=None):
    model.eval()
    all_preds, all_labels = [], []
    desc = "clean" if noise_type == "clean" else f"{noise_type}:{snr:g}"
    for feats, labels, lengths in tqdm(loader, desc=f"  Noise eval {desc}", leave=False):
        feats = feats.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)
        feats = add_noise(feats, lengths, noise_type, snr)
        preds = torch.sigmoid(model(feats, lengths=lengths)["emotion"])
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
    return calculate_metrics(torch.cat(all_preds), torch.cat(all_labels))


def evaluate_result(result, device, args):
    cfg = result["config"]
    datasets, _ = build_iemocap_datasets(cfg["data"])
    dataset = datasets["test"]
    loader = DataLoader(dataset, batch_size=cfg["loader"]["batch_size"], shuffle=False, collate_fn=collate_fn, num_workers=cfg["loader"]["num_workers"], pin_memory=(device.type == "cuda"))
    model = UnifiedSERModel(cfg["model"]).to(device)
    checkpoint = torch.load(result["checkpoint"], map_location=device)
    model.load_state_dict(checkpoint["model"])

    rows = []
    clean = evaluate(model, loader, device)
    for noise_type in args.noise_types:
        for snr in args.snr:
            metrics = evaluate(model, loader, device, noise_type=noise_type, snr=snr)
            rows.append({
                "experiment": result["experiment"],
                "group": result["group"],
                "id": result["id"],
                "seed": result["seed"],
                "protocol": result.get("protocol", ""),
                "fold_index": result.get("split", {}).get("fold_index", ""),
                "test_session": result.get("split", {}).get("test_session", ""),
                "noise_type": noise_type,
                "snr": snr,
                "mean_ccc": metrics["ccc_total"],
                "v_ccc": metrics["ccc_v"],
                "a_ccc": metrics["ccc_a"],
                "d_ccc": metrics["ccc_d"],
                "ccc_drop": clean["ccc_total"] - metrics["ccc_total"],
                "ccc_retention": metrics["ccc_total"] / clean["ccc_total"] if abs(clean["ccc_total"]) > 1e-8 else 0.0,
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(EXPERIMENT_DIR / "output" / "results"))
    parser.add_argument("--output", default=str(EXPERIMENT_DIR / "output" / "noise" / "noise_results.csv"))
    parser.add_argument("--noise-types", nargs="*", default=["white", "babble"], choices=["white", "babble"])
    parser.add_argument("--snr", nargs="*", type=float, default=[20.0, 10.0, 5.0, 0.0])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for result in load_results(args.results, args.limit):
        rows.extend(evaluate_result(result, device, args))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fieldnames = ["experiment", "group", "id", "seed", "protocol", "fold_index", "test_session", "noise_type", "snr", "mean_ccc", "v_ccc", "a_ccc", "d_ccc", "ccc_drop", "ccc_retention"]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved noise CSV: {args.output}")


if __name__ == "__main__":
    main()
