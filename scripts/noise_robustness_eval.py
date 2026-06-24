import argparse
import csv
import os
import sys
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data_utils.metrics import calculate_metrics
from data_utils.reader_ctln import IEMOCAPDataset, collate_fn
from modules.ls4_CTLN import LiteGLSER
from models.unified_ser import UnifiedSERModel
from training.data import build_iemocap_datasets


DEFAULT_MODEL_CONFIG = {
    "input_dim": 80,
    "hidden_dim": 256,
    "num_ls4_layers": 3,
    "d_state": 32,
    "p_order": 3,
    "output_dim": 3,
    "dropout": 0.4,
}


class IdentityFrontend(nn.Module):
    def forward(self, x):
        return x


def valid_mask(feats, lengths):
    positions = torch.arange(feats.size(1), device=feats.device)
    return positions.unsqueeze(0) < lengths.to(feats.device).unsqueeze(1)


def scale_noise_to_snr(feats, noise, lengths, snr_db):
    mask = valid_mask(feats, lengths).unsqueeze(-1)
    feat_power = (feats[mask.expand_as(feats)] ** 2).mean().clamp_min(1e-8)
    noise_power = (noise[mask.expand_as(noise)] ** 2).mean().clamp_min(1e-8)
    target_noise_power = feat_power / (10.0 ** (snr_db / 10.0))
    return noise * torch.sqrt(target_noise_power / noise_power)


def make_noise(feats, lengths, noise_type, snr_db):
    if noise_type == "clean":
        return feats

    if noise_type == "white":
        noise = torch.randn_like(feats)
    elif noise_type == "babble":
        # Feature-level babble proxy: mix shuffled utterances in the batch.
        noise = torch.zeros_like(feats)
        batch = feats.size(0)
        for shift in (1, 2, 3):
            noise = noise + torch.roll(feats, shifts=shift % batch, dims=0)
        noise = noise / 3.0
        noise = noise - noise.mean(dim=1, keepdim=True)
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")

    scaled_noise = scale_noise_to_snr(feats, noise, lengths, snr_db)
    mask = valid_mask(feats, lengths).unsqueeze(-1)
    return torch.where(mask, feats + scaled_noise, feats)


def build_model(checkpoint, frontend):
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config.update(checkpoint.get("config", {}).get("model", {}))
    config.pop("model_name", None)
    model = UnifiedSERModel(config) if "sequence_core" in config else LiteGLSER(**config)
    model.load_state_dict(checkpoint["model"], strict=True)
    if frontend == "identity":
        model.noise_block = IdentityFrontend()
    elif frontend != "robust":
        raise ValueError(f"Unsupported frontend: {frontend}")
    return model


def evaluate(model, loader, device, noise_type, snr_db):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for feats, labels, lengths in tqdm(loader, desc=f"{noise_type}:{snr_db}", leave=False):
            feats = feats.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            feats = make_noise(feats, lengths, noise_type, snr_db)
            preds = torch.sigmoid(model(feats, lengths=lengths)["emotion"])
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    return calculate_metrics(torch.cat(all_preds), torch.cat(all_labels))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="output/training_logs/best_ls4_vad_model.pth")
    parser.add_argument("--data", default=None, help="Legacy fixed-split feature parquet. LOSO checkpoints use their saved fold config.")
    parser.add_argument("--output", default="output/noise_robustness/ls4_noise_robustness.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--snrs", default="20,10,5,0")
    parser.add_argument("--noise-types", default="white,babble")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    cfg = checkpoint.get("config", {})
    if cfg.get("data", {}).get("protocol") == "loso":
        datasets, _ = build_iemocap_datasets(cfg["data"])
        dataset = datasets["test"]
    else:
        if not args.data:
            raise SystemExit("--data is required for legacy non-LOSO checkpoints.")
        dataset = IEMOCAPDataset(args.data, training=False)
    if args.max_samples > 0:
        dataset = torch.utils.data.Subset(dataset, range(min(args.max_samples, len(dataset))))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                        num_workers=args.num_workers)

    snrs = [int(x) for x in args.snrs.split(",") if x.strip()]
    noise_types = [x.strip() for x in args.noise_types.split(",") if x.strip()]
    conditions = [("clean", None)] + [(noise, snr) for noise in noise_types for snr in snrs]

    rows = []
    clean_by_frontend = {}
    for frontend in ("robust", "identity"):
        model = build_model(checkpoint, frontend).to(device)
        for noise_type, snr_db in conditions:
            metrics = evaluate(model, loader, device, noise_type, snr_db)
            key = (frontend, noise_type, snr_db)
            if noise_type == "clean":
                clean_by_frontend[frontend] = metrics["ccc_total"]
            clean_ccc = clean_by_frontend.get(frontend, metrics["ccc_total"])
            rows.append({
                "frontend": frontend,
                "noise_type": noise_type,
                "snr_db": "clean" if snr_db is None else snr_db,
                "ccc_v": metrics["ccc_v"],
                "ccc_a": metrics["ccc_a"],
                "ccc_d": metrics["ccc_d"],
                "mean_ccc": metrics["ccc_total"],
                "mae_v": metrics["mae_v"],
                "mae_a": metrics["mae_a"],
                "mae_d": metrics["mae_d"],
                "mean_mae": metrics["mae_total"],
                "ccc_drop": clean_ccc - metrics["ccc_total"],
                "retention": metrics["ccc_total"] / clean_ccc if abs(clean_ccc) > 1e-8 else 0.0,
            })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {args.output}")
    for row in rows:
        print(
            f"{row['frontend']:8s} {row['noise_type']:6s} {str(row['snr_db']):>5s} "
            f"MeanCCC={row['mean_ccc']:.4f} Drop={row['ccc_drop']:.4f} Ret={row['retention']:.3f}"
        )


if __name__ == "__main__":
    main()
