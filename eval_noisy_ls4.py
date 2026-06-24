import argparse
import csv
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils.experiment import load_config, set_seed
from data_utils.metrics import calculate_metrics
from data_utils.reader_ctln import IEMOCAPDataset, collate_fn
from modules.ls4_CTLN import LiteGLSER
from models.unified_ser import UnifiedSERModel
from training.data import build_iemocap_datasets
from train_ls4 import DEFAULT_CONFIG


def add_gaussian_noise(feats, lengths, snr_db):
    if snr_db is None:
        return feats

    return add_feature_noise(feats, lengths, "white", snr_db)


def valid_mask(feats, lengths):
    positions = torch.arange(feats.size(1), device=feats.device)
    return positions.unsqueeze(0) < lengths.to(feats.device).unsqueeze(1)


def scale_noise_to_snr(feats, noise, lengths, snr_db):
    mask = valid_mask(feats, lengths).unsqueeze(-1)
    signal_power = feats[mask.expand_as(feats)].pow(2).mean().clamp_min(1e-8)
    noise_power = noise[mask.expand_as(noise)].pow(2).mean().clamp_min(1e-8)
    target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    return noise * torch.sqrt(target_noise_power / noise_power)


def add_feature_noise(feats, lengths, noise_type, snr_db):
    if noise_type == "clean":
        return feats

    if noise_type == "white":
        noise = torch.randn_like(feats)
    elif noise_type == "babble":
        noise = torch.zeros_like(feats)
        batch_size = feats.size(0)
        for shift in (1, 2, 3):
            noise = noise + torch.roll(feats, shifts=shift % batch_size, dims=0)
        noise = noise / 3.0
        noise = noise - noise.mean(dim=1, keepdim=True)
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")

    scaled_noise = scale_noise_to_snr(feats, noise, lengths, snr_db)
    mask = valid_mask(feats, lengths).unsqueeze(-1)
    return torch.where(mask, feats + scaled_noise, feats)


@torch.no_grad()
def evaluate(model, loader, device, noise_type="clean", snr_db=None):
    model.eval()
    all_preds, all_labels = [], []
    desc = "clean" if noise_type == "clean" else f"{noise_type}_{snr_db:g}db"

    for feats, labels, lengths in tqdm(loader, desc=f"  Evaluating {desc}", leave=False):
        feats = feats.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)
        feats = add_feature_noise(feats, lengths, noise_type, snr_db)
        preds = torch.sigmoid(model(feats, lengths=lengths)["emotion"])
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    return calculate_metrics(torch.cat(all_preds), torch.cat(all_labels))


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    if "model" not in checkpoint:
        raise ValueError(f"Checkpoint does not contain a 'model' state dict: {path}")
    return checkpoint


def infer_noise_frontend(model_state):
    keys = model_state.keys()
    if any(key.startswith("noise_block.robust.") for key in keys):
        return "gated_robust"
    if any(key.startswith("noise_block.spectral_refinement.") for key in keys):
        return "robust"
    return "identity"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to a saved Lite-GLSER checkpoint.")
    parser.add_argument("--config", default=None, help="Optional config override. Defaults to checkpoint config.")
    parser.add_argument("--snr", nargs="*", type=float, default=[20.0, 10.0, 5.0, 0.0])
    parser.add_argument("--noise-types", nargs="*", default=["white"], choices=["white", "babble"])
    parser.add_argument("--output", default=None, help="Optional CSV output path.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.checkpoint, device)

    cfg = checkpoint.get("config", DEFAULT_CONFIG)
    if args.config:
        cfg = load_config(args.config, cfg)
    cfg.setdefault("model", {})
    if "noise_frontend" not in cfg["model"]:
        cfg["model"]["noise_frontend"] = infer_noise_frontend(checkpoint["model"])

    if cfg["data"].get("protocol") == "loso":
        datasets, _ = build_iemocap_datasets(cfg["data"])
        test_ds = datasets["test"]
    else:
        test_ds = IEMOCAPDataset(cfg["data"]["test_path"], training=False, norm_mode=cfg["data"]["norm_mode"])
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["loader"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg["loader"]["num_workers"],
    )

    model_cfg = dict(cfg["model"])
    model_name = model_cfg.pop("model_name", None)
    if "sequence_core" in model_cfg:
        model = UnifiedSERModel(model_cfg).to(device)
    else:
        model = LiteGLSER(**model_cfg).to(device)
    model.load_state_dict(checkpoint["model"])

    results = []
    clean = evaluate(model, test_loader, device, noise_type="clean", snr_db=None)
    results.append(("clean", "clean", None, clean))
    for noise_type in args.noise_types:
        for snr_db in args.snr:
            metrics = evaluate(model, test_loader, device, noise_type=noise_type, snr_db=snr_db)
            results.append((f"{noise_type}_{snr_db:g}db", noise_type, snr_db, metrics))

    clean_ccc = clean["ccc_total"]
    rows = []
    for name, noise_type, snr_db, metrics in results:
        row = {
            "condition": name,
            "noise_type": noise_type,
            "snr_db": "" if snr_db is None else snr_db,
            "ccc_total": metrics["ccc_total"],
            "ccc_v": metrics["ccc_v"],
            "ccc_a": metrics["ccc_a"],
            "ccc_d": metrics["ccc_d"],
            "mae_total": metrics["mae_total"],
            "ccc_drop": clean_ccc - metrics["ccc_total"],
            "ccc_retention": metrics["ccc_total"] / clean_ccc if abs(clean_ccc) > 1e-8 else 0.0,
        }
        rows.append(row)

    print("\nRobustness evaluation")
    print(f"checkpoint: {args.checkpoint}")
    print(f"noise_frontend: {cfg['model'].get('noise_frontend', 'identity')}")
    print("condition        mean_ccc  drop    retain  v_ccc   a_ccc   d_ccc   mae")
    for row in rows:
        print(
            f"{row['condition']:<16} "
            f"{row['ccc_total']:.4f}   "
            f"{row['ccc_drop']:.4f}  "
            f"{row['ccc_retention']:.3f}   "
            f"{row['ccc_v']:.4f} "
            f"{row['ccc_a']:.4f} "
            f"{row['ccc_d']:.4f} "
            f"{row['mae_total']:.4f}"
        )

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved CSV: {args.output}")


if __name__ == "__main__":
    main()
