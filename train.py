import argparse
import logging
import os
import random
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils.experiment import load_config, set_seed
from data_utils.metrics import calculate_metrics
from data_utils.reader_ctln import IEMOCAPDataset, collate_fn
from losses.ccc_loss import ImprovedMultiTaskLoss
from modules.LabelRefiner import LabelRefiner
from modules.ctln_model import CTLN


DEFAULT_CONFIG = {
    "seed": 42,
    "data": {
        "protocol": "loso",
        "feature_paths": [
            "dataset/IEMOCAP/session_data/ses_1.parquet",
            "dataset/IEMOCAP/session_data/ses_2.parquet",
            "dataset/IEMOCAP/session_data/ses_3.parquet",
            "dataset/IEMOCAP/session_data/ses_4.parquet",
            "dataset/IEMOCAP/session_data/ses_5.parquet",
        ],
        "metadata_paths": [],
        "val_strategy": "cyclic_next",
        "norm_mode": "fixed_db",
    },
    "loader": {"batch_size": 32, "num_workers": 0},
    "model": {
        "input_dim": 80,
        "hidden_dim": 256,
        "num_liquid_layers": 2,
        "output_dim": 2,
        "dropout": 0.4,
    },
    "spec_aug": {"freq_mask_range": 15, "time_mask_range": 40},
    "label_refiner": {"sigma": 0.05},
    "loss": {"mse_weight": 0.2, "task_importance": [1.5, 1.0]},
    "optimizer": {"lr": 1e-4, "weight_decay": 0.01},
    "scheduler": {"factor": 0.5, "patience": 7},
    "training": {"epochs": 100, "early_stop": 20},
}


class SpecAugment(nn.Module):
    def __init__(self, freq_mask_range=15, time_mask_range=40):
        super().__init__()
        self.freq_mask_range = freq_mask_range
        self.time_mask_range = time_mask_range

    def forward(self, x):
        if not self.training:
            return x
        _, t, f = x.shape
        f_mask = min(random.randint(5, self.freq_mask_range), max(f - 1, 1))
        f0 = random.randint(0, max(f - f_mask, 0))
        x[:, :, f0:f0 + f_mask] = 0
        t_mask = min(random.randint(10, self.time_mask_range), max(t - 1, 1))
        t0 = random.randint(0, max(t - t_mask, 0))
        x[:, t0:t0 + t_mask, :] = 0
        return x


def setup_logger():
    log_dir = "output/training_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"CTLN_Final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_filename, encoding="utf-8"), logging.StreamHandler()],
    )
    return logging.getLogger("CTLN"), log_dir


def train_one_epoch(model, spec_aug, label_refiner, loader, optimizer, criterion, device, scaler):
    model.train()
    spec_aug.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    last_details = {}

    pbar = tqdm(loader, desc="  Training", leave=False)
    for feats, labels, lengths in pbar:
        feats = feats.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)
        feats = spec_aug(feats)
        refined_labels = label_refiner(labels[:, :2])

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            raw_out = model(feats, lengths=lengths)["emotion"]
            preds = torch.sigmoid(raw_out)
            loss, details = criterion(preds, refined_labels)

        if device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        all_preds.append(preds.detach().cpu())
        all_labels.append(labels[:, :preds.shape[1]].detach().cpu())
        last_details = details
        pbar.set_postfix(v_w=f"{details.get('v_weight', 0.0):.2f}", a_w=f"{details.get('a_weight', 0.0):.2f}")

    metrics = calculate_metrics(torch.cat(all_preds), torch.cat(all_labels))
    return total_loss / len(loader), metrics, last_details


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for feats, labels, lengths in tqdm(loader, desc="  Evaluating", leave=False):
            feats = feats.to(device)
            lengths = lengths.to(device)
            preds = torch.sigmoid(model(feats, lengths=lengths)["emotion"])
            all_preds.append(preds.cpu())
            all_labels.append(labels[:, :preds.shape[1]].cpu())
    return calculate_metrics(torch.cat(all_preds), torch.cat(all_labels))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Optional YAML config path.")
    args = parser.parse_args()
    cfg = load_config(args.config, DEFAULT_CONFIG)
    if not cfg.get("data", {}).get("allow_speaker_dependent", False):
        raise SystemExit(
            "This legacy IEMOCAP train/test entry is disabled to avoid speaker leakage. "
            "Use `python train_loso.py` or set data.allow_speaker_dependent=true for a legacy diagnostic run."
        )
    set_seed(cfg.get("seed"))

    logger, log_dir = setup_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Config: {cfg}")

    train_ds = IEMOCAPDataset(cfg["data"]["train_path"], training=True, norm_mode=cfg["data"]["norm_mode"])
    test_ds = IEMOCAPDataset(cfg["data"]["test_path"], training=False, norm_mode=cfg["data"]["norm_mode"])
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["loader"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg["loader"]["num_workers"],
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["loader"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg["loader"]["num_workers"],
        pin_memory=True,
    )

    model = CTLN(**cfg["model"]).to(device)
    spec_aug = SpecAugment(**cfg["spec_aug"]).to(device)
    label_refiner = LabelRefiner(**cfg["label_refiner"]).to(device)
    criterion = ImprovedMultiTaskLoss(
        num_tasks=cfg["model"]["output_dim"],
        mse_weight=cfg["loss"]["mse_weight"],
        task_importance=tuple(cfg["loss"]["task_importance"]),
    ).to(device)

    params = list(model.parameters()) + list(criterion.parameters())
    optimizer = torch.optim.AdamW(params, **cfg["optimizer"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=cfg["scheduler"]["factor"],
        patience=cfg["scheduler"]["patience"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_ccc = -1.0
    patience_counter = 0
    for epoch in range(cfg["training"]["epochs"]):
        avg_loss, _, loss_det = train_one_epoch(model, spec_aug, label_refiner, train_loader, optimizer, criterion, device, scaler)
        eval_m = evaluate(model, test_loader, device)
        scheduler.step(eval_m["ccc_total"])

        logger.info(
            f"Epoch {epoch + 1:03d} | Loss: {avg_loss:.4f} | "
            f"V-Weight: {loss_det.get('v_weight', 0.0):.3f} | A-Weight: {loss_det.get('a_weight', 0.0):.3f} | "
            f"V-CCC: {eval_m['ccc_v']:.4f} | A-CCC: {eval_m['ccc_a']:.4f} | Total: {eval_m['ccc_total']:.4f}"
        )

        if eval_m["ccc_total"] > best_ccc:
            best_ccc = eval_m["ccc_total"]
            patience_counter = 0
            torch.save(
                {"model": model.state_dict(), "ccc": best_ccc, "loss_state": criterion.state_dict(), "config": cfg, "epoch": epoch + 1},
                os.path.join(log_dir, "best_model.pth"),
            )
            logger.info(f"  >>> Best saved. CCC: {best_ccc:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= cfg["training"]["early_stop"]:
            logger.info("Early stopping triggered.")
            break


if __name__ == "__main__":
    main()
