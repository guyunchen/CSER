import argparse
import logging
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils.experiment import load_config, set_seed
from data_utils.metrics import calculate_metrics
from data_utils.reader_ctln import IEMOCAPDataset, collate_fn
from losses.ccc_loss import ImprovedMultiTaskLoss
from modules.LabelRefiner import LabelRefiner
from modules.ls4_CTLN import LiteGLSER
from models.original_ls4_ctln import OriginalLS4_CTLN


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
        "model_name": "Lite_GLSER",
        "input_dim": 80,
        "hidden_dim": 256,
        "num_ls4_layers": 3,
        "num_da_ls4_layers": 3,
        "d_state": 32,
        "p_order": 3,
        "output_dim": 3,
        "dropout": 0.4,
        "noise_frontend": "identity",
    },
    "label_refiner": {"sigma": 0.05},
    "loss": {"task_importance": [1.5, 1.0, 1.0]},
    "optimizer": {"lr": 5e-5, "weight_decay": 0.05},
    "scheduler": {"factor": 0.5, "patience": 15},
    "training": {"epochs": 300, "early_stop": 50, "checkpoint_name": "best_lite_glser_vad_model.pth"},
}


def build_model(model_cfg):
    model_cfg = dict(model_cfg)
    model_name = model_cfg.pop("model_name", "Lite_GLSER")
    if model_name == "Original_LS4":
        return OriginalLS4_CTLN(**model_cfg)
    if model_name in {"Lite_GLSER", "Lite-GLSER", "LiteGLSER", "LS4_CTLN"}:
        return LiteGLSER(**model_cfg)
    raise ValueError(f"Unsupported model_name: {model_name}")


def setup_logger():
    log_dir = "output/training_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(
        log_dir,
        f"Lite_GLSER_VAD_adamw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_filename, encoding="utf-8"), logging.StreamHandler()],
    )
    return logging.getLogger("Lite-GLSER-AdamW"), log_dir


def train_one_epoch(model, label_refiner, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    last_details = {}

    optimizer_name = optimizer.__class__.__name__
    pbar = tqdm(loader, desc=f"  Training ({optimizer_name})", leave=False)
    for feats, labels, lengths in pbar:
        feats = feats.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)
        target_labels = label_refiner(labels)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            raw_out = model(feats, lengths=lengths)["emotion"]
            preds = torch.sigmoid(raw_out)
            loss, details = criterion(preds, target_labels)

        if device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

        total_loss += loss.item()
        all_preds.append(preds.detach().cpu())
        all_labels.append(labels.detach().cpu())
        last_details = details

        postfix = {
            "loss": f"{loss.item():.3f}",
            "v_ccc": f"{details.get('v_ccc', 0.0):.3f}",
            "v_w": f"{details.get('v_weight', 0.0):.2f}",
        }
        pbar.set_postfix(**postfix)

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
            all_labels.append(labels.cpu())
    return calculate_metrics(torch.cat(all_preds), torch.cat(all_labels))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Optional YAML config path.")
    parser.add_argument("--resume-from", default=None, help="Optional checkpoint path to resume from.")
    args = parser.parse_args()
    cfg = load_config(args.config, DEFAULT_CONFIG)
    if not cfg.get("data", {}).get("allow_speaker_dependent", False):
        raise SystemExit(
            "This legacy IEMOCAP train/test entry is disabled to avoid speaker leakage. "
            "Use `python train_loso.py --only-ids A3` or set data.allow_speaker_dependent=true for a legacy diagnostic run."
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
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["loader"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg["loader"]["num_workers"],
    )

    model = build_model(cfg["model"]).to(device)
    label_refiner = LabelRefiner(**cfg["label_refiner"]).to(device)
    criterion = ImprovedMultiTaskLoss(
        num_tasks=cfg["model"]["output_dim"],
        task_importance=tuple(cfg["loss"]["task_importance"]),
    ).to(device)

    params = list(model.parameters()) + list(criterion.parameters())
    optimizer = torch.optim.AdamW(params, **cfg["optimizer"])
    logger.info("Optimizer: AdamW")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=cfg["scheduler"]["factor"],
        patience=cfg["scheduler"]["patience"],
    )

    best_ccc = -1.0
    patience_counter = 0
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    start_epoch = 0
    resume_from = args.resume_from or cfg["training"].get("resume_from")
    if resume_from:
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model"])
        best_ccc = checkpoint.get("ccc", best_ccc)
        start_epoch = checkpoint.get("epoch", 0)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            logger.info("Checkpoint has no optimizer state; AdamW state is reinitialized.")
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint and device.type == "cuda":
            scaler.load_state_dict(checkpoint["scaler"])
        logger.info(f"Resumed from {resume_from} at epoch {start_epoch}; best CCC: {best_ccc:.4f}")

    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        avg_loss, _, _ = train_one_epoch(model, label_refiner, train_loader, optimizer, criterion, device, scaler)
        eval_m = evaluate(model, test_loader, device)
        scheduler.step(eval_m["ccc_total"])

        logger.info(
            f"Epoch {epoch + 1:03d} | Loss: {avg_loss:.4f} | "
            f"Mean-CCC: {eval_m['ccc_total']:.4f} | "
            f"V:{eval_m['ccc_v']:.3f} A:{eval_m['ccc_a']:.3f} D:{eval_m['ccc_d']:.3f}"
        )

        if eval_m["ccc_total"] > best_ccc:
            best_ccc = eval_m["ccc_total"]
            patience_counter = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "ccc": best_ccc,
                    "config": cfg,
                    "epoch": epoch + 1,
                },
                os.path.join(log_dir, cfg["training"].get("checkpoint_name", "best_ls4_vad_model.pth")),
            )
            logger.info(f"  >>> Best saved. CCC: {best_ccc:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= cfg["training"]["early_stop"]:
            logger.info("Early stopping triggered.")
            break


if __name__ == "__main__":
    main()
