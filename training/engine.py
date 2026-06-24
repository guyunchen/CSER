import copy
import os
from pathlib import Path

import torch
from tqdm import tqdm

from data_utils.metrics import calculate_metrics


def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def train_one_epoch(model, label_refiner, loader, optimizer, criterion, device, scaler, max_batches=0, amp_enabled=False):
    model.train()
    label_refiner.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    seen_batches = 0

    for batch_idx, (feats, labels, lengths) in enumerate(tqdm(loader, desc="  Training", leave=False), start=1):
        seen_batches = batch_idx
        feats = feats.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)
        targets = label_refiner(labels)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda" and amp_enabled)):
            preds = torch.sigmoid(model(feats, lengths=lengths)["emotion"])
            loss, _ = criterion(preds, targets)

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
        if max_batches and batch_idx >= max_batches:
            break

    if not all_preds:
        raise RuntimeError("Training loader produced no batches.")
    return total_loss / max(seen_batches, 1), calculate_metrics(torch.cat(all_preds), torch.cat(all_labels))


@torch.no_grad()
def evaluate(model, loader, device, max_batches=0, desc="  Evaluating"):
    model.eval()
    all_preds, all_labels = [], []
    for batch_idx, (feats, labels, lengths) in enumerate(tqdm(loader, desc=desc, leave=False), start=1):
        feats = feats.to(device)
        lengths = lengths.to(device)
        preds = torch.sigmoid(model(feats, lengths=lengths)["emotion"])
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        if max_batches and batch_idx >= max_batches:
            break

    if not all_preds:
        raise RuntimeError("Evaluation loader produced no batches.")
    return calculate_metrics(torch.cat(all_preds), torch.cat(all_labels))


def _cpu_state_dict(model):
    base_model = unwrap_model(model)
    return {key: value.detach().cpu().clone() for key, value in base_model.state_dict().items()}


def load_model_state(model, state_dict):
    base_model = unwrap_model(model)
    try:
        base_model.load_state_dict(state_dict)
    except RuntimeError:
        if all(key.startswith("module.") for key in state_dict):
            stripped = {key.removeprefix("module."): value for key, value in state_dict.items()}
            base_model.load_state_dict(stripped)
        else:
            raise


def save_checkpoint(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def fit_and_test(
    model,
    label_refiner,
    criterion,
    optimizer,
    scheduler,
    loaders,
    device,
    cfg,
    logger,
):
    training_cfg = cfg["training"]
    output_cfg = cfg.get("output", {})
    max_train_batches = int(training_cfg.get("max_train_batches", 0) or 0)
    max_eval_batches = int(training_cfg.get("max_eval_batches", 0) or 0)
    amp_enabled = bool(cfg.get("runtime", {}).get("amp", False))
    save_ckpt = bool(cfg.get("save_checkpoint", True)) and bool(output_cfg.get("checkpoint"))

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and amp_enabled))
    best = {"val_ccc": -1.0, "epoch": 0, "val_metrics": None}
    best_state = None
    patience_counter = 0

    for epoch in range(training_cfg["epochs"]):
        avg_loss, _ = train_one_epoch(
            model,
            label_refiner,
            loaders["train"],
            optimizer,
            criterion,
            device,
            scaler,
            max_batches=max_train_batches,
            amp_enabled=amp_enabled,
        )
        val_metrics = evaluate(model, loaders["val"], device, max_eval_batches, desc="  Validating")
        scheduler.step(val_metrics["ccc_total"])

        logger.info(
            f"Epoch {epoch + 1:03d} | Loss: {avg_loss:.4f} | "
            f"Val Mean-CCC: {val_metrics['ccc_total']:.4f} | "
            f"V:{val_metrics['ccc_v']:.4f} A:{val_metrics['ccc_a']:.4f} D:{val_metrics['ccc_d']:.4f} | "
            f"MAE:{val_metrics['mae_total']:.4f}"
        )

        if val_metrics["ccc_total"] > best["val_ccc"]:
            best = {"val_ccc": val_metrics["ccc_total"], "epoch": epoch + 1, "val_metrics": val_metrics}
            patience_counter = 0
            if save_ckpt:
                save_checkpoint(
                    output_cfg["checkpoint"],
                    {
                        "model": unwrap_model(model).state_dict(),
                        "criterion": criterion.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "val_ccc": best["val_ccc"],
                        "config": cfg,
                        "epoch": epoch + 1,
                    },
                )
                logger.info(f"  >>> Best validation checkpoint saved. CCC: {best['val_ccc']:.4f}")
            else:
                best_state = _cpu_state_dict(model)
        else:
            patience_counter += 1

        if patience_counter >= training_cfg["early_stop"]:
            logger.info("Early stopping triggered.")
            break

    if save_ckpt and os.path.exists(output_cfg["checkpoint"]):
        checkpoint = torch.load(output_cfg["checkpoint"], map_location=device)
        load_model_state(model, checkpoint["model"])
    elif best_state is not None:
        load_model_state(model, best_state)

    test_metrics = evaluate(model, loaders["test"], device, max_eval_batches, desc="  Testing")
    return {
        "best_epoch": best["epoch"],
        "val_metrics": best["val_metrics"],
        "test_metrics": test_metrics,
    }
