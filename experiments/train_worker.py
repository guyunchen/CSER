import argparse
import json
import os
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_utils.experiment import deep_update, set_seed
from models.unified_ser import UnifiedSERModel, build_loss
from modules.LabelRefiner import LabelRefiner
from training.data import build_iemocap_datasets, build_loaders
from training.engine import fit_and_test
from training.logging_utils import setup_logger


DEFAULT_CONFIG = {
    "experiment": "manual",
    "group": "manual",
    "id": "manual",
    "seed": 42,
    "save_checkpoint": True,
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
        "fold_index": 0,
        "val_strategy": "cyclic_next",
        "norm_mode": "fixed_db",
    },
    "loader": {"batch_size": 32, "num_workers": 0},
    "model": {
        "input_dim": 80,
        "hidden_dim": 256,
        "num_layers": 3,
        "d_state": 32,
        "p_order": 3,
        "output_dim": 3,
        "dropout": 0.4,
        "sequence_core": "da_ls4",
        "noise_frontend": "identity",
        "remove_sequence_core": False,
        "remove_attention": False,
        "disable_ls4_dynamic": False,
    },
    "data_aug": {"disable_feature_aug": False},
    "label_refiner": {"sigma": 0.05, "apply_smoothing": True},
    "loss": {
        "loss_type": "improved",
        "ccc_weight": 0.5,
        "corr_weight": 0.25,
        "mse_weight": 0.15,
        "smooth_weight": 0.10,
        "task_importance": [1.5, 1.0, 1.0],
    },
    "optimizer": {"lr": 5e-5, "weight_decay": 0.05},
    "scheduler": {"factor": 0.5, "patience": 15},
    "training": {"epochs": 300, "early_stop": 50, "max_train_batches": 0, "max_eval_batches": 0},
    "runtime": {"data_parallel": False, "amp": False},
    "output": {
        "root": "experiments/output",
        "checkpoint": "",
        "log": "",
        "result": "",
    },
}


def load_generated_config(path):
    with open(path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    return deep_update(deepcopy(DEFAULT_CONFIG), user_cfg)


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def metric_fields(prefix, metrics):
    return {
        f"{prefix}_mean_ccc": metrics["ccc_total"],
        f"{prefix}_v_ccc": metrics["ccc_v"],
        f"{prefix}_a_ccc": metrics["ccc_a"],
        f"{prefix}_d_ccc": metrics["ccc_d"],
        f"{prefix}_mean_mae": metrics["mae_total"],
        f"{prefix}_v_mae": metrics["mae_v"],
        f"{prefix}_a_mae": metrics["mae_a"],
        f"{prefix}_d_mae": metrics["mae_d"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-config", required=True)
    args = parser.parse_args()
    cfg = load_generated_config(args.generated_config)

    set_seed(cfg["seed"])
    logger = setup_logger("experiment-worker", cfg["output"].get("log"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Config: {cfg}")

    datasets, split_info = build_iemocap_datasets(cfg["data"])
    if cfg["data_aug"].get("disable_feature_aug"):
        datasets["train"].feature_aug.aug_prob = 0.0
    loaders = build_loaders(datasets, cfg["loader"], device)
    logger.info(f"Split: {split_info}")

    model = UnifiedSERModel(cfg["model"]).to(device)
    if device.type == "cuda" and cfg.get("runtime", {}).get("data_parallel", False):
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            model = torch.nn.DataParallel(model)
            logger.info(f"Using DataParallel on {gpu_count} GPUs.")
        else:
            logger.info("DataParallel requested, but only one CUDA device is available.")
    label_refiner = LabelRefiner(**cfg["label_refiner"]).to(device)
    criterion = build_loss(cfg["loss"], cfg["model"]["output_dim"]).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(criterion.parameters()), **cfg["optimizer"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=cfg["scheduler"]["factor"],
        patience=cfg["scheduler"]["patience"],
    )

    fit_result = fit_and_test(
        model=model,
        label_refiner=label_refiner,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        device=device,
        cfg=cfg,
        logger=logger,
    )
    val_metrics = fit_result["val_metrics"]
    test_metrics = fit_result["test_metrics"]

    result = {
        "experiment": cfg["experiment"],
        "group": cfg["group"],
        "id": cfg["id"],
        "seed": cfg["seed"],
        "status": "success",
        "protocol": cfg["data"].get("protocol", "loso"),
        "speaker_independent": cfg["data"].get("protocol", "loso") == "loso",
        "best_epoch": fit_result["best_epoch"],
        "mean_ccc": test_metrics["ccc_total"],
        "v_ccc": test_metrics["ccc_v"],
        "a_ccc": test_metrics["ccc_a"],
        "d_ccc": test_metrics["ccc_d"],
        "mean_mae": test_metrics["mae_total"],
        "v_mae": test_metrics["mae_v"],
        "a_mae": test_metrics["mae_a"],
        "d_mae": test_metrics["mae_d"],
        "checkpoint": cfg["output"]["checkpoint"] if cfg.get("save_checkpoint", True) else "",
        "log": cfg["output"]["log"],
        "split": split_info,
        "config": cfg,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    result.update(metric_fields("val", val_metrics))
    result.update(metric_fields("test", test_metrics))
    write_json(cfg["output"]["result"], result)
    logger.info(
        f"Test Mean-CCC: {test_metrics['ccc_total']:.4f} | "
        f"V:{test_metrics['ccc_v']:.4f} A:{test_metrics['ccc_a']:.4f} D:{test_metrics['ccc_d']:.4f} | "
        f"MAE:{test_metrics['mae_total']:.4f}"
    )
    logger.info(f"Result saved: {cfg['output']['result']}")


if __name__ == "__main__":
    main()
