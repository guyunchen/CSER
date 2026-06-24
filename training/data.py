import os
from typing import Optional

from torch.utils.data import DataLoader

from data_utils.iemocap_protocol import (
    build_loso_folds,
    fold_summary,
    load_iemocap_dataframe,
    split_loso_dataframe,
)
from data_utils.reader_ctln import IEMOCAPDataset, collate_fn


DEFAULT_FEATURE_PATHS = [
    "dataset/IEMOCAP/session_data/ses_1.parquet",
    "dataset/IEMOCAP/session_data/ses_2.parquet",
    "dataset/IEMOCAP/session_data/ses_3.parquet",
    "dataset/IEMOCAP/session_data/ses_4.parquet",
    "dataset/IEMOCAP/session_data/ses_5.parquet",
]
DEFAULT_METADATA_PATHS = []


def build_iemocap_datasets(data_cfg: dict, fold_index: Optional[int] = None):
    protocol = data_cfg.get("protocol", "loso")
    norm_mode = data_cfg.get("norm_mode", "fixed_db")

    if protocol == "loso":
        feature_paths = data_cfg.get("feature_paths") or DEFAULT_FEATURE_PATHS
        metadata_paths = data_cfg.get("metadata_paths") or DEFAULT_METADATA_PATHS
        df = load_iemocap_dataframe(feature_paths, metadata_paths)
        folds = build_loso_folds(df["session"].unique(), data_cfg.get("val_strategy", "cyclic_next"))
        selected_index = int(fold_index if fold_index is not None else data_cfg.get("fold_index", 0))
        if selected_index < 0 or selected_index >= len(folds):
            raise ValueError(f"fold_index must be in [0, {len(folds) - 1}], got {selected_index}")

        fold = folds[selected_index]
        train_df, val_df, test_df = split_loso_dataframe(df, fold)
        datasets = {
            "train": IEMOCAPDataset(dataframe=train_df, training=True, norm_mode=norm_mode),
            "val": IEMOCAPDataset(dataframe=val_df, training=False, norm_mode=norm_mode),
            "test": IEMOCAPDataset(dataframe=test_df, training=False, norm_mode=norm_mode),
        }
        return datasets, fold_summary(fold, train_df, val_df, test_df)

    if protocol == "fixed":
        train_ds = IEMOCAPDataset(data_cfg["train_path"], training=True, norm_mode=norm_mode)
        val_path = data_cfg.get("val_path") or data_cfg.get("test_path")
        val_ds = IEMOCAPDataset(val_path, training=False, norm_mode=norm_mode)
        test_ds = IEMOCAPDataset(data_cfg["test_path"], training=False, norm_mode=norm_mode)
        return (
            {"train": train_ds, "val": val_ds, "test": test_ds},
            {
                "protocol": "fixed",
                "warning": "Fixed train/test split is not speaker-independent unless it was created externally.",
            },
        )

    raise ValueError(f"Unsupported IEMOCAP protocol: {protocol}")


def build_loaders(datasets: dict, loader_cfg: dict, device):
    batch_size = loader_cfg.get("batch_size", 32)
    num_workers = loader_cfg.get("num_workers", 4)
    if os.name == "nt" and num_workers > 0:
        num_workers = 0
    pin_memory = device.type == "cuda"
    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }
