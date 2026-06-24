import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_utils.iemocap_protocol import add_identity_columns, build_loso_folds


def read_metadata(path):
    schema_names = pq.ParquetFile(path).schema_arrow.names
    columns = [col for col in schema_names if col not in {"bytes", "audio", "logmel_80", "egemaps_88"}]
    df = pd.read_parquet(path, columns=columns)
    drop_cols = [col for col in ("bytes", "audio", "logmel_80", "egemaps_88") if col in df.columns]
    df = df.drop(columns=drop_cols)
    if "session" in df.columns and "speaker" in df.columns:
        return df
    return add_identity_columns(df)


def feature_row_count(path):
    if not Path(path).exists():
        return None
    return pq.ParquetFile(path).metadata.num_rows


def build_manifest(feature_paths, metadata_paths=None):
    metadata_paths = metadata_paths or []
    if metadata_paths and len(feature_paths) != len(metadata_paths):
        raise ValueError("feature_paths and metadata_paths must have the same length.")

    frames = []
    offset = 0
    for shard_index, feature_path in enumerate(feature_paths):
        expected_rows = feature_row_count(feature_path)
        metadata_path = metadata_paths[shard_index] if metadata_paths else feature_path
        meta = read_metadata(metadata_path)
        if expected_rows is not None and len(meta) != expected_rows:
            raise ValueError(
                f"Row count mismatch: {feature_path} has {expected_rows}, "
                f"but {metadata_path} has {len(meta)}."
            )
        meta = meta.copy()
        meta["feature_shard"] = str(feature_path)
        meta["feature_shard_index"] = shard_index
        meta["feature_row"] = range(len(meta))
        meta["global_row"] = range(offset, offset + len(meta))
        offset += len(meta)
        frames.append(meta)

    manifest = pd.concat(frames, ignore_index=True)
    folds = build_loso_folds(manifest["session"].unique())
    rows = []
    base_cols = [
        "global_row",
        "feature_shard",
        "feature_shard_index",
        "feature_row",
        "file",
        "session",
        "speaker",
        "gender",
        "valence",
        "arousal",
        "dominance",
    ]
    base_cols = [col for col in base_cols if col in manifest.columns]

    for fold in folds:
        fold_df = manifest[base_cols].copy()
        fold_df["fold_index"] = fold.index
        fold_df["split"] = "train"
        fold_df.loc[fold_df["session"] == fold.val_session, "split"] = "val"
        fold_df.loc[fold_df["session"] == fold.test_session, "split"] = "test"
        fold_df["train_sessions"] = "|".join(fold.train_sessions)
        fold_df["val_session"] = fold.val_session
        fold_df["test_session"] = fold.test_session
        rows.append(fold_df)

    return pd.concat(rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Write a lightweight IEMOCAP LOSO split manifest.")
    parser.add_argument(
        "--feature-paths",
        nargs="+",
        default=[
            "dataset/IEMOCAP/session_data/ses_1.parquet",
            "dataset/IEMOCAP/session_data/ses_2.parquet",
            "dataset/IEMOCAP/session_data/ses_3.parquet",
            "dataset/IEMOCAP/session_data/ses_4.parquet",
            "dataset/IEMOCAP/session_data/ses_5.parquet",
        ],
    )
    parser.add_argument(
        "--metadata-paths",
        nargs="+",
        default=[],
    )
    parser.add_argument("--output", default="dataset/IEMOCAP/loso_splits.csv")
    args = parser.parse_args()

    manifest = build_manifest(args.feature_paths, args.metadata_paths)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False, encoding="utf-8")
    print(f"Saved LOSO manifest: {output}")
    print(manifest.groupby(["fold_index", "split"])["session"].unique())


if __name__ == "__main__":
    main()
