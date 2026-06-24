import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd


SESSION_RE = re.compile(r"Ses(?P<session>\d{2})", re.IGNORECASE)
UTTERANCE_SPEAKER_RE = re.compile(r"_(?P<gender>[FM])\d+\.wav$", re.IGNORECASE)


@dataclass(frozen=True)
class FoldSpec:
    index: int
    train_sessions: Tuple[str, ...]
    val_session: str
    test_session: str


def parse_session(file_name: str) -> str:
    match = SESSION_RE.search(str(file_name))
    if not match:
        raise ValueError(f"Cannot parse IEMOCAP session from file name: {file_name}")
    return f"Ses{int(match.group('session')):02d}"


def parse_speaker(file_name: str, gender: Optional[str] = None) -> str:
    session = parse_session(file_name)
    match = UTTERANCE_SPEAKER_RE.search(str(file_name))
    if match:
        speaker_gender = match.group("gender").upper()
    elif gender:
        speaker_gender = str(gender).strip()[0].upper()
    else:
        raise ValueError(f"Cannot parse IEMOCAP speaker from file name: {file_name}")
    return f"{session}{speaker_gender}"


def add_identity_columns(df: pd.DataFrame, file_col: str = "file") -> pd.DataFrame:
    if file_col not in df.columns:
        raise ValueError(f"IEMOCAP metadata requires a '{file_col}' column.")

    out = df.copy()
    out["session"] = out[file_col].map(parse_session)
    gender_values = out["gender"] if "gender" in out.columns else [None] * len(out)
    out["speaker"] = [
        parse_speaker(file_name, gender)
        for file_name, gender in zip(out[file_col], gender_values)
    ]
    return out


def _as_list(paths: Optional[Union[Sequence[str], str]]) -> List[str]:
    if paths is None:
        return []
    if isinstance(paths, (str, Path)):
        return [str(paths)]
    return [str(path) for path in paths]


def _read_feature_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "logmel_80" not in df.columns and "egemaps_88" not in df.columns:
        raise ValueError(
            f"{path} does not contain a supported feature column "
            "('logmel_80' or 'egemaps_88')."
        )
    return df


def _read_metadata_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    drop_cols = [col for col in ("bytes", "audio", "logmel_80", "egemaps_88") if col in df.columns]
    return df.drop(columns=drop_cols)


def load_iemocap_dataframe(
    feature_paths: Union[Sequence[str], str],
    metadata_paths: Optional[Union[Sequence[str], str]] = None,
) -> pd.DataFrame:
    feature_path_list = _as_list(feature_paths)
    metadata_path_list = _as_list(metadata_paths)
    if not feature_path_list:
        raise ValueError("At least one IEMOCAP feature parquet path is required.")
    if metadata_path_list and len(feature_path_list) != len(metadata_path_list):
        raise ValueError("feature_paths and metadata_paths must have the same length.")

    frames = []
    for index, feature_path in enumerate(feature_path_list):
        feature_df = _read_feature_parquet(feature_path)
        if "file" not in feature_df.columns:
            if not metadata_path_list:
                raise ValueError(
                    f"{feature_path} has no 'file' column. Provide matching metadata_paths "
                    "so LOSO can recover session and speaker identities."
                )
            meta_df = _read_metadata_parquet(metadata_path_list[index])
            if len(feature_df) != len(meta_df):
                raise ValueError(
                    f"Cannot align {feature_path} ({len(feature_df)} rows) with "
                    f"{metadata_path_list[index]} ({len(meta_df)} rows). Re-extract features "
                    "with metadata columns preserved."
                )
            meta_cols = [col for col in meta_df.columns if col not in feature_df.columns]
            feature_df = pd.concat(
                [
                    feature_df.reset_index(drop=True),
                    meta_df[meta_cols].reset_index(drop=True),
                ],
                axis=1,
            )
        frames.append(feature_df)

    combined = pd.concat(frames, ignore_index=True)
    combined = add_identity_columns(combined)
    sessions = sorted(combined["session"].unique().tolist())
    speakers = sorted(combined["speaker"].unique().tolist())
    if len(sessions) != 5:
        raise ValueError(f"Expected 5 IEMOCAP sessions, found {len(sessions)}: {sessions}")
    if len(speakers) != 10:
        raise ValueError(f"Expected 10 IEMOCAP speakers, found {len(speakers)}: {speakers}")
    return combined


def build_loso_folds(
    sessions: Iterable[str],
    val_strategy: str = "cyclic_next",
) -> List[FoldSpec]:
    ordered = tuple(sorted(set(sessions)))
    if len(ordered) < 3:
        raise ValueError("LOSO requires at least 3 sessions.")
    folds = []
    for index, test_session in enumerate(ordered):
        if val_strategy == "cyclic_next":
            val_session = ordered[(index + 1) % len(ordered)]
        elif val_strategy == "cyclic_prev":
            val_session = ordered[(index - 1) % len(ordered)]
        else:
            raise ValueError(f"Unsupported validation strategy: {val_strategy}")
        train_sessions = tuple(session for session in ordered if session not in {val_session, test_session})
        folds.append(
            FoldSpec(
                index=index,
                train_sessions=train_sessions,
                val_session=val_session,
                test_session=test_session,
            )
        )
    return folds


def split_loso_dataframe(df: pd.DataFrame, fold: FoldSpec) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df[df["session"].isin(fold.train_sessions)].reset_index(drop=True)
    val_df = df[df["session"] == fold.val_session].reset_index(drop=True)
    test_df = df[df["session"] == fold.test_session].reset_index(drop=True)
    validate_speaker_independent_split(train_df, val_df, test_df)
    return train_df, val_df, test_df


def validate_speaker_independent_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    splits = {"train": train_df, "val": val_df, "test": test_df}
    for name, split_df in splits.items():
        if split_df.empty:
            raise ValueError(f"{name} split is empty.")

    for (left_name, left_df), (right_name, right_df) in combinations(splits.items(), 2):
        shared_sessions = set(left_df["session"]) & set(right_df["session"])
        shared_speakers = set(left_df["speaker"]) & set(right_df["speaker"])
        if shared_sessions or shared_speakers:
            raise ValueError(
                f"Speaker-independent split violation between {left_name} and {right_name}: "
                f"sessions={sorted(shared_sessions)}, speakers={sorted(shared_speakers)}"
            )


def fold_summary(fold: FoldSpec, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    return {
        "fold_index": fold.index,
        "train_sessions": list(fold.train_sessions),
        "val_session": fold.val_session,
        "test_session": fold.test_session,
        "train_speakers": sorted(train_df["speaker"].unique().tolist()),
        "val_speakers": sorted(val_df["speaker"].unique().tolist()),
        "test_speakers": sorted(test_df["speaker"].unique().tolist()),
        "train_size": int(len(train_df)),
        "val_size": int(len(val_df)),
        "test_size": int(len(test_df)),
    }
