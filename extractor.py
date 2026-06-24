import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from tqdm import tqdm

from data_utils.iemocap_protocol import add_identity_columns


DEFAULT_INPUTS = [
    "dataset/IEMOCAP/train-00000-of-00003.parquet",
    "dataset/IEMOCAP/train-00001-of-00003.parquet",
    "dataset/IEMOCAP/train-00002-of-00003.parquet",
]


class IEMOCAPLogMelExtractor:
    def __init__(self, sr=16000, n_mels=80, n_fft=400, hop_length=160, preemphasis=0.97):
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.preemphasis = preemphasis
        self.window = np.hanning(n_fft).astype(np.float32)
        self.mel_basis = self._build_mel_basis().astype(np.float32)

    @staticmethod
    def _hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _build_mel_basis(self):
        min_mel = self._hz_to_mel(0.0)
        max_mel = self._hz_to_mel(self.sr / 2.0)
        mel_points = np.linspace(min_mel, max_mel, self.n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)
        bins = np.floor((self.n_fft + 1) * hz_points / self.sr).astype(int)
        bins = np.clip(bins, 0, self.n_fft // 2)

        basis = np.zeros((self.n_mels, self.n_fft // 2 + 1), dtype=np.float32)
        for mel_idx in range(1, self.n_mels + 1):
            left = bins[mel_idx - 1]
            center = bins[mel_idx]
            right = bins[mel_idx + 1]
            if center <= left:
                center = left + 1
            if right <= center:
                right = center + 1
            right = min(right, self.n_fft // 2)

            for fft_bin in range(left, min(center, basis.shape[1])):
                basis[mel_idx - 1, fft_bin] = (fft_bin - left) / max(center - left, 1)
            for fft_bin in range(center, min(right, basis.shape[1])):
                basis[mel_idx - 1, fft_bin] = (right - fft_bin) / max(right - center, 1)

        enorm = 2.0 / np.maximum(hz_points[2 : self.n_mels + 2] - hz_points[: self.n_mels], 1e-8)
        basis *= enorm[:, np.newaxis]
        return basis

    def _decode_audio(self, audio_value):
        if isinstance(audio_value, dict) and "bytes" in audio_value:
            audio_bytes = audio_value["bytes"]
        elif isinstance(audio_value, bytes):
            audio_bytes = audio_value
        else:
            return None, None

        y, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        return y.astype(np.float32, copy=False), sr

    def _resample_if_needed(self, y, sr):
        if sr == self.sr:
            return y
        if len(y) == 0:
            return y
        duration = len(y) / float(sr)
        target_len = max(int(round(duration * self.sr)), 1)
        old_x = np.linspace(0.0, duration, num=len(y), endpoint=False)
        new_x = np.linspace(0.0, duration, num=target_len, endpoint=False)
        return np.interp(new_x, old_x, y).astype(np.float32)

    def _stft_power(self, y):
        if len(y) == 0:
            y = np.zeros(self.n_fft, dtype=np.float32)
        if len(y) < self.n_fft:
            y = np.pad(y, (0, self.n_fft - len(y)), mode="constant")

        pad = self.n_fft // 2
        pad_mode = "reflect" if len(y) > 1 else "constant"
        y = np.pad(y, (pad, pad), mode=pad_mode)
        frame_count = 1 + max((len(y) - self.n_fft) // self.hop_length, 0)
        shape = (frame_count, self.n_fft)
        strides = (self.hop_length * y.strides[0], y.strides[0])
        frames = np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides)
        spectrum = np.fft.rfft(frames * self.window, n=self.n_fft, axis=1)
        return (np.abs(spectrum) ** 2).astype(np.float32)

    def extract(self, audio_value):
        try:
            y, sr = self._decode_audio(audio_value)
            if y is None:
                return None

            y = self._resample_if_needed(y, sr)
            if len(y) > 1:
                y = np.append(y[0], y[1:] - self.preemphasis * y[:-1]).astype(np.float32)

            power = self._stft_power(y)
            mel = np.maximum(power @ self.mel_basis.T, 1e-10)
            ref = np.maximum(mel.max(), 1e-10)
            log_mel = 10.0 * np.log10(mel / ref)
            return log_mel.astype(np.float32)
        except Exception as exc:
            print(f"Feature extraction failed: {exc}")
            return None


def detect_audio_column(df):
    for column in ("bytes", "audio"):
        if column in df.columns:
            return column
    raise ValueError("Input parquet must contain either 'bytes' or 'audio'.")


def read_source_frame(input_path):
    schema_names = pq.ParquetFile(input_path).schema_arrow.names
    wanted = [
        "file",
        "audio",
        "bytes",
        "path",
        "EmoAct",
        "EmoVal",
        "EmoDom",
        "valence",
        "arousal",
        "dominance",
        "gender",
        "transcription",
        "major_emotion",
        "speaking_rate",
        "pitch_mean",
        "pitch_std",
        "rms",
        "relative_db",
    ]
    columns = [column for column in wanted if column in schema_names]
    return pd.read_parquet(input_path, columns=columns)


def ensure_vad_columns(df, input_file):
    rename_map = {}
    if "valence" not in df.columns and "EmoVal" in df.columns:
        rename_map["EmoVal"] = "valence"
    if "arousal" not in df.columns and "EmoAct" in df.columns:
        rename_map["EmoAct"] = "arousal"
    if "dominance" not in df.columns and "EmoDom" in df.columns:
        rename_map["EmoDom"] = "dominance"
    if rename_map:
        df = df.rename(columns=rename_map)

    required = {"file", "valence", "arousal", "dominance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{input_file} is missing required columns: {sorted(missing)}")
    return df


def prepare_feature_frame(input_path, shard_index, extractor):
    df = read_source_frame(input_path)
    if "path" not in df.columns and "audio" in df.columns:
        df["path"] = df["audio"].map(lambda value: value.get("path") if isinstance(value, dict) else None)
    df = ensure_vad_columns(df, input_path)
    df = add_identity_columns(df)
    df["source_shard"] = Path(input_path).name
    df["source_row"] = np.arange(len(df), dtype=np.int64)

    audio_col = detect_audio_column(df)
    tqdm.pandas(desc=f"Extracting {Path(input_path).name}")
    df["logmel_80"] = df[audio_col].progress_apply(extractor.extract)
    df = df.dropna(subset=["logmel_80"]).copy()
    df["logmel_80"] = df["logmel_80"].apply(lambda x: x.tolist())

    columns = [
        "file",
        "path",
        "session",
        "speaker",
        "gender",
        "major_emotion",
        "transcription",
        "speaking_rate",
        "pitch_mean",
        "pitch_std",
        "rms",
        "relative_db",
        "valence",
        "arousal",
        "dominance",
        "source_shard",
        "source_row",
        "logmel_80",
    ]
    return df[[col for col in columns if col in df.columns]].reset_index(drop=True)


def session_output_path(output_dir, session):
    session_num = int(str(session).replace("Ses", ""))
    return Path(output_dir) / f"ses_{session_num}.parquet"


def remove_stale_outputs(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for session_num in range(1, 6):
        path = output_dir / f"ses_{session_num}.parquet"
        if path.exists():
            path.unlink()


def write_session_parquets(input_paths, output_dir, extractor):
    remove_stale_outputs(output_dir)
    writers = {}
    counts = {f"Ses{idx:02d}": 0 for idx in range(1, 6)}

    try:
        for shard_index, input_path in enumerate(input_paths):
            if not Path(input_path).exists():
                raise FileNotFoundError(f"Input not found: {input_path}")

            frame = prepare_feature_frame(input_path, shard_index, extractor)
            for session, session_df in frame.groupby("session", sort=True):
                path = session_output_path(output_dir, session)
                table = pa.Table.from_pandas(session_df.reset_index(drop=True), preserve_index=False)
                if session not in writers:
                    writers[session] = pq.ParquetWriter(path, table.schema)
                else:
                    table = table.cast(writers[session].schema)
                writers[session].write_table(table)
                counts[session] = counts.get(session, 0) + len(session_df)

            del frame
    finally:
        for writer in writers.values():
            writer.close()

    missing = [session for session, count in counts.items() if count == 0]
    if missing:
        raise RuntimeError(f"No rows written for sessions: {missing}")

    print("Saved session-level feature files:")
    for session in sorted(counts):
        print(f"  {session}: {counts[session]} rows -> {session_output_path(output_dir, session)}")


def main():
    parser = argparse.ArgumentParser(description="Extract IEMOCAP Log-Mel features and split by session.")
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--output-dir", default="dataset/IEMOCAP/session_data")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n-mels", type=int, default=80)
    args = parser.parse_args()

    extractor = IEMOCAPLogMelExtractor(sr=args.sr, n_mels=args.n_mels)
    write_session_parquets(args.inputs, args.output_dir, extractor)


if __name__ == "__main__":
    main()
