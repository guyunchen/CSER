import numpy as np
import torch


def normalize_logmel(feat, mode="fixed_db", mean=None, std=None, eps=1e-8):
    """Normalize Log-Mel features consistently across training and inference."""
    if mode == "fixed_db":
        return (feat + 40.0) / 40.0

    if mode == "fixed_db80":
        return np.clip((feat + 80.0) / 80.0, 0.0, 1.0)

    if mode == "standard":
        cur_mean = feat.mean() if mean is None else mean
        cur_std = feat.std() if std is None else std
        return (feat - cur_mean) / (cur_std + eps)

    if mode == "none":
        return feat

    raise ValueError(f"Unsupported normalization mode: {mode}")


def to_float32_array(raw_feat, feature_dim=80):
    """Convert parquet-stored nested feature values to a 2D float32 array."""
    try:
        feat = np.array(raw_feat, dtype=np.float32)
        if feat.ndim == 1:
            feat = feat.reshape(-1, feature_dim)
        return feat
    except (ValueError, TypeError):
        return np.array([np.array(f, dtype=np.float32) for f in raw_feat], dtype=np.float32)


def lengths_to_mask(lengths, max_len=None):
    """Build a boolean mask with True at valid time steps."""
    if lengths is None:
        return None

    if not torch.is_tensor(lengths):
        lengths = torch.tensor(lengths)

    if max_len is None:
        max_len = int(lengths.max().item())

    positions = torch.arange(max_len, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)
