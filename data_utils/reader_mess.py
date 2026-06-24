import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

from .normalization import normalize_logmel, to_float32_array


class MESSDataset(Dataset):
    def __init__(self, parquet_path, norm_mode="fixed_db"):
        self.df = pd.read_parquet(parquet_path)
        self.norm_mode = norm_mode
        self.features = self.df["logmel_80"].values
        self.v_labels = self.df["valence"].values.astype(np.float32)
        self.a_labels = self.df["arousal"].values.astype(np.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = to_float32_array(self.features[idx], feature_dim=80)
        feat = normalize_logmel(feat, mode=self.norm_mode)
        label = np.array([self.v_labels[idx], self.a_labels[idx]], dtype=np.float32)
        return torch.tensor(feat, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


def collate_fn(batch):
    feats, labels = zip(*batch)
    lengths = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
    padded_feats = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True)
    labels_stack = torch.stack(labels)
    return padded_feats, labels_stack, lengths
