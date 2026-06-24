import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

from .feature_augment import FeatureAugmentor
from .normalization import normalize_logmel, to_float32_array


class IEMOCAPDataset(Dataset):
    def __init__(
        self,
        parquet_path=None,
        training=True,
        norm_mode="fixed_db",
        dataframe=None,
        feature_column=None,
    ):
        if dataframe is None:
            if parquet_path is None:
                raise ValueError("Either parquet_path or dataframe must be provided.")
            self.df = pd.read_parquet(parquet_path)
        else:
            self.df = dataframe.reset_index(drop=True).copy()

        self.training = training
        self.norm_mode = norm_mode
        self.feature_aug = FeatureAugmentor(aug_prob=0.3)

        self.feature_column = feature_column or self._infer_feature_column()
        self.feature_dim = 80 if self.feature_column == "logmel_80" else 88
        self.features = self.df[self.feature_column].values
        self.v_labels = self._normalize_label(self.df["valence"].values)
        self.a_labels = self._normalize_label(self.df["arousal"].values)
        if "dominance" in self.df.columns:
            self.d_labels = self._normalize_label(self.df["dominance"].values)
        else:
            self.d_labels = np.full(len(self.df), 0.5, dtype=np.float32)

    def _infer_feature_column(self):
        for column in ("logmel_80", "egemaps_88"):
            if column in self.df.columns:
                return column
        raise ValueError("IEMOCAPDataset requires 'logmel_80' or 'egemaps_88'.")

    @staticmethod
    def _normalize_label(values):
        arr = np.asarray(values, dtype=np.float32)
        if len(arr) and (np.nanmax(arr) > 1.5 or np.nanmin(arr) < -0.05):
            arr = (arr - 1.0) / 4.0
        return np.clip(arr, 0.0, 1.0).astype(np.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = to_float32_array(self.features[idx], feature_dim=self.feature_dim)
        if self.feature_column == "logmel_80":
            feat = normalize_logmel(feat, mode=self.norm_mode)
        feat = torch.tensor(feat, dtype=torch.float32)

        if self.training:
            feat = self.feature_aug(feat)

        label = np.array(
            [self.v_labels[idx], self.a_labels[idx], self.d_labels[idx]],
            dtype=np.float32,
        )
        return feat, torch.tensor(label, dtype=torch.float32)


def collate_fn(batch):
    feats, labels = zip(*batch)
    lengths = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
    padded_feats = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True)
    labels = torch.stack(labels)
    return padded_feats, labels, lengths
