import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class IEMOCAPDataset(Dataset):
    def __init__(self, parquet_path):
        # 载入 parquet 文件
        self.df = pd.read_parquet(parquet_path)

        # 读取 eGeMAPS 88维特征列
        # 确保提取脚本中使用的列名为 "egemaps_88"
        self.features = self.df["egemaps_88"].values

        # 标签归一化 1-5 -> 0-1 (V-A-D 三维度)
        self.v_labels = (self.df["valence"].values - 1.0) / 4.0
        self.a_labels = (self.df["arousal"].values - 1.0) / 4.0
        self.d_labels = (self.df["dominance"].values - 1.0) / 4.0

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # 1. 获取特征数据 (原始存储为 list)
        raw_feat = self.features[idx]

        # 2. 转换为 numpy 数组
        feat = np.array(raw_feat, dtype=np.float32)

        # 3. 维度适配 (针对 LS4 序列模型)
        # eGeMAPS Functionals 是静态向量 (88,)
        # 我们需要将其扩展为 (Time=1, Dimension=88) 以适配序列模型的输入要求
        if feat.ndim == 1:
            feat = feat[np.newaxis, :]  # 形状变为 (1, 88)

        # 4. 标签处理 (V, A, D 三个维度)
        label = np.array([
            self.v_labels[idx],
            self.a_labels[idx],
            self.d_labels[idx]
        ], dtype=np.float32)

        # 使用 from_numpy 效率更高
        return torch.from_numpy(feat), torch.from_numpy(label)


def collate_fn(batch):
    """
    处理 Padding。
    虽然 eGeMAPS 功能特征长度固定为 1，但保留此函数以确保训练脚本的通用性。
    """
    feats, labels = zip(*batch)

    # 记录长度（对于 eGeMAPS 统计特征，长度通常都为 1）
    lengths = torch.tensor([f.shape[0] for f in feats])

    # 填充序列: [Batch, Max_Time, 88]
    padded_feats = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True)

    return padded_feats, torch.stack(labels), lengths