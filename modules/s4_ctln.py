import torch
import torch.nn as nn
import torch.nn.functional as F

from .noiseResilient import RobustSpectralRefinement
from .TemporalConvEncoder import TemporalConvEncoder
from .S4 import S4Layer  # 导入刚才写的标准 S4 层
from .temporal_attention import TemporalAttention
from .regression_head import EmotionRegressionHead
from data_utils.normalization import lengths_to_mask

class S4_CTLN(nn.Module):
    """
    基于标准 S4 (Structured State-Space) 的连续时间网络
    对比 Liquid-S4，它移除了输入依赖的液体核，回归纯粹的线性长程建模
    """
    def __init__(
            self,
            input_dim=80,
            hidden_dim=256,
            num_s4_layers=2,
            d_state=64,
            output_dim=3,
            dropout=0.3
    ):
        super().__init__()

        # 1️⃣ 抗噪模块 (保持一致以便公平对比)
        self.noise_block = RobustSpectralRefinement(input_dim)

        # 2️⃣ 特征编码器
        self.encoder = TemporalConvEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        # 3️⃣ 标准 S4 Layers
        self.s4_layers = nn.ModuleList()
        for _ in range(num_s4_layers):
            self.s4_layers.append(nn.ModuleDict({
                "s4": S4Layer(hidden_dim, d_state=d_state), # 替换为标准 S4
                "norm": nn.LayerNorm(hidden_dim),
                "dropout": nn.Dropout(dropout),
                "activation": nn.GELU()
            }))

        # 4️⃣ 时间注意力 (全局池化)
        self.temporal_attention = TemporalAttention(hidden_dim)

        # 5️⃣ 回归预测头
        self.regression_head = EmotionRegressionHead(
            hidden_dim,
            output_dim,
            dropout
        )

    def forward(self, x, lengths=None, return_attention=False):
        # 1. 原始特征抗噪
        x = self.noise_block(x)

        # 2. 编码局部特征 (下采样时间轴)
        features = self.encoder(x)
        encoded_lengths = self.encoder.downsample_lengths(lengths)
        mask = lengths_to_mask(encoded_lengths, features.size(1)) if encoded_lengths is not None else None

        # 3. S4 动力学建模 (线性长程依赖提取)
        for layer in self.s4_layers:
            residual = features
            features = layer["s4"](features) # 标准 S4 前向传播
            features = layer["activation"](features)
            features = layer["norm"](features + residual)
            features = layer["dropout"](features)

        # 4. 全局时间加权
        pooled, attention = self.temporal_attention(features, mask=mask)

        # 5. 回归预测
        emotion = self.regression_head(pooled)

        outputs = {"emotion": emotion}
        if return_attention:
            outputs["attention"] = attention

        return outputs
