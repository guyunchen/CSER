import torch
import torch.nn as nn

from .noiseResilient import RobustSpectralRefinement
from .TemporalConvEncoder import TemporalConvEncoder
from .liquid_layer import OptimizedCfCLayer
from .temporal_attention import TemporalAttention
from .regression_head import EmotionRegressionHead
from data_utils.normalization import lengths_to_mask

class CTLN(nn.Module):
    """
    更新版：基于 OptimizedCfC 的连续时间液态网络
    """
    def __init__(
        self,
        input_dim=80,
        hidden_dim=256,
        num_liquid_layers=1, # 建议先从1层开始，CfC单层表达力已经很强
        output_dim=2,
        dropout=0.3
    ):
        super().__init__()

        # 1️⃣ 抗噪模块 (保持不变)
        self.noise_block = RobustSpectralRefinement(input_dim)

        # 2️⃣ 特征编码器
        # 强烈建议在此模块加入 stride，将 T 维度缩小 2-4 倍
        self.encoder = TemporalConvEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        # 3️⃣ Liquid layers (替换为 Optimized 版本)
        self.liquid_layers = nn.ModuleList()
        for _ in range(num_liquid_layers):
            self.liquid_layers.append(
                OptimizedCfCLayer(
                    input_size=hidden_dim,
                    hidden_size=hidden_dim,
                    dropout=dropout
                )
            )

        # 4️⃣ 时间注意力 (保持不变)
        self.temporal_attention = TemporalAttention(hidden_dim)

        # 5️⃣ 回归头 (保持不变)
        self.regression_head = EmotionRegressionHead(
            hidden_dim,
            output_dim,
            dropout
        )

    def forward(self, x, lengths=None, return_attention=False):
        # 1. 抗噪
        x = self.noise_block(x)

        # 2. 编码 (提取局部时序特征)
        features = self.encoder(x)
        encoded_lengths = self.encoder.downsample_lengths(lengths)
        mask = lengths_to_mask(encoded_lengths, features.size(1)) if encoded_lengths is not None else None

        # 3. CfC 动力学建模 (提取深层动态依赖)
        # 现在这是一个带有状态记忆的递归过程，但通过 JIT 进行了加速
        for layer in self.liquid_layers:
            features = layer(features)

        # 4. 时间加权
        pooled, attention = self.temporal_attention(features, mask=mask)

        # 5. 回归预测
        emotion = self.regression_head(pooled)

        outputs = {"emotion": emotion}
        if return_attention:
            outputs["attention"] = attention

        return outputs
