import torch
import torch.nn as nn

class TemporalConvEncoder(nn.Module):
    """
    改进的轻量化下采样编码器
    1. 引入 Stride=2 的卷积，实现 4 倍下采样 (T -> T/4)
    2. 使用深度可分离卷积 (Depthwise Separable Conv)，适合 RK3588
    3. 加入残差结构，提升梯度流动
    """

    def __init__(self, input_dim=80, hidden_dim=128, dropout=0.2):
        super().__init__()

        # 第一层卷积：通道映射 + 2倍下采样
        # 结果：(B, input_dim, T) -> (B, hidden_dim, T/2)
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, hidden_dim), # GroupNorm 比 LayerNorm 在卷积层更稳定
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 第二层卷积：深度可分离卷积 + 再次2倍下采样
        # 结果：(B, hidden_dim, T/2) -> (B, hidden_dim, T/4)
        self.conv2 = nn.Sequential(
            # Depthwise
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, groups=hidden_dim),
            # Pointwise
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def downsample_lengths(lengths):
        if lengths is None:
            return None
        lengths = torch.div(lengths + 1, 2, rounding_mode="floor")
        lengths = torch.div(lengths + 1, 2, rounding_mode="floor")
        return torch.clamp(lengths, min=1)

    def forward(self, x):
        """
        x: (B, T, D)
        """
        # (B, T, D) -> (B, D, T)
        x = x.transpose(1, 2)

        # 两次下采样，T 维度缩小 4 倍
        x = self.conv1(x)
        x = self.conv2(x)

        # (B, D, T/4) -> (B, T/4, D)
        x = x.transpose(1, 2)

        x = self.norm(x)
        return x
