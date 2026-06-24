import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """通道注意力机制：帮助模型聚焦于对情感表达更有意义的频段"""

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class ConvBlock(nn.Module):
    """带残差和归一化的 1D 卷积块"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out + residual)
        return out


class AcousticFrontEnd(nn.Module):
    """卷积前端：将原始 Log-Mel 转化为高阶声学表征"""

    def __init__(self, input_dim=80, hidden_dim=256):
        super().__init__()
        # 第一层：扩大通道数
        self.layer1 = ConvBlock(input_dim, 128, kernel_size=5, stride=1)

        # 第二层：时间维度下采样 (stride=2)，减少后续 Liquid Layer 的计算压力
        self.layer2 = ConvBlock(128, 128, kernel_size=3, stride=2)

        # 第三层：提取深层纹理并加入 SE 注意力
        self.layer3 = ConvBlock(128, hidden_dim, kernel_size=3, stride=1)
        self.se = SEBlock(hidden_dim)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x shape: (Batch, Time, Freq=80)
        # PyTorch Conv1d 期待: (Batch, Freq, Time)
        x = x.transpose(1, 2)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.se(x)
        x = self.dropout(x)

        # 返回 Liquid Layer 期待的形状: (Batch, New_Time, Hidden_Dim)
        return x.transpose(1, 2)