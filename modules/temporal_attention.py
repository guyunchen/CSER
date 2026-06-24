import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        """
        改进版：多头统计注意力池化
        Args:
            hidden_dim: 输入特征维度
            num_heads: 注意力头数，建议 4 或 8
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # 多头注意力映射
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, num_heads)
        )

        # 最后的线性投影，将多头特征融合并保持输出维度
        # 注意：因为我们拼接了 Mean 和 Std，特征维度会翻倍
        self.out_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x, mask=None):
        """
        Input x: [Batch, Time, Hidden_dim]
        Output:
            pooled: [Batch, Hidden_dim]
            weights: [Batch, Time] (返回平均后的权重用于可视化)
        """
        # 1. 计算注意力分数 [B, T, Num_heads]
        scores = self.attention_net(x)
        if mask is not None:
            mask = mask.to(device=x.device, dtype=torch.bool)
            scores = scores.masked_fill(~mask.unsqueeze(-1), torch.finfo(scores.dtype).min)

        # 2. 在时间维度进行 Softmax
        weights = F.softmax(scores, dim=1)  # [B, T, Num_heads]

        # 3. 计算加权均值 (Weighted Mean)
        # x: [B, T, D], weights: [B, T, H] -> 扩展维度进行批量相乘
        # 通过 einsum 方便处理多头权重
        # b: batch, t: time, d: dim, h: head
        weighted_mean = torch.einsum('btd,bth->bhd', x, weights)  # [B, Num_heads, Hidden_dim]

        # 4. 计算加权标准差 (Weighted Std)
        # Var = E[x^2] - (E[x])^2
        weighted_sq_mean = torch.einsum('btd,bth->bhd', x ** 2, weights)
        variance = weighted_sq_mean - weighted_mean ** 2
        # 使用 relu 确保方差不为负，并加上 eps 后开方
        std = torch.sqrt(torch.relu(variance) + 1e-6)  # [B, Num_heads, Hidden_dim]

        # 5. 拼接均值和标准差
        # 融合所有头的信息：先在头维度取平均，然后拼接 Mean 和 Std
        mean_combined = torch.mean(weighted_mean, dim=1)  # [B, Hidden_dim]
        std_combined = torch.mean(std, dim=1)  # [B, Hidden_dim]

        combined = torch.cat([mean_combined, std_combined], dim=-1)  # [B, Hidden_dim * 2]

        # 6. 映射回原维度 (或者直接输出 2*D，取决于你 RegressionHead 的输入)
        pooled = self.out_proj(combined)

        # 为了保持兼容性，返回平均注意力权重
        avg_weights = torch.mean(weights, dim=-1)  # [B, T]

        return pooled, avg_weights
