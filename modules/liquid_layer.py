import torch
import torch.nn as nn
import torch.nn.functional as F


# 使用 JIT 编译 Cell 逻辑，减少 Python 调度开销，这是在 RK3588 上提速的关键
@torch.jit.script
def cfc_step(x_pre_t, h, h_proj_w, h_proj_b, w_tau, hidden_size: int, dt: float):
    # h_proj: (B, 3*H)
    h_proj = F.linear(h, h_proj_w, h_proj_b)

    # 将预计算的输入投影与当前的隐藏状态投影相加
    # 官方 CfC 包含：ff1, ff2, gate, 以及用于时间的系数
    combined = x_pre_t + h_proj

    # 拆分通道
    f1_i = combined[:, 0:hidden_size]
    f2_i = combined[:, hidden_size:2 * hidden_size]
    g_i = combined[:, 2 * hidden_size:3 * hidden_size]
    t_i = combined[:, 3 * hidden_size:4 * hidden_size]

    # 激活函数
    f1 = torch.tanh(f1_i)
    f2 = torch.tanh(f2_i)
    g = torch.sigmoid(g_i)

    # 时间衰减项：使用 softplus 保证 tau > 0，防止 NaN
    # tau = exp(- (w_tau + t_i) * dt)
    tau = F.softplus(w_tau + t_i)
    decay = torch.exp(-tau * dt)

    # CfC 核心方程实现
    # h = f1 * (1 - g * decay) + f2 * (g * decay)
    h_next = f1 * (1.0 - g * decay) + f2 * (g * decay)
    return h_next


class OptimizedCfCLayer(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size

        # 输入投影 (一次性在序列上完成)
        self.input_proj = nn.Linear(input_size, 4 * hidden_size)
        # 隐藏状态投影 (必须在循环内更新)
        self.hidden_proj = nn.Linear(hidden_size, 4 * hidden_size)

        # 可学习的时间常数基准
        self.w_tau = nn.Parameter(torch.zeros(hidden_size))
        self.dropout = nn.Dropout(dropout)

        # 初始化建议：正交初始化有助于 RNN 稳定性
        nn.init.orthogonal_(self.hidden_proj.weight)

    def forward(self, x, timespans: float = 1.0):
        """
        x: (B, T, D)
        """
        B, T, _ = x.shape
        device = x.device

        # --- 提速关键：预投影 ---
        # 此时 x_pre 的形状是 (B, T, 4*H)，这一步在 GPU 上是高度并行的
        x_pre = self.input_proj(x)

        h = torch.zeros(B, self.hidden_size, device=device)
        outputs = []

        # 缓存权重引用，避免循环内多次查找属性
        hw = self.hidden_proj.weight
        hb = self.hidden_proj.bias

        # 逐时间步迭代 (动力学建模)
        for t in range(T):
            h = cfc_step(x_pre[:, t, :], h, hw, hb, self.w_tau, self.hidden_size, timespans)
            outputs.append(h)

        # 堆叠输出 (B, T, H)
        out = torch.stack(outputs, dim=1)
        return self.dropout(out)