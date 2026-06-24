import torch
import torch.nn as nn
import numpy as np


class S4Layer(nn.Module):
    """
    标准的 S4 (Structured State Space) 层实现
    基于 S4D (Diagonal) 的参数化方式
    """

    def __init__(self, d_model, d_state=64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # S4 参数 (对角化参数化)
        # 使用 log 空间确保 A 的实部为负，保证系统稳定（衰减记忆）
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()))
        self.B = nn.Parameter(torch.randn(d_model, d_state) / np.sqrt(d_state))
        self.C = nn.Parameter(torch.randn(d_model, d_state) / np.sqrt(d_state))
        self.D = nn.Parameter(torch.ones(d_model))  # Skip connection

    def forward(self, u):
        # u shape: (B, L, D) -> 转置为 (B, D, L) 以适配卷积
        u_orig = u
        u = u.transpose(-1, -2)
        batch, dim, L = u.shape

        # 1. 生成标准的线性 SSM 卷积核 K
        # 这一部分是静态的，不依赖于 u 的具体数值
        dt = 1.0 / L
        A = -torch.exp(self.A_log)  # (d_state)

        # 计算 exp(A * t)
        t = torch.arange(L, device=u.device) * dt
        # (d_state, L)
        v = torch.exp(A.unsqueeze(-1) * t.unsqueeze(0))

        # K = C * exp(At) * B -> (D_model, L)
        # 这是标准的 S4 卷积核，用于捕捉长程依赖
        K = torch.einsum('ms,sl,ms->ml', self.C, v, self.B)

        # 2. 使用 FFT 进行高效卷积
        L_fft = 2 * L
        u_f = torch.fft.rfft(u, n=L_fft)
        K_f = torch.fft.rfft(K, n=L_fft)
        y = torch.fft.irfft(u_f * K_f, n=L_fft)[..., :L]

        # 3. Skip connection (D 项)
        y = y + u * self.D.unsqueeze(-1)

        return y.transpose(-1, -2)  # 返回 (B, L, D)