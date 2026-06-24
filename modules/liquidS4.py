import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DALS4Layer(nn.Module):
    """
    Dynamic Adaptive Liquid State-Space layer (DA-LS4).

    This is the improved state-space module used by Lite-GLSER. Compared with
    a static LS4-style layer, it adapts the state transition dynamics with an
    input-conditioned gate and a learned dynamic lambda term.
    """

    def __init__(self, d_model, d_state=64, p_order=2, lambda_hidden=32):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.p_order = p_order

        # --- A: HiPPO-like base (HiPPO 矩阵的简化版，用于长短期记忆平衡) ---
        hippo_scale = torch.arange(1, d_state + 1).float()
        self.register_buffer("A_base", -hippo_scale)
        self.A_log = nn.Parameter(torch.zeros(d_state))

        # --- B, C, D 参数 ---
        self.B = nn.Parameter(torch.randn(d_model, d_state) / np.sqrt(d_state))
        self.C = nn.Parameter(torch.randn(d_model, d_state) / np.sqrt(d_state))
        self.D = nn.Parameter(torch.ones(d_model))

        # --- 轻量液体门控 (用于根据输入调节系统时间常数) ---
        self.A_gate = nn.Linear(d_model, d_state, bias=False)

        # --- Liquid correction weights ---
        self.liquid_weights = nn.Parameter(
            torch.randn(p_order - 1, d_model) / np.sqrt(d_model)
        )

        # --- 动态 λ 生成 MLP (让系统动力学随输入变化) ---
        self.lambda_mlp = nn.Sequential(
            nn.Linear(d_model, lambda_hidden),
            nn.ReLU(),
            nn.Linear(lambda_hidden, d_state)
        )

    def forward(self, u):
        """
        u: (B, L, d_model)
        """
        u_orig = u
        B, L, D_feat = u.shape

        # 转换维度用于卷积: (B, d_model, L)
        u_conv = u.transpose(-1, -2)

        # 归一化时间步
        dt = 1.0 / L
        t = torch.arange(L, device=u.device) * dt  # (L,)

        # --- 1. 计算基础 A 矩阵 ---
        A = self.A_base * torch.exp(self.A_log)  # (d_state)

        # --- 2. 液体动力学：计算动态 λ 和 门控 ---
        # 使用序列的平均特征来指导该序列的动力学系数
        u_mean = u_orig.mean(dim=1)  # (B, d_model)
        delta_lambda = self.lambda_mlp(u_mean)  # (B, d_state)
        g = torch.sigmoid(self.A_gate(u_mean))  # (B, d_state)

        # 结合基础 A 与动态调节因子
        # A_dyn: (B, d_state)
        A_dyn = A.unsqueeze(0) * (1 + g) + delta_lambda

        # --- 3. 构造卷积核 (Continuous-time kernel) ---
        # v: (B, d_state, L)
        v = torch.exp(A_dyn.unsqueeze(-1) * t.unsqueeze(0))

        # 结合 B 和 C 矩阵得到最终核 K
        BC = self.B * self.C  # (d_model, d_state)
        K = torch.einsum('bsl,ms->bml', v, BC)  # (B, d_model, L)

        # --- 4. FFT 卷积 ---
        L_fft = 2 * L
        u_f = torch.fft.rfft(u_conv, n=L_fft)
        K_f = torch.fft.rfft(K, n=L_fft)
        y = torch.fft.irfft(u_f * K_f, n=L_fft)[..., :L]

        # --- 5. Liquid correction (高阶修正) ---
        for p in range(self.p_order - 1):
            w = self.liquid_weights[p]
            y = y + u_conv * w.unsqueeze(-1)

        # --- 6. Skip connection (D 项) ---
        y = y + u_conv * self.D.unsqueeze(-1)

        return y.transpose(-1, -2)  # 还原回 (B, L, d_model)


# Backward-compatible alias for existing checkpoints, configs, and scripts.
LiquidS4Layer = DALS4Layer
