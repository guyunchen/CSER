import math

import numpy as np
import torch
import torch.nn as nn

from data_utils.normalization import lengths_to_mask
from losses.ccc_loss import ImprovedMultiTaskLoss
from modules.TemporalConvEncoder import TemporalConvEncoder
from modules.S4 import S4Layer
from modules.liquidS4 import DALS4Layer
from modules.liquid_layer import OptimizedCfCLayer
from modules.ls4_CTLN import (
    ConservativeRobustFrontend,
    GatedRobustFrontend,
    IdentityFrontend,
    SubBandDeepFilterLiteFrontend,
)
from modules.noiseResilient import RobustSpectralRefinement
from modules.regression_head import EmotionRegressionHead
from modules.temporal_attention import TemporalAttention
from models.original_ls4 import OriginalLS4Layer


class StaticDALS4Layer(nn.Module):
    def __init__(self, d_model, d_state=64, p_order=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.p_order = p_order
        hippo_scale = torch.arange(1, d_state + 1).float()
        self.register_buffer("A_base", -hippo_scale)
        self.A_log = nn.Parameter(torch.zeros(d_state))
        self.B = nn.Parameter(torch.randn(d_model, d_state) / np.sqrt(d_state))
        self.C = nn.Parameter(torch.randn(d_model, d_state) / np.sqrt(d_state))
        self.D = nn.Parameter(torch.ones(d_model))
        self.liquid_weights = nn.Parameter(torch.randn(max(p_order - 1, 0), d_model) / math.sqrt(d_model))

    def forward(self, u):
        u_conv = u.transpose(-1, -2)
        _, length, _ = u.shape
        dt = 1.0 / length
        t = torch.arange(length, device=u.device) * dt
        a = self.A_base * torch.exp(self.A_log)
        v = torch.exp(a.unsqueeze(-1) * t.unsqueeze(0))
        bc = self.B * self.C
        kernel = torch.einsum("sl,ms->ml", v, bc).unsqueeze(0)
        fft_len = 2 * length
        y = torch.fft.irfft(
            torch.fft.rfft(u_conv, n=fft_len) * torch.fft.rfft(kernel, n=fft_len),
            n=fft_len,
        )[..., :length]
        for p in range(self.p_order - 1):
            y = y + u_conv * self.liquid_weights[p].unsqueeze(-1)
        y = y + u_conv * self.D.unsqueeze(-1)
        return y.transpose(-1, -2)


class MeanStdPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.out_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x, mask=None):
        if mask is not None:
            mask = mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            mean = (x * mask).sum(dim=1) / denom
            variance = (((x - mean.unsqueeze(1)) ** 2) * mask).sum(dim=1) / denom
        else:
            mean = x.mean(dim=1)
            variance = x.var(dim=1, unbiased=False)
        pooled = self.out_proj(torch.cat([mean, torch.sqrt(torch.relu(variance) + 1e-6)], dim=-1))
        weights = torch.zeros(x.size(0), x.size(1), device=x.device, dtype=x.dtype)
        return pooled, weights


class UnifiedSERModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_dim = config["input_dim"]
        hidden_dim = config["hidden_dim"]
        dropout = config["dropout"]
        self.sequence_core = "none" if config.get("remove_sequence_core") else config.get("sequence_core", "da_ls4")

        noise_frontend = config.get("noise_frontend", "identity")
        if noise_frontend == "identity":
            self.noise_block = IdentityFrontend()
        elif noise_frontend == "robust":
            self.noise_block = RobustSpectralRefinement(input_dim)
        elif noise_frontend == "gated_robust":
            self.noise_block = GatedRobustFrontend(input_dim)
        elif noise_frontend == "conservative_robust":
            self.noise_block = ConservativeRobustFrontend(input_dim)
        elif noise_frontend == "subband_df_lite":
            self.noise_block = SubBandDeepFilterLiteFrontend(input_dim)
        else:
            raise ValueError(f"Unsupported noise_frontend: {noise_frontend}")

        self.encoder = TemporalConvEncoder(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.layers = nn.ModuleList()
        num_layers = config["num_layers"]
        d_state = config["d_state"]
        p_order = config["p_order"]
        for _ in range(num_layers):
            if self.sequence_core in {"da_ls4", "ls4"}:
                core = (
                    StaticDALS4Layer(hidden_dim, d_state=d_state, p_order=p_order)
                    if config.get("disable_ls4_dynamic")
                    else DALS4Layer(hidden_dim, d_state=d_state, p_order=p_order)
                )
            elif self.sequence_core == "original_ls4":
                core = OriginalLS4Layer(
                    hidden_dim,
                    d_state=d_state,
                    p_order=p_order,
                    liquid_kernel=config.get("original_liquid_kernel", "polyb"),
                    dropout=dropout,
                )
            elif self.sequence_core == "s4":
                core = S4Layer(hidden_dim, d_state=d_state)
            elif self.sequence_core == "cfc":
                core = OptimizedCfCLayer(hidden_dim, hidden_dim, dropout=dropout)
            elif self.sequence_core == "none":
                core = nn.Identity()
            else:
                raise ValueError(f"Unsupported sequence_core: {self.sequence_core}")
            self.layers.append(
                nn.ModuleDict(
                    {
                        "core": core,
                        "norm": nn.LayerNorm(hidden_dim),
                        "dropout": nn.Dropout(dropout),
                        "activation": nn.GELU(),
                    }
                )
            )

        self.pooling = MeanStdPooling(hidden_dim) if config.get("remove_attention") else TemporalAttention(hidden_dim)
        self.regression_head = EmotionRegressionHead(hidden_dim, config["output_dim"], dropout)

    def forward(self, x, lengths=None):
        x = self.noise_block(x)
        features = self.encoder(x)
        encoded_lengths = self.encoder.downsample_lengths(lengths)
        mask = lengths_to_mask(encoded_lengths, features.size(1)) if encoded_lengths is not None else None

        for layer in self.layers:
            residual = features
            if self.sequence_core == "original_ls4":
                features = layer["core"](features, lengths=encoded_lengths)
            else:
                features = layer["core"](features)
            if self.sequence_core != "none":
                features = layer["activation"](features)
                features = layer["norm"](features + residual)
                features = layer["dropout"](features)

        pooled, attention = self.pooling(features, mask=mask)
        return {"emotion": self.regression_head(pooled), "attention": attention}


def build_loss(loss_cfg, output_dim):
    cfg = dict(loss_cfg)
    loss_type = cfg.pop("loss_type", "improved")
    if loss_type == "ccc_only":
        cfg.update({"ccc_weight": 1.0, "corr_weight": 0.0, "mse_weight": 0.0, "smooth_weight": 0.0})
        cfg.setdefault("task_importance", [1.0] * output_dim)
    elif loss_type != "improved":
        raise ValueError(f"Unsupported loss_type: {loss_type}")
    return ImprovedMultiTaskLoss(num_tasks=output_dim, **cfg)
