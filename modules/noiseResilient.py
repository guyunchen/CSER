import torch
import torch.nn as nn
import torch.nn.functional as F


class RobustSpectralRefinement(nn.Module):
    """
    Lightweight Noise-Robust Spectral Refinement
    for Robust Speech Emotion Recognition
    """

    def __init__(self, input_dim=80, reduction=4):
        super().__init__()

        hidden_dim = input_dim // 2

        # 1️⃣ Spectral Refinement
        self.spectral_refinement = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )

        # 2️⃣ Frequency Attention
        self.freq_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // reduction),
            nn.GELU(),
            nn.Linear(input_dim // reduction, input_dim),
            nn.Sigmoid()
        )

        # 3️⃣ Noise-aware Gate
        self.noise_gate = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

        # Residual scaling
        self.residual_scale = nn.Parameter(torch.tensor(0.5))

        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        """
        x: [B, T, F]
        """

        residual = x

        # ---------------------------------
        # 1️⃣ Spectral refinement
        # ---------------------------------
        enhanced = self.spectral_refinement(x)

        # ---------------------------------
        # 2️⃣ Frequency attention
        # combine raw + enhanced features
        # ---------------------------------
        attention_input = x + enhanced

        freq_weights = self.freq_attention(attention_input)

        enhanced = enhanced * freq_weights

        # ---------------------------------
        # 3️⃣ Lightweight noise statistics
        # temporal std as noise hint
        # ---------------------------------
        noise_hint = torch.std(x, dim=1, keepdim=True)

        noise_hint = noise_hint.expand_as(x)

        gate_input = torch.cat([enhanced, noise_hint], dim=-1)

        gate = self.noise_gate(gate_input)

        # ---------------------------------
        # 4️⃣ Soft suppression
        # avoid destroying emotion cues
        # ---------------------------------
        filtered = enhanced * (1.0 + 0.3 * gate)

        # ---------------------------------
        # 5️⃣ Residual fusion
        # ---------------------------------
        out = filtered + self.residual_scale * residual

        out = self.norm(out)

        return out


class ConservativeRobustRefinement(nn.Module):
    """
    Conservative denoising frontend centered on the identity mapping.

    It estimates short-term high-frequency residuals and suppresses only a
    small, gated part of them. The global mix is capped so clean emotion cues
    cannot be fully overwritten by the denoising branch.
    """

    def __init__(
        self,
        input_dim=80,
        reduction=4,
        kernel_size=5,
        max_mix=0.35,
        init_mix_logit=-2.2,
        max_delta=0.25,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")

        hidden_dim = max(input_dim // reduction, 16)
        self.kernel_size = kernel_size
        self.max_mix = max_mix
        self.max_delta = max_delta
        self.mix_logit = nn.Parameter(torch.tensor(float(init_mix_logit)))

        self.noise_gate = nn.Sequential(
            nn.Linear(input_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )
        self.residual_correction = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )
        nn.init.zeros_(self.residual_correction[-1].weight)
        nn.init.zeros_(self.residual_correction[-1].bias)

    def forward(self, x):
        if x.size(1) <= 1:
            return x

        pad = self.kernel_size // 2
        x_t = x.transpose(1, 2)
        smooth = F.avg_pool1d(
            F.pad(x_t, (pad, pad), mode="replicate"),
            kernel_size=self.kernel_size,
            stride=1,
        ).transpose(1, 2)

        high_residual = x - smooth
        high_abs = high_residual.abs()
        seq_std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
        seq_std = seq_std.expand_as(x)

        noise_score = high_abs / seq_std
        heuristic_gate = torch.sigmoid(2.0 * (noise_score - 1.0))
        learned_gate = self.noise_gate(torch.cat([x, high_abs, seq_std], dim=-1))
        gate = heuristic_gate * learned_gate

        learned_delta = torch.tanh(self.residual_correction(high_residual)) * self.max_delta
        denoised = x - gate * high_residual + 0.1 * learned_delta
        delta = self.max_delta * torch.tanh((denoised - x) / self.max_delta)
        mix = self.max_mix * torch.sigmoid(self.mix_logit)
        return x + mix * delta
