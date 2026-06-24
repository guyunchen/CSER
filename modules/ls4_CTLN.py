import torch
import torch.nn as nn
import torch.nn.functional as F

from .noiseResilient import ConservativeRobustRefinement, RobustSpectralRefinement
from .TemporalConvEncoder import TemporalConvEncoder
from .liquidS4 import DALS4Layer
from .temporal_attention import TemporalAttention
from .regression_head import EmotionRegressionHead
from data_utils.normalization import lengths_to_mask


class IdentityFrontend(nn.Module):
    def forward(self, x):
        return x


class GatedRobustFrontend(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.robust = RobustSpectralRefinement(input_dim)
        self.mix_logit = nn.Parameter(torch.tensor(-6.0))

    def forward(self, x):
        mix = torch.sigmoid(self.mix_logit)
        return x + mix * (self.robust(x) - x)


class ConservativeRobustFrontend(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.robust = ConservativeRobustRefinement(input_dim)

    def forward(self, x):
        return self.robust(x)


class LiteGLSER(nn.Module):
    """
    Lite-GLSER: Lightweight Gated Liquid Speech Emotion Regression network.

    The model combines a gated robust acoustic frontend, a lightweight
    convolutional encoder, stacked DA-LS4 modules, temporal attention pooling,
    and a continuous emotion regression head.
    """
    def __init__(
            self,
            input_dim=80,
            hidden_dim=256,
            num_ls4_layers=2,
            num_da_ls4_layers=None,
            d_state=64,
            p_order=2,
            output_dim=2,
            # output_dim=3, # 修改：默认设为 3
            dropout=0.3,
            noise_frontend="identity"
    ):
        super().__init__()
        num_layers = num_da_ls4_layers if num_da_ls4_layers is not None else num_ls4_layers

        if noise_frontend == "robust":
            self.noise_block = RobustSpectralRefinement(input_dim)
        elif noise_frontend == "gated_robust":
            self.noise_block = GatedRobustFrontend(input_dim)
        elif noise_frontend == "conservative_robust":
            self.noise_block = ConservativeRobustFrontend(input_dim)
        elif noise_frontend == "identity":
            self.noise_block = IdentityFrontend()
        else:
            raise ValueError(f"Unsupported noise_frontend: {noise_frontend}")

        # 2️⃣ 特征编码器
        self.encoder = TemporalConvEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        # 3. DA-LS4 layers
        self.da_ls4_layers = nn.ModuleList()
        for _ in range(num_layers):
            da_ls4 = DALS4Layer(hidden_dim, d_state=d_state, p_order=p_order)
            self.da_ls4_layers.append(nn.ModuleDict({
                "da_ls4": da_ls4,
                "ls4": da_ls4,
                "norm": nn.LayerNorm(hidden_dim),
                "dropout": nn.Dropout(dropout),
                "activation": nn.GELU()
            }))
        self.ls4_layers = self.da_ls4_layers

        # 4️⃣ 时间注意力
        self.temporal_attention = TemporalAttention(hidden_dim)

        # 5️⃣ 回归预测头
        self.regression_head = EmotionRegressionHead(
            hidden_dim,
            output_dim,
            dropout
        )

    def forward(self, x, lengths=None, return_attention=False):
        x = self.noise_block(x)
        features = self.encoder(x)
        encoded_lengths = self.encoder.downsample_lengths(lengths)
        mask = lengths_to_mask(encoded_lengths, features.size(1)) if encoded_lengths is not None else None

        for layer in self.da_ls4_layers:
            residual = features
            features = layer["da_ls4"](features)
            features = layer["activation"](features)
            features = layer["norm"](features + residual)
            features = layer["dropout"](features)

        pooled, attention = self.temporal_attention(features, mask=mask)
        emotion = self.regression_head(pooled)

        outputs = {"emotion": emotion}
        if return_attention:
            outputs["attention"] = attention

        return outputs

    def load_state_dict(self, state_dict, strict=True, assign=False):
        migrated = dict(state_dict)
        for key, value in list(state_dict.items()):
            layer_prefixes = [key]
            if key.startswith("ls4_layers."):
                layer_prefixes.append(key.replace("ls4_layers.", "da_ls4_layers.", 1))
            if key.startswith("da_ls4_layers."):
                layer_prefixes.append(key.replace("da_ls4_layers.", "ls4_layers.", 1))

            for prefixed_key in layer_prefixes:
                migrated.setdefault(prefixed_key, value)
                if ".ls4." in prefixed_key:
                    migrated.setdefault(prefixed_key.replace(".ls4.", ".da_ls4."), value)
                if ".da_ls4." in prefixed_key:
                    migrated.setdefault(prefixed_key.replace(".da_ls4.", ".ls4."), value)
        return super().load_state_dict(migrated, strict=strict, assign=assign)


# Backward-compatible alias for existing checkpoints, configs, and scripts.
LS4_CTLN = LiteGLSER
