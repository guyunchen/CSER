import torch
import torch.nn as nn

from data_utils.normalization import lengths_to_mask
from modules.TemporalConvEncoder import TemporalConvEncoder
from modules.ls4_CTLN import ConservativeRobustFrontend, GatedRobustFrontend, IdentityFrontend
from modules.noiseResilient import RobustSpectralRefinement
from modules.regression_head import EmotionRegressionHead
from modules.temporal_attention import TemporalAttention

from .original_ls4 import OriginalLS4Layer


class OriginalLS4_CTLN(nn.Module):
    """
    CTLN architecture with the current LiquidS4Layer replaced by OriginalLS4Layer.
    """

    def __init__(
        self,
        input_dim=80,
        hidden_dim=256,
        num_ls4_layers=2,
        d_state=64,
        p_order=2,
        liquid_kernel="polyb",
        output_dim=3,
        dropout=0.3,
        noise_frontend="identity",
        model_name=None,
    ):
        super().__init__()
        del model_name

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

        self.encoder = TemporalConvEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.ls4_layers = nn.ModuleList()
        for _ in range(num_ls4_layers):
            self.ls4_layers.append(nn.ModuleDict({
                "ls4": OriginalLS4Layer(
                    hidden_dim,
                    d_state=d_state,
                    p_order=p_order,
                    liquid_kernel=liquid_kernel,
                    dropout=dropout,
                ),
                "norm": nn.LayerNorm(hidden_dim),
                "dropout": nn.Dropout(dropout),
                "activation": nn.GELU(),
            }))

        self.temporal_attention = TemporalAttention(hidden_dim)
        self.regression_head = EmotionRegressionHead(hidden_dim, output_dim, dropout)

    def forward(self, x, lengths=None, return_attention=False):
        x = self.noise_block(x)
        features = self.encoder(x)
        encoded_lengths = self.encoder.downsample_lengths(lengths)
        mask = lengths_to_mask(encoded_lengths, features.size(1)) if encoded_lengths is not None else None

        for layer in self.ls4_layers:
            residual = features
            features = layer["ls4"](features, lengths=encoded_lengths)
            features = layer["activation"](features)
            features = layer["norm"](features + residual)
            features = layer["dropout"](features)

        pooled, attention = self.temporal_attention(features, mask=mask)
        emotion = self.regression_head(pooled)

        outputs = {"emotion": emotion}
        if return_attention:
            outputs["attention"] = attention

        return outputs
