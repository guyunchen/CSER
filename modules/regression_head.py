import torch
import torch.nn as nn


class EmotionRegressionHead(nn.Module):

    def __init__(self, hidden_dim, output_dim=2, dropout=0.2):

        super().__init__()

        self.mlp = nn.Sequential(

            nn.Linear(hidden_dim, hidden_dim // 2),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):

        return self.mlp(x)
