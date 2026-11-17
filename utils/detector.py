"""Shared training/inference architecture; preserves checkpoint parameter names."""
import torch.nn as nn


class PrefillMLPDetector(nn.Module):
    """Paper Appendix A.3 MLP detector for one token's cross-layer hidden states."""

    def __init__(self, input_dim: int, hidden_dim: int, encoder_out_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, encoder_out_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(encoder_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled)

