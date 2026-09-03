"""Candidate-level residual ranker."""

from __future__ import annotations

import torch


class ResidualCandidateRanker(torch.nn.Module):
    """Learn a bounded residual over MNovo's length-normalized CTC score."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 96,
        dropout: float = 0.05,
        residual_scale: float = 0.5,
    ) -> None:
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.input_norm = torch.nn.LayerNorm(input_dim)
        self.scorer = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        baseline = features[..., 0]
        delta = self.scorer(self.input_norm(features)).squeeze(-1)
        final_score = baseline + self.residual_scale * torch.tanh(delta)
        return final_score, delta
