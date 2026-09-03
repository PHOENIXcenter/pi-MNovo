"""Spectrum-conditioned peptide-length prediction."""

from __future__ import annotations

import torch


class SpectrumLengthPredictor(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        max_length: int = 40,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.max_length = int(max_length)
        self.network = torch.nn.Sequential(
            torch.nn.LayerNorm(self.embedding_dim + 2),
            torch.nn.Linear(self.embedding_dim + 2, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, self.max_length),
        )

    def forward(
        self,
        spectrum_embedding: torch.Tensor,
        precursor: torch.Tensor,
    ) -> torch.Tensor:
        neutral_mass = torch.log1p(precursor[:, 0].float().clamp_min(0.0)) / 10.0
        charge = precursor[:, 1].float() / 10.0
        context = torch.cat(
            [
                spectrum_embedding.float(),
                neutral_mass.unsqueeze(1),
                charge.unsqueeze(1),
            ],
            dim=1,
        )
        return self.network(context)


def length_candidate_features(
    logits: torch.Tensor,
    candidate_length: torch.Tensor,
) -> torch.Tensor:
    """Build candidate log-probability, deviation, and top-1-relative features."""
    log_prob = torch.log_softmax(logits.float(), dim=-1)
    max_length = int(log_prob.size(1))
    indices = candidate_length.long().clamp(1, max_length) - 1
    candidate_log_prob = torch.gather(log_prob, 1, indices)
    lengths = torch.arange(
        1,
        max_length + 1,
        device=log_prob.device,
        dtype=log_prob.dtype,
    )
    expected = torch.sum(log_prob.exp() * lengths.unsqueeze(0), dim=1, keepdim=True)
    deviation = -(candidate_length.float() - expected).abs() / 10.0
    improvement = candidate_log_prob - candidate_log_prob[:, :1]
    return torch.stack(
        [candidate_log_prob, deviation, improvement],
        dim=-1,
    )
