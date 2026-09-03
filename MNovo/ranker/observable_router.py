"""Observable spectrum-level routing between R1, R2, and R3."""

from __future__ import annotations

from dataclasses import dataclass

import torch


class ObservablePathRouter(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 96,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 3),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def multi_positive_path_loss(
    logits: torch.Tensor,
    path_correct: torch.Tensor,
) -> torch.Tensor:
    positives = path_correct.bool().clone()
    no_correct_path = ~positives.any(dim=1)
    positives[no_correct_path, 0] = True
    return torch.logsumexp(logits, dim=1) - torch.logsumexp(
        logits.masked_fill(~positives, -1e9),
        dim=1,
    )


@dataclass
class RouteMetrics:
    spectra: int
    baseline_recall: float
    routed_recall: float
    net_gain: float
    good_flip_rate: float
    bad_flip_rate: float
    switch_fraction: float
    route_r1_fraction: float
    route_r2_fraction: float
    route_r3_fraction: float


def apply_frozen_threshold(
    logits: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probabilities = torch.softmax(logits.float(), dim=1)
    expert_probability, expert_offset = probabilities[:, 1:].max(dim=1)
    expert_index = expert_offset + 1
    expert_advantage = expert_probability - probabilities[:, 0]
    switch = expert_advantage >= float(threshold)
    route = torch.where(switch, expert_index, torch.zeros_like(expert_index))
    return route, probabilities, expert_advantage


def route_metrics(
    route: torch.Tensor,
    path_correct: torch.Tensor,
) -> RouteMetrics:
    path_correct = path_correct.bool()
    row = torch.arange(route.numel())
    baseline = path_correct[:, 0]
    routed = path_correct[row, route.long()]
    denominator = max(int(route.numel()), 1)
    return RouteMetrics(
        spectra=int(route.numel()),
        baseline_recall=float(baseline.float().mean()),
        routed_recall=float(routed.float().mean()),
        net_gain=float(routed.float().mean() - baseline.float().mean()),
        good_flip_rate=float((~baseline & routed).sum()) / denominator,
        bad_flip_rate=float((baseline & ~routed).sum()) / denominator,
        switch_fraction=float((route != 0).float().mean()),
        route_r1_fraction=float((route == 0).float().mean()),
        route_r2_fraction=float((route == 1).float().mean()),
        route_r3_fraction=float((route == 2).float().mean()),
    )
