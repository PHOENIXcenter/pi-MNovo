"""Monotonic confidence calibration utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class CalibrationMetrics:
    brier: float
    ece: float
    nll: float
    accuracy: float
    mean_confidence: float


class PositivePlattCalibrator(torch.nn.Module):
    """Map an arbitrary score to [0, 1] while preserving score order."""

    def __init__(self) -> None:
        super().__init__()
        self.log_scale = torch.nn.Parameter(torch.zeros(()))
        self.bias = torch.nn.Parameter(torch.zeros(()))

    @property
    def scale(self) -> torch.Tensor:
        return F.softplus(self.log_scale) + 1e-6

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.scale * scores + self.bias)

    def fit(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        max_iter: int = 100,
    ) -> "PositivePlattCalibrator":
        scores = scores.detach().float().reshape(-1)
        labels = labels.detach().float().reshape(-1)
        if scores.numel() == 0 or labels.unique().numel() < 2:
            raise ValueError("Calibration requires non-empty positive and negative labels.")
        optimizer = torch.optim.LBFGS(
            self.parameters(),
            lr=0.5,
            max_iter=max_iter,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            probabilities = self(scores).clamp(1e-7, 1 - 1e-7)
            loss = F.binary_cross_entropy(probabilities, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self

    def state_dict_json(self) -> dict[str, float]:
        return {
            "log_scale": float(self.log_scale.detach().cpu()),
            "bias": float(self.bias.detach().cpu()),
            "scale": float(self.scale.detach().cpu()),
        }

    def load_json(self, values: dict[str, float]) -> "PositivePlattCalibrator":
        with torch.no_grad():
            self.log_scale.copy_(torch.tensor(float(values["log_scale"])))
            self.bias.copy_(torch.tensor(float(values["bias"])))
        return self


def calibration_metrics(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> CalibrationMetrics:
    probabilities = probabilities.detach().float().reshape(-1).clamp(1e-7, 1 - 1e-7)
    labels = labels.detach().float().reshape(-1)
    brier = torch.mean((probabilities - labels) ** 2)
    nll = F.binary_cross_entropy(probabilities, labels)
    ece = probabilities.new_zeros(())
    boundaries = torch.linspace(0, 1, n_bins + 1, device=probabilities.device)
    for idx in range(n_bins):
        left, right = boundaries[idx], boundaries[idx + 1]
        mask = (probabilities >= left) & (
            probabilities <= right if idx == n_bins - 1 else probabilities < right
        )
        if mask.any():
            weight = mask.float().mean()
            ece = ece + weight * torch.abs(
                probabilities[mask].mean() - labels[mask].mean()
            )
    return CalibrationMetrics(
        brier=float(brier.cpu()),
        ece=float(ece.cpu()),
        nll=float(nll.cpu()),
        accuracy=float(labels.mean().cpu()),
        mean_confidence=float(probabilities.mean().cpu()),
    )
