"""Candidate features and confidence mapping used by release inference."""

from __future__ import annotations

import torch


CORE_FEATURE_NAMES = [
    "ctc_length_normalized_score",
    "score_delta_to_top1",
    "reciprocal_rank",
]


def candidate_core_features(
    beam_nlls: list[float],
    lengths: list[int],
    length_alpha: float,
) -> torch.Tensor:
    """Build the three shared candidate-ranking features."""
    if not beam_nlls:
        raise ValueError("At least one candidate is required.")
    if len(lengths) != len(beam_nlls):
        raise ValueError("Candidate feature inputs must have equal lengths.")

    alpha = max(float(length_alpha), 1e-6)
    scores = [
        -float(nll) / (max(int(length), 1) ** alpha)
        for nll, length in zip(beam_nlls, lengths)
    ]
    top_score = scores[0]
    return torch.tensor(
        [
            [score, score - top_score, 1.0 / float(rank + 1)]
            for rank, score in enumerate(scores)
        ],
        dtype=torch.float32,
    )


def isotope_aware_mass_features(
    candidate_mass: torch.Tensor,
    precursor_neutral_mass: torch.Tensor,
    isotope_errors: tuple[int, ...] = (0, 1),
    isotope_mass: float = 1.003354835,
) -> torch.Tensor:
    """Return isotope-aware absolute ppm error and improvement over top-1."""
    target = precursor_neutral_mass.float().unsqueeze(1) - 18.01
    errors = []
    for isotope in isotope_errors:
        corrected = target - float(isotope) * float(isotope_mass)
        ppm = (
            (candidate_mass.float() - corrected).abs()
            / corrected.abs().clamp_min(1e-6)
            * 1e6
        )
        errors.append(ppm)
    best_ppm = torch.stack(errors, dim=-1).amin(dim=-1)
    top_ppm = best_ppm[:, :1]
    return torch.stack(
        [
            -best_ppm.clamp_max(1000.0) / 100.0,
            (top_ppm - best_ppm).clamp(min=-1000.0, max=1000.0) / 100.0,
        ],
        dim=-1,
    )


def calibrated_confidence(
    raw_score: torch.Tensor,
    scale: float,
    bias: float,
) -> torch.Tensor:
    """Map an R1 raw score monotonically to the released [0, 1] Score."""
    if scale <= 0:
        raise ValueError("Confidence calibration scale must be positive.")
    return torch.sigmoid(float(scale) * raw_score + float(bias))
