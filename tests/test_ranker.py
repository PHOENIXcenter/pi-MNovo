import torch
import pytest

from MNovo.ranker.features import (
    calibrated_confidence,
    candidate_core_features,
    isotope_aware_mass_features,
)
from MNovo.ranker.model import ResidualCandidateRanker
from MNovo.candidate_generation import merge_pmc_candidate


def test_feature_schema_and_ranker_shapes():
    features = candidate_core_features(
        [1.0, 2.0, 3.0],
        [10, 10, 11],
        0.7,
    )
    model = ResidualCandidateRanker(features.shape[-1], hidden_dim=8)
    scores, delta = model(features.unsqueeze(0))
    assert scores.shape == (1, 3)
    assert delta.shape == (1, 3)
    assert torch.isfinite(scores).all()


def test_released_confidence_is_bounded_and_monotonic():
    scores = torch.linspace(-4, 4, 200)
    probabilities = calibrated_confidence(scores, scale=1.7, bias=-0.2)
    assert bool(((probabilities >= 0) & (probabilities <= 1)).all())
    assert bool((probabilities[1:] >= probabilities[:-1]).all())


def test_released_confidence_rejects_reversed_scale():
    with pytest.raises(ValueError):
        calibrated_confidence(torch.tensor([0.0]), scale=0.0, bias=0.0)


def test_isotope_aware_mass_features_shape():
    candidate_mass = torch.tensor([[980.0, 981.003354835]])
    precursor_neutral_mass = torch.tensor([998.01])
    features = isotope_aware_mass_features(candidate_mass, precursor_neutral_mass)
    assert features.shape == (1, 2, 2)
    assert torch.isfinite(features).all()


def test_pmc_union_preserves_all_ctc_candidates_and_appends():
    merged, nlls, sources, injected = merge_pmc_candidate(
        [[1], [2], [3]],
        [1.0, 2.0, 3.0],
        [9],
        1.5,
        beam_size=3,
        mode="union",
    )
    assert merged == [[1], [2], [3], [9]]
    assert nlls == [1.0, 2.0, 3.0, 1.5]
    assert sources == ["ctc", "ctc", "ctc", "pmc_union"]
    assert injected


def test_pmc_union_keeps_existing_candidate_order_without_duplicate():
    candidates = [[1], [2], [3]]
    merged, _nlls, sources, injected = merge_pmc_candidate(
        candidates,
        [1.0, 2.0, 3.0],
        [2],
        2.0,
        beam_size=3,
        mode="union",
    )
    assert merged == candidates
    assert sources == ["ctc", "ctc", "ctc"]
    assert not injected
