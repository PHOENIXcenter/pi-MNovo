import torch

from MNovo.ranker.cache import candidate_features, mnovo_confidence
from MNovo.ranker.calibration import PositivePlattCalibrator
from MNovo.ranker.model import ResidualCandidateRanker
from MNovo.candidate_generation import merge_pmc_candidate


def test_mnovo_confidence_range():
    assert mnovo_confidence(0.0) == 1.0
    assert 0.0 <= mnovo_confidence(10.0) <= 1.0
    assert mnovo_confidence(float("inf")) == 0.0


def test_feature_schema_and_ranker_shapes():
    features = candidate_features(
        [1.0, 2.0, 3.0],
        [10, 10, 11],
        [5.0, 10.0, 2.0],
        ["ctc", "ctc", "pmc"],
        0.7,
    )
    model = ResidualCandidateRanker(features.shape[-1], hidden_dim=8)
    scores, delta = model(features.unsqueeze(0))
    assert scores.shape == (1, 3)
    assert delta.shape == (1, 3)
    assert torch.isfinite(scores).all()


def test_calibrator_is_bounded_and_monotonic():
    scores = torch.linspace(-4, 4, 200)
    labels = (scores > 0).float()
    calibrator = PositivePlattCalibrator().fit(scores, labels, max_iter=30)
    probabilities = calibrator(scores)
    assert bool(((probabilities >= 0) & (probabilities <= 1)).all())
    assert bool((probabilities[1:] >= probabilities[:-1]).all())


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
