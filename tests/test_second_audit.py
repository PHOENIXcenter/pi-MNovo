"""Second audit: exercise public boundaries omitted by the initial fixes."""

from types import SimpleNamespace
from contextlib import nullcontext

import numpy as np
import pytest
import torch
import yaml

from MNovo import cli
from MNovo import runtime as runtime_module
from MNovo.runtime import MNovoRuntime, RuntimeOptions, _selected


def test_single_candidate_margin_is_finite_independent_of_batch_padding():
    for scores, mask in [
        (torch.tensor([[2.0]]), torch.tensor([[True]])),
        (torch.tensor([[2.0, 9.0]]), torch.tensor([[True, False]])),
    ]:
        index, margin = _selected(scores, mask)
        assert index.item() == 0
        assert torch.isfinite(margin).all()
        assert margin.item() == 0


def test_external_config_cannot_change_inference_mass_or_preprocessing(tmp_path):
    frozen = {
        "max_charge": 10,
        "residues": {"A": 71.037114},
        "n_peaks": 800,
        "runtime": {"batch_size": 128},
    }
    (tmp_path / "config").mkdir()
    (tmp_path / "config/inference.yaml").write_text(yaml.safe_dump(frozen))
    changed = dict(frozen, max_charge=2)
    override = tmp_path / "override.yaml"
    override.write_text(yaml.safe_dump(changed))
    args = SimpleNamespace(
        model_dir=str(tmp_path),
        config=str(override),
        task="eval",
        mode=None,
        batch_size=None,
        n_workers=0,
        ctc_processes=1,
        temp_dir=None,
    )
    with pytest.raises(ValueError, match="frozen"):
        cli.apply_config(args)


def test_python_prediction_api_rejects_fractional_indices_before_coercion(monkeypatch):
    monkeypatch.setattr(
        runtime_module,
        "LmdbSpectrumIndex",
        lambda *a, **k: SimpleNamespace(env=SimpleNamespace(close=lambda: None)),
    )
    monkeypatch.setattr(runtime_module, "SpectrumDataset", lambda *a, **k: [None] * 3)
    monkeypatch.setattr(runtime_module, "DataLoader", lambda *a, **k: [])
    instance = object.__new__(MNovoRuntime)
    instance.config = dict(
        max_charge=10,
        n_peaks=800,
        min_mz=1,
        max_mz=6500,
        min_intensity=0,
        remove_precursor_tol=1,
    )
    instance.options = RuntimeOptions(n_workers=0)
    with pytest.raises(ValueError, match="integers"):
        list(instance.predict_lmdb("fixture", np.array([1.9])))


def test_incomplete_prediction_stream_cannot_publish_complete_output(
    tmp_path, monkeypatch
):
    fake = SimpleNamespace(
        config={"residues": {"A": 71.037114, "K": 128.094963}},
        candidate_statistics={},
        device="cpu",
        predict_lmdb=lambda *a: iter([]),
    )
    monkeypatch.setattr(cli, "MNovoRuntime", lambda *a, **k: fake)
    monkeypatch.setattr(cli, "LmdbTitleLookup", lambda *a: nullcontext(None))
    output = tmp_path / "result.tsv"
    output.write_text("previous valid result")
    args = SimpleNamespace(
        indices=None,
        max_samples=0,
        task="denovo",
        mode="fast",
        candidate_mode="beam-only",
        batch_size=2,
        n_workers=0,
        ctc_processes=1,
        model_dir="fixture",
        device="cpu",
        output=str(output),
        config="fixture",
    )
    with pytest.raises(RuntimeError, match="Prediction count mismatch"):
        cli.run(args, "fixture", [], 2)
    assert output.read_text() == "previous valid result"


@pytest.mark.parametrize("mixed", [False, True])
def test_singleton_bypasses_router_and_mixed_rows_keep_routing(mixed):
    from MNovo.router_schema import ROUTER_FEATURE_NAMES

    instance = object.__new__(MNovoRuntime)
    instance.device = torch.device("cpu")
    instance.r1 = lambda x: (x[..., 0], None)
    instance.r2 = lambda x: (torch.arange(x.shape[1]).float().expand(x.shape[:2]), None)
    instance.r3 = instance.r1
    instance.length_predictor = lambda embedding, precursor: torch.zeros(
        len(embedding), 40
    )
    instance._fragment_tensor = lambda data, spectra: torch.zeros(
        *data["mask"].shape, 12
    )
    instance.r1_checkpoint = {"ranker_calibrator": {"scale": 1.0, "bias": 0.0}}
    instance.fast_router_checkpoint = {
        "feature_names": ROUTER_FEATURE_NAMES,
        "feature_mean": torch.zeros(36),
        "feature_std": torch.ones(36),
        "frozen_threshold": 0.12,
    }
    seen = []

    def router(features):
        assert torch.isfinite(features).all()
        seen.append(len(features))
        return torch.tensor([[0.0, 10.0, 0.0]]).expand(len(features), 3)

    instance.fast_router = router
    batch = 2 if mixed else 1
    data = {
        "core": torch.tensor(
            [[[2.0, 0.0, 1.0], [9.0, 0.0, 0.5]], [[3.0, 0.0, 1.0], [1.0, 0.0, 0.5]]]
        )[:batch],
        "mask": torch.tensor([[True, False], [True, True]])[:batch],
        "embedding": torch.zeros(batch, 2),
        "precursor": torch.tensor([[250.0, 2.0, 126.0]]).expand(batch, 3),
        "mass0": torch.zeros(batch, 2, 1),
        "length": torch.full((batch, 2), 2.0),
        "is_pmc": torch.zeros(batch, 2, dtype=torch.bool),
    }
    indices, score, routes = instance._observable_select(data, torch.zeros(batch, 1, 2))
    assert torch.isfinite(score).all()
    assert routes == (["r1", "r2_long"] if mixed else ["r1"])
    assert indices.tolist() == ([0, 1] if mixed else [0])
    assert seen == ([1] if mixed else [])


def test_runtime_only_override_keeps_frozen_evaluation_config(tmp_path):
    (tmp_path / "config").mkdir()
    frozen = tmp_path / "config/inference.yaml"
    frozen.write_text(
        yaml.safe_dump(
            {"max_charge": 8, "runtime": {"batch_size": 128, "n_workers": 4}}
        )
    )
    override = tmp_path / "override.yaml"
    override.write_text("runtime:\n  batch_size: 7\n")
    args = SimpleNamespace(
        model_dir=str(tmp_path),
        config=str(override),
        task="eval",
        mode=None,
        batch_size=None,
        n_workers=None,
        ctc_processes=None,
        temp_dir=None,
    )
    result = cli.apply_config(args)
    assert result.config == str(frozen.resolve())
    assert (result.max_charge, result.batch_size, result.n_workers) == (8, 7, 4)


@pytest.mark.parametrize("wrong_indices", [[1, 0], [0, 0], [0, 1, 2]])
def test_wrong_or_extra_prediction_rows_preserve_previous_output(
    tmp_path, monkeypatch, wrong_indices
):
    from MNovo.runtime import Prediction

    stream = [Prediction(index, "AK", 0.5, "r1") for index in wrong_indices]
    fake = SimpleNamespace(
        config={"residues": {"A": 71.037114, "K": 128.094963}},
        candidate_statistics={},
        device="cpu",
        predict_lmdb=lambda *a: iter(stream),
    )
    monkeypatch.setattr(cli, "MNovoRuntime", lambda *a, **k: fake)
    lookup = SimpleNamespace(
        spectrum=lambda index: {
            "title": str(index),
            "precursor_charge": 2,
            "precursor_mz": 100,
        }
    )
    monkeypatch.setattr(cli, "LmdbTitleLookup", lambda *a: nullcontext(lookup))
    output = tmp_path / "result.tsv"
    output.write_text("previous valid result")
    args = SimpleNamespace(
        indices=None,
        max_samples=0,
        task="denovo",
        mode="fast",
        candidate_mode="beam-only",
        batch_size=2,
        n_workers=0,
        ctc_processes=1,
        model_dir="fixture",
        device="cpu",
        output=str(output),
        config="fixture",
    )
    with pytest.raises(RuntimeError, match="Prediction (count|index) mismatch"):
        cli.run(args, "fixture", [], 2)
    assert output.read_text() == "previous valid result"
