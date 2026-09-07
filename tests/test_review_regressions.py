from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch
import yaml

from MNovo.ctc import required_steps, validate_targets
from MNovo.evaluation import tokens
from MNovo.selection import validate_selection
from MNovo.candidate_generation import ctc_nll_for_batch, pmc_candidates
from MNovo.denovo.metric_counts import CountRatio, match_counts
from MNovo.denovo.data import DeNovoDataModule
from MNovo.denovo import mass_con
from MNovo.pmc_schema import PMC_RESIDUES, validate_pmc_schema
from MNovo.runtime import MNovoRuntime, RuntimeOptions, _selected
from MNovo.release import resolve_model_release, REQUIRED


@pytest.mark.parametrize(
    "value", ["AXK", "AS+79.966K", "A[Oxidation]SK", "A+42.011K", "+42.011", "A$K"]
)
def test_unknown_or_misplaced_tokens_rejected(value):
    with pytest.raises(ValueError):
        tokens(value)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("", []),
        (None, []),
        ("ILK", ["L", "L", "K"]),
        ("M[UNIMOD:35]C[Carbamidomethyl]K", ["M+15.995", "C+57.021", "K"]),
        ("+43.006-17.027AN[+0.984]K", ["+43.006-17.027", "A", "N+0.984", "K"]),
    ],
)
def test_supported_tokens(value, expected):
    assert tokens(value) == expected


def test_ctc_repeat_feasibility_and_real_nll():
    model = SimpleNamespace(decoder=SimpleNamespace(get_blank_idx=lambda: 0))
    probs = torch.log_softmax(torch.tensor([[[0.0, 2.0, 1.0], [0.0, 1.0, 2.0]]]), -1)
    assert required_steps([1, 1]) == 3
    assert math.isinf(ctc_nll_for_batch(model, probs, [[1, 1]])[0])
    assert math.isfinite(ctc_nll_for_batch(model, probs, [[1, 2]])[0])
    with pytest.raises(ValueError, match="CTC-unalignable"):
        validate_targets(torch.tensor([[1, 1]]), [2], 2, 0)
    validate_targets(torch.tensor([[1, 2]]), [2], 2, 0)
    assert math.isinf(ctc_nll_for_batch(model, probs, [[0]])[0])


def test_metrics_truth_first_empty_internal_stop_and_short_tail():
    masses = {"A": 71.037114, "K": 128.094963}
    correct, true, predicted, peptides, total = match_counts(
        ["AK", "AK"], ["AK", ""], masses
    )
    assert (correct, true, predicted, peptides, total) == (2, 4, 2, 1, 2)
    assert correct / predicted == 1
    assert correct / true == 0.5
    assert match_counts(["AK"], ["A$K"], masses) == (0, 2, 0, 0, 1)
    metric = CountRatio()
    metric.update(1, 2)
    metric.update(0, 1)
    assert float(metric.compute()) == pytest.approx(1 / 3)
    metric.reset()
    assert int(metric.denominator) == 0


def test_fit_does_not_construct_or_prepare_test(monkeypatch):
    import MNovo.denovo.data as data

    seen = []
    monkeypatch.setattr(
        data, "LmdbSpectrumIndex", lambda path, *a, **k: seen.append(path) or path
    )
    monkeypatch.setattr(data, "SpectrumDataset", lambda paths, **k: paths)
    module = DeNovoDataModule(
        train_index_path=["train"],
        val_index_path=["valid"],
        test_index_path=["forbidden_test"],
        mode="fit",
    )
    module.prepare_data()
    module.setup("fit")
    assert "forbidden_test" not in seen
    assert module.test_dataset is None
    monkeypatch.setattr(module, "_make_loader", lambda dataset: dataset)
    assert module.val_dataloader() == ["valid"]


def test_pmc_failure_propagates_and_disabled_never_calls(monkeypatch):
    def fail(*a):
        raise RuntimeError("simulated CUDA failure")

    monkeypatch.setattr(mass_con, "knapDecode", fail)
    model = SimpleNamespace(
        PMC_enable=True,
        mass_control_tol=0.1,
        decoder=SimpleNamespace(
            _idx2aa={1: "A"}, _peptide_mass=SimpleNamespace(masses={"A": 71.0})
        ),
    )
    stats = dict(pmc_attempted=0, pmc_generated=0, pmc_failed=0)
    with pytest.raises(RuntimeError, match="union inference aborted"):
        pmc_candidates(
            model,
            torch.zeros(1, 40, 28),
            torch.tensor([[500.0, 2.0, 251.0]]),
            [[1]],
            True,
            stats,
        )
    assert stats["pmc_failed"] == 1 and stats["pmc_attempted"] == 1
    assert pmc_candidates(
        model, torch.zeros(1, 40, 28), torch.tensor([[500.0, 2.0, 251.0]]), [[1]], False
    ) == [[]]


@pytest.mark.parametrize(
    "indices",
    [
        np.array([-1]),
        np.array([3]),
        np.array([1, 1]),
        np.array([]),
        np.array([[1]]),
        np.array([1.2]),
        np.array([True]),
    ],
)
def test_invalid_selection(indices):
    with pytest.raises(ValueError):
        validate_selection(indices, 3)


def test_subset_eval_and_negative_budget_rejected():
    with pytest.raises(ValueError, match="all input spectra"):
        validate_selection(None, 3, 1, "eval")
    with pytest.raises(ValueError, match="non-negative"):
        validate_selection(None, 3, -1)
    assert validate_selection(np.array([2, 0]), 3).tolist() == [2, 0]


def test_empty_pool_bypasses_ranker_router_calibrator():
    runtime = object.__new__(MNovoRuntime)
    runtime.options = RuntimeOptions(candidate_mode="beam-only")
    runtime.device = torch.device("cpu")
    runtime.backbone = SimpleNamespace(
        encoder=lambda x: (torch.zeros(len(x), 1, 2), None),
        decoder=lambda *args: (torch.zeros(2, 2, 3), None, None),
    )
    runtime.decoder5 = SimpleNamespace(decode_all=lambda p: (None, None))
    runtime._rows_from_decode = lambda *a: [{"texts": []}, {"texts": []}]

    def forbidden(*a):
        raise AssertionError("empty pool entered ranker")

    runtime._observable_select = forbidden
    predictions = runtime.predict_batch(
        torch.zeros(2, 1, 2), torch.zeros(2, 3), torch.tensor([5, 8])
    )
    assert [p.dataset_index for p in predictions] == [5, 8]
    assert all(
        p.status == "no_valid_candidate" and p.route == "none" and p.confidence == 0
        for p in predictions
    )
    with pytest.raises(ValueError, match="empty candidate"):
        _selected(torch.zeros(1, 2), torch.zeros(1, 2, dtype=torch.bool))


def test_pmc_schema_order_mass_length_and_kernel_shape():
    decoder = SimpleNamespace(
        _idx2aa=dict(enumerate([*PMC_RESIDUES, "_"])),
        _peptide_mass=SimpleNamespace(masses=dict(PMC_RESIDUES)),
    )
    validate_pmc_schema(decoder, 40)
    with pytest.raises(ValueError):
        validate_pmc_schema(decoder, 41)
    decoder._idx2aa[0], decoder._idx2aa[1] = decoder._idx2aa[1], decoder._idx2aa[0]
    with pytest.raises(ValueError):
        validate_pmc_schema(decoder, 40)
    with pytest.raises(ValueError, match="shape"):
        mass_con.knapDecode(torch.zeros(1, 39, 28), torch.zeros(1), 0.1)


def make_bundle(path, extra=None):
    payloads = {name: b"fixture component bytes" for name in REQUIRED | {"router.pt"}}
    payloads["MNovo_config.yaml"] = yaml.safe_dump(
        {
            "max_length": 40,
            "learning_rate": 0.1,
            "random_seed": -1,
            "residues": {"A": 71.0},
        }
    ).encode()
    if extra:
        payloads.update(extra)
    manifest = {
        "components": {
            name: {"sha256": hashlib.sha256(data).hexdigest()}
            for name, data in payloads.items()
        }
    }
    torch.save(
        {
            "format": "mnovo-unified-sequence-checkpoint",
            "manifest": manifest,
            "component_payloads": payloads,
        },
        path,
    )


def test_release_rehashes_components_config_and_legacy_marker(tmp_path):
    bundle = tmp_path / "model.ckpt"
    make_bundle(bundle)
    root = resolve_model_release(bundle, tmp_path / "cache")
    assert resolve_model_release(root) == root
    assert "learning_rate" not in (root / "config/inference.yaml").read_text()
    (root / "components/r1_ranker.pt").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        resolve_model_release(root)
    with pytest.raises(RuntimeError, match="hash mismatch"):
        resolve_model_release(bundle, tmp_path / "cache")
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "verified.json").write_text("{}")
    with pytest.raises(ValueError, match="Legacy cache"):
        resolve_model_release(legacy)


@pytest.mark.parametrize(
    "name", ["../escape.pt", "/absolute.pt", "C:\\escape.pt", "a/b.pt"]
)
def test_release_rejects_unsafe_component_names(tmp_path, name):
    bundle = tmp_path / "model.ckpt"
    make_bundle(bundle, {name: b"unsafe"})
    with pytest.raises(ValueError, match="Unsafe component"):
        resolve_model_release(bundle, tmp_path / "cache")
    assert not (tmp_path / "escape.pt").exists()


def test_concurrent_release_extraction_and_config_tamper(tmp_path):
    bundle = tmp_path / "model.ckpt"
    make_bundle(bundle)
    with ThreadPoolExecutor(max_workers=2) as pool:
        roots = list(
            pool.map(
                lambda _: resolve_model_release(bundle, tmp_path / "cache"), range(2)
            )
        )
    assert roots[0] == roots[1]
    (roots[0] / "config/inference.yaml").write_text("max_length: 20\n")
    with pytest.raises(RuntimeError, match="config mismatch"):
        resolve_model_release(roots[0])


def test_mixed_empty_pool_preserves_indices_and_only_ranks_valid_rows():
    runtime = object.__new__(MNovoRuntime)
    runtime.options = RuntimeOptions(candidate_mode="beam-only")
    runtime.device = torch.device("cpu")
    runtime.backbone = SimpleNamespace(
        encoder=lambda x: (torch.zeros(len(x), 1, 2), None),
        decoder=lambda *args: (torch.zeros(3, 2, 3), None, None),
    )
    runtime.decoder5 = SimpleNamespace(decode_all=lambda p: (None, None))
    runtime._rows_from_decode = lambda *a: [
        {"texts": []},
        {"texts": ["AK"]},
        {"texts": []},
    ]
    runtime._tensorize = lambda rows, precursors, memory: {"rows": rows}

    def select(data, spectra):
        assert len(data["rows"]) == 1 and spectra.shape[0] == 1
        return torch.tensor([0]), torch.tensor([0.8]), ["r1"]

    runtime._observable_select = select
    predictions = runtime.predict_batch(
        torch.zeros(3, 1, 2), torch.zeros(3, 3), torch.tensor([7, 2, 9])
    )
    assert [p.dataset_index for p in predictions] == [7, 2, 9]
    assert [p.peptide for p in predictions] == ["", "AK", ""]
    assert [p.status for p in predictions] == [
        "no_valid_candidate",
        "complete",
        "no_valid_candidate",
    ]


def test_actual_training_loss_rejects_repeats_and_preserves_valid_mean(monkeypatch):
    import MNovo.denovo.model as model_module

    monkeypatch.setattr(model_module, "CTCBeamSearchDecoder", lambda *args: None)
    model = model_module.Spec2Pep(
        custom_ctc_loss=False,
        dim_model=16,
        n_head=2,
        dim_feedforward=32,
        n_layers=1,
        max_length=4,
        residues=PMC_RESIDUES,
        max_charge=3,
        n_beams=0,
        PMC_enable=False,
        dropout=0.0,
    )
    monkeypatch.setattr(model, "log", lambda *args, **kwargs: None)
    spectra = torch.tensor([[[150.0, 0.4], [230.0, 0.6]]])
    precursors = torch.tensor([[217.0, 2.0, 109.5]])
    batch = (spectra, precursors, ["AK"])
    _, truth, outputs = model._forward_step(*batch)
    logits = outputs[-1].permute(1, 0, 2)
    old = torch.nn.CTCLoss(blank=model.decoder.get_blank_idx(), zero_infinity=True)(
        logits.log_softmax(-1), truth, torch.tensor([4]), torch.tensor([2])
    )
    assert torch.allclose(model.training_step(batch), old, rtol=1e-5)
    with pytest.raises(ValueError, match="CTC-unalignable"):
        model.training_step((spectra, precursors, ["AAAA"]))


def test_lightning_epoch_aggregation_and_callback_sees_current_metrics():
    import pytorch_lightning as pl
    from MNovo.denovo.model import Spec2Pep

    class Fixture(pl.LightningModule):
        _record_matches = Spec2Pep._record_matches

        def __init__(self):
            super().__init__()
            self.decoder = SimpleNamespace(
                _peptide_mass=SimpleNamespace(masses={"A": 71.037114, "K": 128.094963})
            )
            self.sequence_metrics = torch.nn.ModuleDict(
                {
                    f"valid_{name}": CountRatio()
                    for name in ("aa_precision", "aa_recall", "pep_recall")
                }
            )

        def validation_step(self, batch, batch_idx):
            size = len(batch[0])
            predictions = ["AK" if int(value) == 0 else "" for value in batch[0]]
            self._record_matches("valid", ["AK"] * size, predictions)

    observed = []

    class Capture(pl.Callback):
        def on_validation_epoch_end(self, trainer, model):
            observed.append(float(trainer.callback_metrics["valid/pep_recall"]))

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[Capture()],
    )
    fixture = Fixture()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.arange(3)), batch_size=2
    )
    result = trainer.validate(fixture, dataloaders=loader, verbose=False)[0]
    assert result["valid/pep_recall"] == pytest.approx(1 / 3)
    assert observed == pytest.approx([1 / 3])


def test_audit_modified_key_overlap_and_ctc_repeats(tmp_path):
    from scripts.audit_training_labels import audit

    labels = tmp_path / "labels.tsv"
    labels.write_text(
        "role\tsequence\ntrain\tAM[Oxidation]IL\ntest\tAMLL\ntrain\tAAAA\nvalid\tAXK\n"
    )
    report = audit(labels, 4)
    assert report["roles"]["train"]["unalignable"] == 2
    assert report["roles"]["valid"]["unsupported"] == 1
    assert (
        next(
            row
            for row in report["overlaps"]
            if {row["first"], row["second"]} == {"train", "test"}
        )["canonical_keys"]
        == 1
    )


def test_metadata_recursive_sanitization_preserves_tensor():
    from scripts.audit_checkpoint_metadata import clean_metadata

    tensor = torch.tensor([1.0, 2.0])
    findings = []
    output = clean_metadata(
        {
            "nested": [("prefix /home/user/data/file.pt", tensor)],
            "windows": r"C:\private\x",
        },
        findings,
    )
    assert len(findings) == 2
    assert output["nested"][0][1] is tensor
    assert "/home/" not in output["nested"][0][0]


def test_cli_rejects_subset_eval_before_model_loading(monkeypatch):
    from MNovo import cli
    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pi-mnovo",
            "--model",
            "eval",
            "--input",
            "missing.mgf",
            "--output",
            "unused.tsv",
            "--max-samples",
            "1",
        ],
    )

    def forbidden(*args):
        raise AssertionError("model was loaded before argument validation")

    monkeypatch.setattr(cli, "resolve_model_release", forbidden)
    with pytest.raises(ValueError, match="all input spectra"):
        cli.main()


def test_union_without_cuda_fails_before_reading_model(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="PMC union requires CUDA"):
        MNovoRuntime(tmp_path, RuntimeOptions(), device="cpu")


def test_metric_prediction_count_mismatch_is_not_truncated():
    with pytest.raises(ValueError, match="number of spectra"):
        match_counts(["AK", "AK"], ["AK"], {"A": 71.037114, "K": 128.094963})


def test_fragment_current_runtime_matches_fixed_golden_vector():
    from MNovo.ranker.fragment import fragment_features, PROTON, WATER

    masses = {"A": 71.037114, "K": 128.094963}
    mz = np.array([masses["A"] + PROTON, masses["K"] + WATER + PROTON])
    intensity = np.ones(2)
    golden = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], dtype=np.float32)
    assert np.allclose(fragment_features("AK", mz, intensity, 1, masses), golden)
    runtime = object.__new__(MNovoRuntime)
    runtime.config = {"residues": masses}
    runtime.device = torch.device("cpu")
    data = {
        "mask": torch.tensor([[True]]),
        "precursor": torch.tensor([[217.0, 1.0, 218.0]]),
        "rows": [{"texts": ["AK"]}],
    }
    spectrum = torch.tensor(
        np.stack([mz, intensity], axis=1)[None], dtype=torch.float64
    )
    assert np.allclose(runtime._fragment_tensor(data, spectrum).numpy()[0, 0], golden)
