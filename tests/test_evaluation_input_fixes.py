import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from MNovo import cli
from MNovo.evaluation import evaluate_predictions
from MNovo.input import materialize_mgf_lmdb
from MNovo.input_audit import InputAudit
from MNovo.denovo.spectrum_dataset import SpectrumDataset


def block(title, peaks="200 1\n300 2", charge=2, seq="AK"):
    return f"BEGIN IONS\nTITLE={title}\nCHARGE={charge}+\nPEPMASS=500\nSEQ={seq}\n{peaks}\nEND IONS\n"


def config(root):
    path = root / "config/inference.yaml"
    path.parent.mkdir(parents=True)
    values = dict(max_charge=10, runtime={}, n_peaks=150, min_mz=140,
                  max_mz=2500, min_intensity=.01, remove_precursor_tol=2,
                  residues={"A": 71.037114, "K": 128.094963})
    path.write_text(yaml.safe_dump(values))
    return path, values


def test_all_rejected_eval_replaces_previous_metrics(tmp_path, monkeypatch):
    config(tmp_path)
    source = tmp_path / "input.mgf"
    source.write_text(block("bad", charge=11))
    output = tmp_path / "result.tsv"
    metrics = tmp_path / "custom.json"
    metrics.write_text('{"peptide_recall": 0.95, "previous": true}')
    monkeypatch.setattr(sys, "argv", ["pi-mnovo", "--model", "eval", "--input", str(source),
                        "--model-dir", str(tmp_path), "--output", str(output),
                        "--metrics-output", str(metrics)])
    monkeypatch.setattr(cli, "resolve_model_release", lambda _: tmp_path)
    monkeypatch.setattr(cli, "MNovoRuntime", lambda *a, **k: pytest.fail("loaded model"))
    cli.main()
    result = json.loads(metrics.read_text())
    assert result["status"] == "no_evaluable_spectra"
    assert result["peptide_recall"] is None and result["aa_recall"] is None
    assert result["original_input"] == 1 and result["rejected"] == 1
    assert result["evaluable_spectra"] == 0 and "previous" not in result


def test_evaluation_reports_accepted_denominator_and_failure_replaces_metrics(tmp_path):
    path, values = config(tmp_path)
    source = tmp_path / "input.mgf"
    source.write_text(block("good") + block("bad", charge=11))
    with InputAudit(tmp_path / "prediction.tsv") as audit:
        n = materialize_mgf_lmdb([source], tmp_path / "input.lmdb", 10, True,
                                 audit=audit, preprocessing_config=values)
    prediction = tmp_path / "prediction.tsv"
    prediction.write_text("Sequence\nAK\n")
    output = tmp_path / "metrics.json"
    result = evaluate_predictions(prediction, tmp_path / "input.lmdb", path, output,
        dict(original_input=audit.total, accepted=n, rejected=audit.rejected))
    assert (result["original_input"], result["accepted"], result["rejected"]) == (2, 1, 1)
    assert result["peptide_recall_denominator"] == "accepted_input_spectra"
    assert result["peptide_recall_denominator_count"] == 1
    prediction.write_text("WrongColumn\nAK\n")
    for _ in range(2):  # Failure must close the LMDB for the next evaluation.
        with pytest.raises(ValueError):
            evaluate_predictions(prediction, tmp_path / "input.lmdb", path, output)
        result = json.loads(output.read_text())
        assert result["status"] == "evaluation_failed" and result["peptide_recall"] is None


def test_preprocessing_rejections_are_counted_and_explain_actual_configuration(tmp_path):
    _, values = config(tmp_path)
    source = tmp_path / "input.mgf"
    source.write_text(block("good") + block("zero", "200 0\n300 0")
        + block("negative", "200 -1\n300 -2") + block("outside", "100 1\n110 2")
        + block("precursor", "500 2") + block("after"))
    with InputAudit(tmp_path / "result.tsv") as audit:
        count = materialize_mgf_lmdb([source], tmp_path / "input.lmdb", 10,
                                     audit=audit, preprocessing_config=values)
    result = audit.save("complete_with_rejections", count)
    assert (result["original_input"], count, result["rejected"]) == (6, 2, 4)
    assert result["count_verified"]
    with audit.paths["rejected"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert all(row["field"] == "preprocessing" for row in rows)
    assert "intensity_max=0.0" in rows[0]["observed"]
    assert "140..2500" in rows[2]["supported"]
    assert "usable_peaks=0" in rows[2]["observed"]
    assert "precursor peak removal" in rows[3]["reason"]
    # An explicit model range must override defaults, not merely change the report.
    values["min_mz"] = 600
    with InputAudit(tmp_path / "other.tsv") as audit2:
        assert materialize_mgf_lmdb([source], tmp_path / "other.lmdb", 10,
            audit=audit2, preprocessing_config=values) == 0


def test_valid_preprocessing_is_unchanged():
    dataset = SpectrumDataset([])
    mz, intensity = np.array([200., 300.]), np.array([1., 2.])
    assert torch.equal(dataset._process_peaks(mz, intensity, 500, 2),
                       dataset._process_peaks(mz, intensity, 500, 2, strict=True))


def test_offline_and_batched_validation_use_identical_metrics(tmp_path):
    from MNovo.denovo.metric_counts import CountRatio, match_counts, metric_components
    path, values = config(tmp_path)
    source = tmp_path / "input.mgf"
    source.write_text(block("one") + block("two") + block("three"))
    materialize_mgf_lmdb([source], tmp_path / "input.lmdb", 10, annotated=True,
                         preprocessing_config=values)
    prediction = tmp_path / "pred.tsv"
    prediction.write_text("Sequence\nAK\n\"\"\nnot-a-peptide\n")
    result = evaluate_predictions(prediction, tmp_path / "input.lmdb", path,
                                  tmp_path / "metrics.json")
    aggregate = {name: CountRatio() for name in ("aa_precision", "aa_recall", "pep_recall")}
    for truths, predictions in [(["AK", "AK"], ["AK", ""]), (["AK"], ["not-a-peptide"])]:
        for name, numerator, denominator in metric_components(match_counts(truths, predictions, values["residues"])):
            aggregate[name].update(numerator, denominator)
    assert result["spectra"] == 3 and result["peptide_correct"] == 1
    for name, output in (("aa_precision", "aa_precision"), ("aa_recall", "aa_recall"),
                         ("pep_recall", "peptide_recall")):
        assert result[output] == aggregate[name].compute().item()
    assert result["aa_precision"] == 1.0
    assert result["peptide_recall"] == 1 / 3
