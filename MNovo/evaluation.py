"""Evaluate fused pi-MNovo predictions with mass-aware sequence metrics."""

from __future__ import annotations

import csv
import json
import re
import os
import tempfile
from pathlib import Path

import numpy as np
import yaml

TOKEN_RE = re.compile(
    r"\+43\.006-17\.027|C\+57\.021|M\+15\.995|N\+0\.984|Q\+0\.984|"
    r"\+42\.011|\+43\.006|-17\.027|[ACDEFGHIKLMNPQRSTVWY]"
)


def tokens(sequence: str | None) -> list[str]:
    if not sequence:
        return []
    value = str(sequence).strip()
    replacements = {
        "[UNIMOD:4]": "+57.021",
        "[Carbamidomethyl]": "+57.021",
        "[UNIMOD:35]": "+15.995",
        "[Oxidation]": "+15.995",
        "[UNIMOD:7]": "+0.984",
        "[Deamidated]": "+0.984",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    value = re.sub(r"([A-Z])\[([+-][0-9.]+)\]", r"\1\2", value)
    value = value.replace("I", "L")
    result = []
    position = 0
    for match in TOKEN_RE.finditer(value):
        if match.start() != position:
            raise ValueError(
                f"Unsupported peptide content at offset {position}: {sequence!r}"
            )
        token = match.group()
        if token[0] in "+-" and result:
            raise ValueError(
                f"N-terminal modification at an internal position: {sequence!r}"
            )
        result.append(token)
        position = match.end()
    if position != len(value):
        raise ValueError(
            f"Unsupported peptide content at offset {position}: {sequence!r}"
        )
    if result and not any(token[0].isalpha() for token in result):
        raise ValueError(f"Peptide contains only a modification: {sequence!r}")
    return result


def write_metrics_state(path, result):
    """Atomically replace metrics, including non-success states for this run."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                     delete=False, suffix=".json.tmp") as handle:
        temporary = Path(handle.name)
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
    try:
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return result


def evaluate_predictions(predictions_path, lmdb_path, config_path, output_path,
                         input_counts=None):
    try:
        result = _evaluate_predictions(predictions_path, lmdb_path, config_path, output_path)
        total = result["spectra"]
        counts = input_counts or {"original_input": None, "accepted": total, "rejected": None}
        if counts["accepted"] != total:
            raise ValueError("Evaluation count does not match accepted input count")
        if input_counts is not None and (
            counts["rejected"] < 0 or counts["original_input"] != total + counts["rejected"]
        ):
            raise ValueError("Original input must equal accepted plus rejected")
        result.update(counts)
        result["evaluable_spectra"] = total
        result["peptide_recall_denominator"] = (
            "accepted_input_spectra" if input_counts is not None else "provided_lmdb_spectra")
        return write_metrics_state(output_path, result)
    except Exception as error:
        write_metrics_state(output_path, dict(status="evaluation_failed",
                            error=str(error), peptide_recall=None,
                            aa_precision=None, aa_recall=None))
        raise


def _evaluate_predictions(
    predictions_path: str | Path,
    lmdb_path: str | Path,
    config_path: str | Path,
    output_path: str | Path,
) -> dict:
    from MNovo.denovo.spectrum_dataset import SpectrumDataset
    from MNovo.denovo.spectrum_index import LmdbSpectrumIndex
    from MNovo.denovo.metric_counts import match_counts, metric_components, count_ratio

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    valid_charge = np.arange(1, int(config["max_charge"]) + 1)
    index = LmdbSpectrumIndex(
        str(lmdb_path),
        None,
        2,
        valid_charge,
        True,
        lock=False,
    )
    try:
        dataset = SpectrumDataset(
            [index],
            n_peaks=int(config["n_peaks"]),
            min_mz=float(config["min_mz"]),
            max_mz=float(config["max_mz"]),
            min_intensity=float(config["min_intensity"]),
            remove_precursor_tol=float(config["remove_precursor_tol"]),
            random_state=3407,
        )
        predictions: list[str] = []
        with Path(predictions_path).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                sequence = row.get("Sequence", row.get("peptide"))
                if sequence is None:
                    raise ValueError("Prediction TSV must contain a Sequence column.")
                predictions.append(sequence)
        total_spectra = len(dataset)
        if len(predictions) != total_spectra:
            raise RuntimeError(
                f"Prediction count mismatch: {len(predictions)} vs {total_spectra}. "
                "Peptide recall must use every input spectrum as its denominator."
            )

        truths = [index[position][-1] for position in range(total_spectra)]
        counts = match_counts(truths, predictions, config["residues"])
        n_aa_correct, n_aa_true, n_aa_pred, peptide_correct, total_spectra = counts
        metrics = {name: count_ratio(numerator, denominator)
                   for name, numerator, denominator in metric_components(counts)}
        aa_precision = metrics["aa_precision"]
        aa_recall = metrics["aa_recall"]
        peptide_recall = metrics["pep_recall"]
        result = {
            "status": "complete" if total_spectra else "no_evaluable_spectra",
            "spectra": total_spectra,
            "peptide_recall_denominator_count": total_spectra,
            "n_aa_true": int(n_aa_true),
            "n_aa_pred": int(n_aa_pred),
            "n_aa_correct": int(n_aa_correct),
            "peptide_correct": peptide_correct,
            "aa_precision": float(aa_precision) if total_spectra else None,
            "aa_recall": float(aa_recall) if total_spectra else None,
            "peptide_recall": float(peptide_recall) if total_spectra else None,
            "aa_match_mode": "best",
            "metric_implementation": (
                "MNovo.denovo.metric_counts.match_counts + metric_components + count_ratio"
            ),
        }
        return result
    finally:
        index.env.close()
