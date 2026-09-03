"""Evaluate fused pi-MNovo predictions with mass-aware sequence metrics."""

from __future__ import annotations

import csv
import json
import re
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
    return TOKEN_RE.findall(value)


def evaluate_predictions(
    predictions_path: str | Path,
    lmdb_path: str | Path,
    config_path: str | Path,
    output_path: str | Path,
) -> dict:
    from MNovo.denovo.db_dataset import DbDataset
    from MNovo.denovo.db_index import DB_Index
    from MNovo.denovo.evaluate import aa_match_batch, aa_match_metrics

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    valid_charge = np.arange(1, int(config["max_charge"]) + 1)
    index = DB_Index(
        str(lmdb_path),
        None,
        2,
        valid_charge,
        True,
        lock=False,
    )
    dataset = DbDataset(
        [index],
        n_peaks=int(config["n_peaks"]),
        min_mz=float(config["min_mz"]),
        max_mz=float(config["max_mz"]),
        min_intensity=float(config["min_intensity"]),
        remove_precursor_tol=float(config["remove_precursor_tol"]),
        random_state=3407,
    )
    predictions: list[list[str]] = []
    with Path(predictions_path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            sequence = row.get("Sequence", row.get("peptide"))
            if sequence is None:
                raise ValueError(
                    "Prediction TSV must contain a Sequence column."
                )
            predictions.append(tokens(sequence))
    total_spectra = len(dataset)
    if len(predictions) != total_spectra:
        raise RuntimeError(
            f"Prediction count mismatch: {len(predictions)} vs {total_spectra}. "
            "Peptide recall must use every input spectrum as its denominator."
        )

    truths = []
    ordered_predictions = []
    for position in range(total_spectra):
        _spectrum, _mz, _charge, truth = dataset[position]
        truth_tokens = tokens(truth)
        if not truth_tokens:
            raise ValueError(
                "Evaluation requires an annotated MGF with a non-empty SEQ= "
                f"value for every spectrum; missing at index {position}."
            )
        truths.append(truth_tokens)
        ordered_predictions.append(predictions[position])
    index.env.close()

    matches, n_aa_true, n_aa_pred = aa_match_batch(
        truths,
        ordered_predictions,
        config["residues"],
        mode="best",
    )
    aa_precision, aa_recall, _ = aa_match_metrics(
        matches,
        n_aa_true,
        n_aa_pred,
    )
    peptide_correct = int(sum(bool(row[1]) for row in matches))
    peptide_recall = peptide_correct / max(total_spectra, 1)
    result = {
        "spectra": total_spectra,
        "peptide_recall_denominator_count": total_spectra,
        "n_aa_true": int(n_aa_true),
        "n_aa_pred": int(n_aa_pred),
        "n_aa_correct": int(sum(row[0].sum() for row in matches)),
        "peptide_correct": peptide_correct,
        "aa_precision": float(aa_precision),
        "aa_recall": float(aa_recall),
        "peptide_recall": float(peptide_recall),
        "peptide_recall_denominator": "all_input_spectra",
        "aa_match_mode": "best",
        "metric_implementation": (
            "MNovo.denovo.evaluate.aa_match_batch + aa_match_metrics"
        ),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
