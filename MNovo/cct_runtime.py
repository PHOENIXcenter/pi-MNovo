# Export matched Target and Counterpart rows with validity and reference status.
# 输出配对的 Target 与 Counterpart，并保留有效性及参考标签状态。
"""Paired CCT feature export without changing sequence selection."""

from __future__ import annotations

import csv
from pathlib import Path

from MNovo import cct_reference_features as reference
from MNovo.cct_adapter import FEATURE_COLUMNS
from MNovo.cct_policy_adapter import paired_reference_features_with_policy


class CCTFeatureExporter:
    def __init__(self, decoder, residues, output=None):
        self.blank = int(decoder.get_blank_idx())
        self.symbols = decoder._idx2aa
        self.residues = residues
        if not decoder.reverse:
            raise ValueError("Reference features require C-to-N decoder rows")
        self.rows = []
        self.output = output
        self.complete_pairs = 0
        self.invalid_pairs = 0
        self.labelled_pairs = 0
        self._handle = None
        self._writer = None
        if output is not None:
            self._handle = Path(output).open("x", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(
                self._handle,
                fieldnames=("pair_id", "sequence", "reference_sequence", "denovo_label",
                            "is_decoy", "valid", "error", "native_route", "native_status",
                            "spearman_degenerate", "nonfinite_replaced", "feature_policy")
                + tuple("f_" + column for column in FEATURE_COLUMNS),
            )
            self._writer.writeheader()

    def close(self):
        if self._handle is not None:
            self._handle.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _flush(self):
        if len(self.rows) % 2:
            raise RuntimeError("CCT exporter lost a paired view")
        self.complete_pairs += sum(row["valid"] and row["is_decoy"] == 0 for row in self.rows)
        self.invalid_pairs += sum(not row["valid"] and row["is_decoy"] == 0 for row in self.rows)
        self.labelled_pairs += sum(row["denovo_label"] in (0, 1) and row["is_decoy"] == 0 for row in self.rows)
        if self._writer is not None:
            self._writer.writerows(self.rows)
            self.rows.clear()

    def _labels(self, predictions, truth):
        if truth is None:
            return [None] * len(predictions)
        return [None if not isinstance(sequence, str) or not sequence.strip() else
                int(prediction.peptide.strip().replace("I", "L") == sequence.strip().replace("I", "L"))
                for prediction, sequence in zip(predictions, truth)]

    def _append(self, prediction, label, sequence, view, features=None, error="", audit=None):
        row = {
            "pair_id": int(prediction.dataset_index),
            "sequence": prediction.peptide,
            "reference_sequence": sequence,
            "denovo_label": label,
            "is_decoy": view,
            "valid": features is not None,
            "error": error,
            "native_route": prediction.route,
            "native_status": prediction.status,
        }
        audit = audit or {}
        row.update(spearman_degenerate=audit.get("spearman_degenerate", False),
                   nonfinite_replaced=";".join(audit.get("nonfinite_replaced", [])),
                   feature_policy="constant_spearman_zero_v1")
        if features is not None:
            row.update({"f_" + key: value for key, value in features.items()})
        self.rows.append(row)

    def collect_empty(self, predictions, truth):
        references = [None] * len(predictions) if truth is None else truth
        for prediction, label, sequence in zip(
            predictions, self._labels(predictions, truth), references
        ):
            for view in (0, 1):
                self._append(prediction, label, sequence, view, error=prediction.status)
        self._flush()

    def collect(
        self, predictions, candidates, valid, selected, attention,
        log_probs, spectra, precursors, truth,
    ):
        labels = self._labels(predictions, truth)
        references = [None] * len(predictions) if truth is None else truth
        selected_tokens = {
            row_index: candidates[row_index]["tokens"][choice]
            for row_index, choice in zip(valid, selected)
        }
        for index, prediction in enumerate(predictions):
            try:
                if index not in selected_tokens:
                    raise ValueError(prediction.status)
                (target, counterpart), flags = paired_reference_features_with_policy(
                    attention[index], log_probs[index], selected_tokens[index],
                    prediction.peptide, spectra[index], precursors[index],
                    blank=self.blank, symbols=self.symbols,
                    residues=self.residues, reference=reference,
                )
            except (ValueError, KeyError, IndexError, ZeroDivisionError) as error:
                for view in (0, 1):
                    self._append(prediction, labels[index], references[index], view, error=str(error))
            else:
                for view, features in enumerate((target, counterpart)):
                    self._append(prediction, labels[index], references[index], view, features, audit=flags[view])
        self._flush()


