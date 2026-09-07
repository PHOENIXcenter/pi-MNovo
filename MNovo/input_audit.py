"""Account for every MGF block, including inputs that cannot be predicted."""

import csv
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
from pyteomics.mgf import MGF


FIELDS = ["source_file", "spectrum_number", "TITLE", "status", "dataset_index",
          "field", "supported", "observed", "reason"]


def blocks(path):
    """Yield one-based source positions; malformed blocks remain countable."""
    header, block = [], None
    number = 0
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line in handle:
            marker = line.strip().upper()
            if marker == "BEGIN IONS":
                if block is not None:
                    yield number, header, block, False
                number += 1
                block = [line]
            elif block is not None:
                block.append(line)
                if marker == "END IONS":
                    yield number, header, block, True
                    block = None
            elif number == 0:
                header.append(line)
    if block is not None:
        yield number, header, block, False


class InputAudit:
    def __init__(self, output):
        self.output = Path(output)
        self.total = self.accepted = self.rejected = 0
        self.reasons = Counter()
        self.paths = {name: self.output.with_suffix(suffix) for name, suffix in (
            ("input_audit", ".input-audit.tsv"), ("rejected", ".rejected.tsv"),
            ("summary", ".input-summary.json"))}

    def __enter__(self):
        if self.output.resolve() in {path.resolve() for path in self.paths.values()}:
            raise ValueError("Prediction output must not use an input-audit sidecar filename.")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.handles = []
        self.writers = []
        for name in ("input_audit", "rejected"):
            handle = self.paths[name].open("w", encoding="utf-8", newline="")
            self.handles.append(handle)
            writer = csv.DictWriter(handle, FIELDS, delimiter="\t")
            writer.writeheader()
            self.writers.append(writer)
        # Replace any previous successful count claim before starting this run.
        self.save("parsing")
        return self

    def record(self, row):
        self.total += 1
        self.writers[0].writerow(row)
        if row["status"] == "rejected":
            self.rejected += 1
            self.reasons[row["field"]] += 1
            self.writers[1].writerow(row)
        else:
            self.accepted += 1

    def save(self, status, predictions=None, not_selected=0):
        result = dict(status=status, original_input=self.total,
                      accepted=self.accepted, rejected=self.rejected,
                      predictions=predictions, not_selected=not_selected,
                      rejected_by_field=dict(self.reasons),
                      files={key: str(path.resolve()) for key, path in self.paths.items()})
        if predictions is not None:
            result["count_verified"] = (
                self.total == predictions + self.rejected + not_selected
                and self.accepted == predictions + not_selected)
            if not result["count_verified"]:
                raise RuntimeError(f"Input accounting mismatch: {result}")
            result["equation"] = (
                "original_input = predictions + rejected" if not not_selected else
                "original_input = predictions + rejected + not_selected")
        self.paths["summary"].write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    def __exit__(self, typ, value, traceback):
        for handle in self.handles:
            handle.close()
        self.save("parse_failed" if typ else "parsed")


def spectra(path, max_charge, audit=None, start_index=0):
    """Validate independently so one bad MGF block never hides its neighbours."""
    accepted = start_index
    for number, header, block, complete in blocks(path):
        params = {}
        for line in header + block:
            if "=" in line:
                key, value = line.split("=", 1)
                params[key.strip().upper()] = value.strip()
        row = dict(source_file=str(Path(path).resolve()), spectrum_number=number,
                   TITLE=params.get("TITLE", f"index={number - 1}"),
                   status="rejected", dataset_index="", field="", supported="",
                   observed="", reason="")

        def reject(field, supported, observed, detail):
            row.update(field=field, supported=supported, observed=observed,
                       reason=f"{detail}; supported: {supported}; input: {observed}")

        spectrum = None
        if not complete:
            reject("MGF", "BEGIN IONS ... END IONS", "missing END IONS", "Incomplete spectrum")
        elif "CHARGE" not in params:
            reject("CHARGE", f"one integer in 1..{max_charge}", "missing", "Missing CHARGE")
        elif "PEPMASS" not in params:
            reject("PEPMASS", "finite precursor m/z > 0", "missing", "Missing PEPMASS")
        else:
            try:
                with MGF(io.StringIO("".join(header + block))) as reader:
                    spectrum = next(iter(reader))
                charges = spectrum["params"].get("charge", [])
                if len(charges) != 1 or not 1 <= int(charges[0]) <= max_charge:
                    reject("CHARGE", f"one integer in 1..{max_charge}", params["CHARGE"],
                           "Unsupported or ambiguous precursor charge")
                else:
                    mz = float(spectrum["params"]["pepmass"][0])
                    if not np.isfinite(mz) or mz <= 0:
                        reject("PEPMASS", "finite precursor m/z > 0", params["PEPMASS"],
                               "Invalid precursor mass")
                    elif (not len(spectrum["m/z array"])
                          or not np.isfinite(spectrum["m/z array"]).all()
                          or not np.isfinite(spectrum["intensity array"]).all()):
                        reject("peaks", "nonempty finite m/z and intensity arrays", "empty or nonfinite peaks",
                               "Invalid peak data")
            except Exception as error:
                # Only parsing this block is isolated; I/O and model failures propagate.
                reject("MGF", f"parseable peaks, PEPMASS > 0, one CHARGE in 1..{max_charge}",
                       f"CHARGE={params.get('CHARGE')}; PEPMASS={params.get('PEPMASS')}",
                       f"{type(error).__name__}: {error}")
        if not row["reason"]:
            row.update(status="accepted", dataset_index=accepted)
            spectrum["params"]["title"] = row["TITLE"]
            accepted += 1
        if audit:
            audit.record(row)
        if row["status"] == "accepted":
            yield spectrum
