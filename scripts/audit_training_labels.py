"""Audit a TSV of role, sequence, source, run before training; never filter test."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from MNovo.ctc import required_steps
from MNovo.evaluation import tokens
from MNovo.pmc_schema import PMC_RESIDUES


def audit(path, time_steps=40):
    counts = defaultdict(Counter)
    lengths = defaultdict(Counter)
    repeats = defaultdict(Counter)
    keys = defaultdict(set)
    examples = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not {"role", "sequence"}.issubset(reader.fieldnames or []):
            raise ValueError("TSV requires role and sequence columns.")
        for row_number, row in enumerate(reader, 2):
            role = row["role"].strip()
            if not role:
                raise ValueError(f"Empty data role at row {row_number}.")
            counts[role]["spectra"] += 1
            try:
                labels = tokens(row["sequence"])
                if not labels:
                    raise ValueError("Empty sequence")
                if any(label not in PMC_RESIDUES for label in labels):
                    raise ValueError("Token is outside the released residue vocabulary")
            except ValueError as error:
                counts[role]["unsupported"] += 1
                if len(examples) < 20:
                    examples.append(dict(row=row_number, role=role, error=str(error)))
                continue
            key = "".join(label[0] for label in labels if label[0].isalpha())
            keys[role].add(key)
            needed = required_steps(labels)
            lengths[role][len(key)] += 1
            repeats[role][needed - len(labels)] += 1
            counts[role]["unalignable" if needed > time_steps else "alignable"] += 1
    overlaps = []
    for first, second in itertools.combinations(sorted(counts), 2):
        shared = keys[first] & keys[second]
        overlaps.append(dict(first=first, second=second, canonical_keys=len(shared)))
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return dict(
        input_sha256=digest.hexdigest(),
        time_steps=time_steps,
        roles={role: dict(count) for role, count in counts.items()},
        residue_length_histograms=dict(lengths),
        adjacent_repeat_histograms=dict(repeats),
        unique_keys={role: len(value) for role, value in keys.items()},
        overlaps=overlaps,
        unsupported_examples=examples,
        policy="Audit only. Do not exclude unsupported final-evaluation spectra. "
        "Cross-role sharing needs role-specific interpretation; final evaluation must be isolated.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--time-steps", type=int, default=40)
    args = parser.parse_args()
    if args.time_steps <= 0:
        parser.error("time-steps must be positive")
    result = audit(args.labels, args.time_steps)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
