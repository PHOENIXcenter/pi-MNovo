#!/usr/bin/env python3
"""Command-line interface for pi-MNovo."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from MNovo.input import (
    LmdbTitleLookup,
    lmdb_spectra_count,
    materialize_mgf_lmdb,
    resolve_mgf_inputs,
)
from MNovo.evaluation import evaluate_predictions, write_metrics_state
from MNovo.release import resolve_model_release
from MNovo.runtime import MNovoRuntime, RuntimeOptions
from MNovo.selection import validate_selection
from MNovo.input_audit import InputAudit

DEFAULT_RELEASE = Path("models/pi-MNovo-v0.1.0.ckpt")
OUTPUT_HEADER = (
    "TITLE\tScan_No\tExp.MH+\tCharge\tSequence\tCalc.MH+\t"
    "Mass_Shift(Exp.-Calc.)\tScore\tModification\tStatus\tRoute\tdataset_index\n"
)

PROTON_MASS = 1.007276466621
WATER_MASS = 18.0105646837
SCAN_PATTERNS = (
    re.compile(r"\bscan(?:_no)?s?\s*[=:]\s*(\d+)", re.I),
    re.compile(r"\bindex\s*[=:]\s*(\d+)", re.I),
    re.compile(r"\.(\d+)\.\d+\.\d+(?:\D|$)"),
)
MODIFICATION_NAMES = {
    "C+57.021": "Carbamidomethyl[C]",
    "M+15.995": "Oxidation[M]",
    "N+0.984": "Deamidated[N]",
    "Q+0.984": "Deamidated[Q]",
    "+42.011": "Acetyl[AnyN-term]",
    "+43.006": "Carbamyl[AnyN-term]",
    "-17.027": "Ammonia-loss[AnyN-term]",
    "+43.006-17.027": "Carbamyl+Ammonia-loss[AnyN-term]",
}


def scan_number(title: str, fallback_index: int) -> int:
    """Extract a scan number from common MGF TITLE conventions."""
    for pattern in SCAN_PATTERNS:
        match = pattern.search(title)
        if match:
            return int(match.group(1))
    return int(fallback_index) + 1


def sequence_details(
    sequence: str,
    residues: dict[str, float],
) -> tuple[float | None, str]:
    """Calculate peptide MH+ and summarize encoded modifications."""
    if not sequence:
        return None, ""
    residue_names = sorted(residues, key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(name) for name in residue_names))
    matches = list(pattern.finditer(sequence))
    if not matches:
        return None, ""
    calc_mh = (
        sum(float(residues[match.group(0)]) for match in matches)
        + WATER_MASS
        + PROTON_MASS
    )
    modifications = []
    position = 0
    for match in matches:
        token = match.group(0)
        if len(token) == 1 and token.isalpha():
            position += 1
        elif token[0].isalpha():
            position += 1
            name = MODIFICATION_NAMES.get(token, token)
            modifications.append(f"{position},{name};")
        else:
            name = MODIFICATION_NAMES.get(token, token)
            modifications.append(f"0,{name};")
    return calc_mh, "".join(modifications)


def output_row(
    record: dict,
    dataset_index: int,
    peptide: str,
    confidence: float,
    residues: dict[str, float],
) -> list[str]:
    """Build one publication-facing TSV row without extra model work."""
    title = (
        str(record.get("title", record.get("pep", f"index={dataset_index}")))
        .replace("\t", " ")
        .replace("\r", " ")
        .replace("\n", " ")
    )
    charge = int(record["precursor_charge"])
    precursor_mz = float(record["precursor_mz"])
    exp_mh = precursor_mz * charge - (charge - 1) * PROTON_MASS if charge > 0 else None
    calc_mh, modification = sequence_details(peptide, residues)
    mass_shift = (
        exp_mh - calc_mh if exp_mh is not None and calc_mh is not None else None
    )
    number = lambda value: "" if value is None else f"{value:.6f}"
    return [
        title,
        str(scan_number(title, dataset_index)),
        number(exp_mh),
        str(charge),
        peptide,
        number(calc_mh),
        number(mass_shift),
        f"{confidence:.4f}",
        modification,
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "--task",
        dest="task",
        choices=("denovo", "eval", "train"),
        default="denovo",
        help="Task to run: denovo prediction, annotated evaluation, or training.",
    )
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_RELEASE),
        help="Unified MNovo .ckpt package or an extracted model directory.",
    )
    parser.add_argument(
        "--config",
        help=("Inference YAML. Defaults to <model-dir>/config/inference.yaml."),
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--input",
        help=(
            "Single MGF file, directory, or glob such as '*.mgf' or "
            "'**/*.mgf'; the .mgf extension is case-insensitive."
        ),
    )
    inputs.add_argument("--lmdb", help="Existing preprocessed LMDB input.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("fast",))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--n-workers", type=int)
    parser.add_argument("--ctc-processes", type=int)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help=(
            "Backbone device. The default selects CUDA; PMC still requires "
            "a CUDA-capable system when --device cpu is used."
        ),
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--indices", help="Unique 1-D integer .npy indices; unavailable for eval."
    )
    parser.add_argument(
        "--candidate-mode",
        choices=("union", "beam-only"),
        default="union",
        help="beam-only is diagnostic and changes the candidate algorithm.",
    )
    parser.add_argument(
        "--metrics-output",
        help="Evaluation JSON path; defaults beside --output.",
    )
    parser.add_argument("--validation-input")
    parser.add_argument("--test-input", help="Deprecated: ignored during training.")
    parser.add_argument(
        "--checkpoint",
        help="Optional .ckpt initialization/resume weight for --model train.",
    )
    parser.add_argument(
        "--temp-dir",
        help="Optional parent directory for the automatically created MGF LMDB.",
    )
    return parser.parse_args()


def apply_config(args: argparse.Namespace) -> argparse.Namespace:
    model_dir = Path(args.model_dir).expanduser()
    config_path = (
        Path(args.config).expanduser()
        if args.config
        else model_dir / "config" / "inference.yaml"
    )
    if not config_path.is_file():
        raise FileNotFoundError(f"Inference config not found: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if args.config and getattr(args, "task", "denovo") != "train":
        frozen_path = model_dir / "config" / "inference.yaml"
        frozen = yaml.safe_load(frozen_path.read_text(encoding="utf-8"))
        changed = [
            key
            for key, value in config.items()
            if key != "runtime" and (key not in frozen or value != frozen[key])
        ]
        if changed:
            raise ValueError(
                f"Cannot override frozen inference fields: {sorted(changed)}"
            )
        # Every stage must use the same preprocessing and mass schema. Only
        # operational controls may be overridden by an external inference YAML.
        overrides = config.get("runtime", {})
        if not isinstance(overrides, dict):
            raise TypeError(f"'runtime' must be a mapping in {config_path}")
        config = dict(frozen, runtime={**frozen.get("runtime", {}), **overrides})
        config_path = frozen_path
    runtime = config.get("runtime", {})
    if not isinstance(runtime, dict):
        raise TypeError(f"'runtime' must be a mapping in {config_path}")

    args.mode = args.mode or runtime.get("mode", "fast")
    args.batch_size = (
        args.batch_size
        if args.batch_size is not None
        else int(runtime.get("batch_size", 128))
    )
    args.n_workers = (
        args.n_workers
        if args.n_workers is not None
        else int(runtime.get("n_workers", 16))
    )
    args.ctc_processes = (
        args.ctc_processes
        if args.ctc_processes is not None
        else int(runtime.get("ctc_processes", 16))
    )
    if args.temp_dir is None:
        args.temp_dir = runtime.get("temp_dir")
    args.max_charge = int(config.get("max_charge", 10))
    args.config = str(config_path.resolve())
    return args


def run(
    args: argparse.Namespace,
    lmdb: str,
    input_files: list[Path],
    total_spectra: int,
) -> int:
    indices = validate_selection(
        np.load(args.indices, allow_pickle=False) if args.indices else None,
        total_spectra,
        args.max_samples,
        args.task,
    )
    progress_total = len(indices) if indices is not None else total_spectra
    if progress_total == 0:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(OUTPUT_HEADER, encoding="utf-8")
        temporary.replace(output)
        print(json.dumps({"status": "no_accepted_spectra", "processed": 0}))
        return 0
    options = RuntimeOptions(
        mode=args.mode,
        candidate_mode=args.candidate_mode,
        batch_size=args.batch_size,
        n_workers=args.n_workers,
        ctc_processes=args.ctc_processes,
    )
    runtime = MNovoRuntime(args.model_dir, options, device=args.device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    count = 0
    route_counts = {"r1": 0, "r2_long": 0, "r3_fragment": 0, "none": 0}
    temporary = output.with_suffix(output.suffix + ".tmp")
    with (
        temporary.open("w", encoding="utf-8") as handle,
        LmdbTitleLookup(lmdb) as titles,
    ):
        handle.write(OUTPUT_HEADER)
        with tqdm(
            total=progress_total,
            desc="MNovo",
            unit="spectra",
            dynamic_ncols=True,
            disable=not sys.stderr.isatty(),
        ) as progress:
            for prediction in runtime.predict_lmdb(lmdb, indices):
                if count >= progress_total:
                    raise RuntimeError(
                        "Prediction count mismatch: too many output rows."
                    )
                expected_index = int(indices[count]) if indices is not None else count
                if prediction.dataset_index != expected_index:
                    raise RuntimeError(
                        f"Prediction index mismatch at row {count}: "
                        f"{prediction.dataset_index} vs {expected_index}."
                    )
                count += 1
                route_counts[prediction.route] += 1
                record = titles.spectrum(prediction.dataset_index)
                handle.write(
                    "\t".join(
                        output_row(
                            record,
                            prediction.dataset_index,
                            prediction.peptide,
                            prediction.confidence,
                            runtime.config["residues"],
                        )
                        + [
                            prediction.status,
                            prediction.route,
                            str(prediction.dataset_index),
                        ]
                    )
                    + "\n"
                )
                progress.update(1)
                if count % 5000 == 0:
                    elapsed = max(time.time() - started, 1e-6)
                    print(
                        json.dumps(
                            {
                                "processed": count,
                                "total": progress_total,
                                "progress_fraction": count / progress_total,
                                "spectra_per_second": count / elapsed,
                                "candidate_generation": {
                                    "beam_width": 5,
                                    "pmc_mode": "union"
                                    if args.candidate_mode == "union"
                                    else "off",
                                    "statistics": dict(runtime.candidate_statistics),
                                    "dynamic_expansion": False,
                                },
                                "route_fraction": {
                                    name: value / count
                                    for name, value in route_counts.items()
                                },
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
    if count != progress_total:
        raise RuntimeError(f"Prediction count mismatch: {count} vs {progress_total}.")
    temporary.replace(output)
    elapsed = max(time.time() - started, 1e-6)
    print(
        json.dumps(
            {
                "status": "complete_with_no_valid_candidate"
                if route_counts["none"]
                else "complete",
                "candidate_mode": args.candidate_mode,
                "output": str(output.resolve()),
                "processed": count,
                "seconds": elapsed,
                "spectra_per_second": count / elapsed,
                "candidate_generation": {
                    "beam_width": 5,
                    "pmc_mode": "union" if args.candidate_mode == "union" else "off",
                    "statistics": dict(runtime.candidate_statistics),
                    "dynamic_expansion": False,
                },
                "route_counts": route_counts,
                "input_type": "mgf" if input_files else "lmdb",
                "input_files": len(input_files) if input_files else 1,
                "config": args.config,
                "runtime": {
                    "mode": args.mode,
                    "batch_size": args.batch_size,
                    "n_workers": args.n_workers,
                    "ctc_processes": args.ctc_processes,
                    "device": str(runtime.device),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


    return count


def main() -> None:
    parsed = parse_args()
    metrics_output = parsed.metrics_output or str(Path(parsed.output).with_suffix(".metrics.json"))
    if parsed.max_samples < 0:
        raise ValueError("--max-samples must be non-negative.")
    if parsed.task == "eval" and (parsed.indices or parsed.max_samples):
        raise ValueError(
            "eval requires all input spectra; subset parameters are unsupported."
        )
    if parsed.task == "eval":
        if Path(metrics_output).resolve() == Path(parsed.output).resolve():
            raise ValueError("Metrics and prediction outputs must be different files.")
        write_metrics_state(metrics_output, dict(status="in_progress", peptide_recall=None,
                                                 aa_precision=None, aa_recall=None))
    if parsed.task != "train":
        parsed.model_dir = str(resolve_model_release(parsed.model_dir))
    config_was_explicit = parsed.config is not None
    args = apply_config(parsed)
    if args.task == "train":
        if args.lmdb:
            raise ValueError("--model train requires MGF input, not --lmdb.")
        if not config_was_explicit:
            raise ValueError("--model train requires an explicit training --config.")
        if not args.validation_input:
            raise ValueError("--model train requires --validation-input.")
        command = [
            sys.executable,
            "-m",
            "MNovo.backbone_cli",
            "--mode",
            "train",
            "--peak_path",
            args.input,
            "--peak_path_val",
            args.validation_input,
            "--config",
            args.config,
            "--output",
            args.output,
        ]
        if args.checkpoint:
            command.extend(["--model", args.checkpoint])
        subprocess.run(command, check=True)
        return

    audit = None
    if args.lmdb:
        lmdb_path = str(Path(args.lmdb).expanduser())
        context = nullcontext((lmdb_path, [], lmdb_spectra_count(lmdb_path)))
    else:
        sources = resolve_mgf_inputs(args.input)
        print(
            json.dumps(
                {
                    "status": "materializing_mgf",
                    "input_files": len(sources),
                    "first_file": str(sources[0]),
                    "last_file": str(sources[-1]),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        temporary = tempfile.TemporaryDirectory(
            prefix="mnovo_mgf_",
            dir=args.temp_dir,
        )

        class MgfContext:
            def __enter__(self):
                nonlocal audit
                lmdb = Path(temporary.name) / "input.lmdb"
                audit = InputAudit(args.output)
                with audit:
                    count = materialize_mgf_lmdb(
                        sources,
                        lmdb,
                        max_charge=args.max_charge,
                        annotated=args.task == "eval",
                        audit=audit,
                        preprocessing_config=yaml.safe_load(Path(args.config).read_text(encoding="utf-8")),
                    )
                print(
                    json.dumps(
                        {
                            "status": "mgf_ready",
                            "input_files": len(sources),
                            "spectra": count,
                            "temporary_lmdb": str(lmdb),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                return str(lmdb), sources, count

            def __exit__(self, exc_type, exc_value, traceback):
                temporary.cleanup()

        context = MgfContext()
    with context as (lmdb, input_files, total_spectra):
        try:
            predicted = run(args, lmdb, input_files, total_spectra)
        except Exception:
            if audit:
                audit.save("prediction_failed")
            if args.task == "eval":
                write_metrics_state(metrics_output, dict(status="prediction_failed", peptide_recall=None,
                                                         aa_precision=None, aa_recall=None))
            raise
        if audit:
            not_selected = total_spectra - predicted
            status = "complete_with_rejections" if audit.rejected else "complete"
            if not predicted:
                status = "no_predictions"
            result = audit.save(status, predicted, not_selected)
            print(json.dumps({"input_accounting": result}, ensure_ascii=False, indent=2))
        if args.task == "eval":
            result = evaluate_predictions(
                args.output,
                lmdb,
                args.config,
                metrics_output,
                input_counts=dict(original_input=audit.total, accepted=audit.accepted,
                                  rejected=audit.rejected) if audit else None,
            )
            print(json.dumps({"evaluation": result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
