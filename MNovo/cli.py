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
from MNovo.evaluation import evaluate_predictions
from MNovo.release import resolve_model_release
from MNovo.runtime import MNovoRuntime, RuntimeOptions

DEFAULT_RELEASE = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "pi-MNovo-v0.1.0.ckpt"
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
    title = str(
        record.get("title", record.get("pep", f"index={dataset_index}"))
    ).replace("\t", " ").replace("\r", " ").replace("\n", " ")
    charge = int(record["precursor_charge"])
    precursor_mz = float(record["precursor_mz"])
    exp_mh = (
        precursor_mz * charge - (charge - 1) * PROTON_MASS
        if charge > 0
        else None
    )
    calc_mh, modification = sequence_details(peptide, residues)
    mass_shift = (
        exp_mh - calc_mh
        if exp_mh is not None and calc_mh is not None
        else None
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
        help=(
            "Inference YAML. Defaults to "
            "<model-dir>/config/inference.yaml."
        ),
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
    parser.add_argument("--indices")
    parser.add_argument(
        "--metrics-output",
        help="Evaluation JSON path; defaults beside --output.",
    )
    parser.add_argument("--validation-input")
    parser.add_argument("--test-input")
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
) -> None:
    indices = np.load(args.indices) if args.indices else None
    if indices is not None and args.max_samples > 0:
        indices = indices[: args.max_samples]
    elif indices is None and args.max_samples > 0:
        indices = np.arange(
            min(args.max_samples, total_spectra),
            dtype=np.int64,
        )
    progress_total = len(indices) if indices is not None else total_spectra
    options = RuntimeOptions(
        mode=args.mode,
        batch_size=args.batch_size,
        n_workers=args.n_workers,
        ctc_processes=args.ctc_processes,
    )
    runtime = MNovoRuntime(args.model_dir, options, device=args.device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    count = 0
    route_counts = {"r1": 0, "r2_long": 0, "r3_fragment": 0}
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle, LmdbTitleLookup(
        lmdb
    ) as titles:
        handle.write(
            "TITLE\tScan_No\tExp.MH+\tCharge\tSequence\tCalc.MH+\t"
            "Mass_Shift(Exp.-Calc.)\tScore\tModification\n"
        )
        with tqdm(
            total=progress_total,
            desc="MNovo",
            unit="spectra",
            dynamic_ncols=True,
            disable=not sys.stderr.isatty(),
        ) as progress:
            for prediction in runtime.predict_lmdb(lmdb, indices):
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
                                    "pmc_mode": "union",
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
    temporary.replace(output)
    elapsed = max(time.time() - started, 1e-6)
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(output.resolve()),
                "processed": count,
                "seconds": elapsed,
                "spectra_per_second": count / elapsed,
                "candidate_generation": {
                    "beam_width": 5,
                    "pmc_mode": "union",
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


def main() -> None:
    parsed = parse_args()
    if parsed.task != "train":
        parsed.model_dir = str(resolve_model_release(parsed.model_dir))
    config_was_explicit = parsed.config is not None
    args = apply_config(parsed)
    if args.task == "train":
        if args.lmdb:
            raise ValueError("--model train requires MGF input, not --lmdb.")
        if not config_was_explicit:
            raise ValueError("--model train requires an explicit training --config.")
        if not args.validation_input or not args.test_input:
            raise ValueError(
                "--model train requires --validation-input and --test-input."
            )
        command = [
            sys.executable,
            "-m",
            "MNovo.MNovo",
            "--mode",
            "train",
            "--peak_path",
            args.input,
            "--peak_path_val",
            args.validation_input,
            "--peak_path_test",
            args.test_input,
            "--config",
            args.config,
            "--output",
            args.output,
        ]
        if args.checkpoint:
            command.extend(["--model", args.checkpoint])
        subprocess.run(command, check=True)
        return

    if args.lmdb:
        lmdb_path = str(Path(args.lmdb).expanduser())
        context = nullcontext(
            (lmdb_path, [], lmdb_spectra_count(lmdb_path))
        )
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
                lmdb = Path(temporary.name) / "input.lmdb"
                count = materialize_mgf_lmdb(
                    sources,
                    lmdb,
                    max_charge=args.max_charge,
                    annotated=args.task == "eval",
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
        run(args, lmdb, input_files, total_spectra)
        if args.task == "eval":
            metrics_output = args.metrics_output or str(
                Path(args.output).with_suffix(".metrics.json")
            )
            result = evaluate_predictions(
                args.output,
                lmdb,
                args.config,
                metrics_output,
            )
            print(json.dumps({"evaluation": result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
