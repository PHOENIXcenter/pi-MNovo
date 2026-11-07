#!/usr/bin/env python3
# Keep input accounting, model output and optional quality control separate.
# 输入审计、序列预测与可选质控分别产出结果。
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

from MNovo.data.input import (
    LmdbTitleLookup,
    lmdb_spectra_count,
    materialize_mgf_lmdb,
    resolve_mgf_inputs,
)
from MNovo.evaluation import check_output_paths, evaluate_predictions, write_metrics_state
from MNovo.release import resolve_model_release
from MNovo.inference.runtime import MNovoRuntime, RuntimeOptions
from MNovo.inference.selection import validate_selection
from MNovo.inference.config import EDITABLE, resolve_settings
from MNovo.data.input_audit import InputAudit

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
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", dest="task", required=True, choices=("denovo", "eval", "train"))
    parser.add_argument("--config", required=True, help="Path to the task YAML.")
    parser.add_argument("--input", required=True, help="MGF file, directory, glob or LMDB.")
    parser.add_argument("--output", help="Output TSV (training: run directory). Defaults to a new run under the code root's outputs/.")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        parser.error("config must be a YAML mapping")
    run = config.get("runtime", {})
    defaults = dict(model_dir=None, input_format="mgf", mode=None, batch_size=None,
                    n_workers=None, ctc_processes=None, device="auto", max_samples=0,
                    indices=None, candidate_mode=None, fdr_mode="off", gpus=None,
                    cct_profile=None, cct_scope_id=None, cct_features_output=None, cct_fit=None,
                    psm_fdr=.05, peptide_fdr=.05, metrics_output=None,
                    validation_input=None, checkpoint=None, temp_dir=None)
    if not isinstance(run, dict) or run.keys() - defaults.keys():
        parser.error("invalid runtime mapping or unknown runtime key")
    for key, value in {**defaults, **run}.items():
        setattr(args, key, value)
    if args.task != "train" and not args.model_dir:
        parser.error("set runtime.model_dir to the downloaded checkpoint")
    if args.fdr_mode not in ("on", "off"):
        parser.error('runtime.fdr_mode must be the quoted string "on" or "off"')
    if args.fdr_mode == "on":
        if args.task == "train":
            parser.error("FDR applies to denovo/eval, not backbone training")
        from MNovo.cct.workflow import fit_settings
        args.cct_fit = fit_settings(args.cct_fit)
        if args.cct_profile and not Path(args.cct_profile).is_file():
            parser.error(f"Profile not found: {args.cct_profile}. Correct the path or set cct_profile: null to fit.")
        if not args.cct_scope_id:
            parser.error("Set runtime.cct_scope_id to your study identifier (for an existing profile use its calibrated scope).")
        if args.input_format == "lmdb" and args.cct_fit["reference_table"]:
            parser.error("An external reference table requires MGF input; for LMDB use embedded reference_sequence.")
    if args.input_format not in ("lmdb", "mgf"):
        parser.error("runtime.input_format must be mgf or lmdb")
    if args.device not in ("auto", "cpu", "cuda") or args.candidate_mode not in (None, "union", "beam-only"):
        parser.error("invalid runtime.device or runtime.candidate_mode")
    for key, minimum in (("batch_size", 1), ("n_workers", 0), ("ctc_processes", 1), ("max_samples", 0)):
        value = getattr(args, key)
        if value is not None and (type(value) is not int or value < minimum):
            parser.error(f"runtime.{key} must be an integer >= {minimum}")
    for key in ("psm_fdr", "peptide_fdr"):
        value = getattr(args, key)
        if type(value) not in (int, float) or not 0 < value <= 1:
            parser.error(f"runtime.{key} must be in (0, 1]")
    if args.gpus is not None:
        args.gpus = str(args.gpus)
    args.lmdb = args.input if args.input_format == "lmdb" else None
    if args.lmdb:
        args.input = None
    if args.output is None:
        root = Path(__file__).resolve().parents[1] / "outputs"
        try:
            root.mkdir(exist_ok=True)
            run = Path(tempfile.mkdtemp(prefix=f"{args.task}-", dir=root))
        except OSError as error:
            parser.error(f"Cannot create output under {root}: {error}. Specify --output with a writable path.")
        args.output = str(run if args.task == "train" else run / "predictions.tsv")
        print(f"Output: {args.output}", file=sys.stderr, flush=True)
    return args


def apply_config(args: argparse.Namespace) -> argparse.Namespace:
    model_dir = Path(args.model_dir or ".").expanduser()
    config_path = (
        Path(args.config).expanduser()
        if args.config
        else model_dir / "config" / "inference.yaml"
    )
    if not config_path.is_file():
        raise FileNotFoundError(f"Inference config not found: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    args.inference_overrides = {}
    if args.config and getattr(args, "task", "denovo") != "train":
        frozen_path = model_dir / "config" / "inference.yaml"
        frozen = yaml.safe_load(frozen_path.read_text(encoding="utf-8"))
        changed = [
            key
            for key, value in config.items()
            if key not in {"runtime", "n_beams"} and key not in EDITABLE
            and (key not in frozen or value != frozen[key])
        ]
        if "residues" in config and list(config["residues"]) != list(frozen.get("residues", {})):
            raise ValueError("Residue order must match the checkpoint vocabulary")
        args.beam_width = config.get("n_beams", getattr(args, "beam_width", None))
        if changed:
            raise ValueError(
                f"Cannot override frozen inference fields: {sorted(changed)}"
            )
        effective = resolve_settings(frozen, {k: config[k] for k in EDITABLE if k in config})
        args.inference_overrides = {k: effective[k] for k in EDITABLE
                                    if k in effective and effective[k] != frozen.get(k)}
        overrides = config.get("runtime", {})
        if not isinstance(overrides, dict):
            raise TypeError(f"'runtime' must be a mapping in {config_path}")
        config = dict(effective, runtime={**frozen.get("runtime", {}), **overrides})
        config_path = frozen_path
    runtime = config.get("runtime", {})
    if not isinstance(runtime, dict):
        raise TypeError(f"'runtime' must be a mapping in {config_path}")

    args.mode = args.mode or runtime.get("mode", "fast")
    if getattr(args, "candidate_mode", None) is None:
        args.candidate_mode = "union" if config.get("PMC_enable", True) else "beam-only"
    elif args.candidate_mode == "union" and not config.get("PMC_enable", True):
        raise ValueError("runtime.candidate_mode union conflicts with PMC_enable: false")
    width = getattr(args, "beam_width", None)
    if width is None:
        width = runtime.get("beam_width")
    if width is not None and (type(width) is not int or width < 1):
        raise ValueError("beam_width must be a positive integer")
    # The default keeps the original runtime identity for existing CCT profiles.
    args.beam_width = None if width == config.get("n_beams") else width
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
    fdr_mode = getattr(args, "fdr_mode", "off")
    requested_fdr_mode = fdr_mode
    cct_enabled = fdr_mode == "on"
    cct_features = getattr(args, "cct_features_output", None)
    if cct_enabled:
        if not getattr(args, "cct_profile", None):
            raise ValueError("runtime.fdr_mode=on requires runtime.cct_profile; fitting is separate")
        cct_features = cct_features or str(Path(args.output).with_suffix(".cct_features.csv"))
        fdr_mode = "off"
    elif getattr(args, "cct_profile", None):
        raise ValueError("runtime.cct_profile requires runtime.fdr_mode=on")
    if cct_features and (fdr_mode != "off" or not getattr(args, "cct_scope_id", None)):
        raise ValueError("CCT export requires runtime.cct_scope_id and no legacy confidence filtering")
    indices = validate_selection(
        np.load(args.indices, allow_pickle=False) if args.indices else None,
        total_spectra,
        args.max_samples,
        args.task,
    )
    progress_total = len(indices) if indices is not None else total_spectra
    if progress_total == 0:
        if cct_features:
            raise ValueError("CCT input contains no spectra; no completion artifact produced")
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(
            OUTPUT_HEADER,
            encoding="utf-8",
        )
        temporary.replace(output)
        print(json.dumps({"status": "no_accepted_spectra", "processed": 0}))
        return 0
    options = RuntimeOptions(
        mode=args.mode,
        candidate_mode=args.candidate_mode,
        batch_size=args.batch_size,
        n_workers=args.n_workers,
        ctc_processes=args.ctc_processes,
        initial_beam=getattr(args, "beam_width", None),
        inference_overrides=getattr(args, "inference_overrides", None),
    )
    cct_profile = None
    cct_output = Path(args.output).with_suffix(".cct")
    if cct_features:
        from MNovo.cct.cct_pipeline import runtime_identity, load_profile
        artifacts = [Path(p).resolve() for p in (args.output, cct_features,
                     str(cct_features) + ".receipt.json", cct_output,
                     str(args.output) + ".tmp", str(args.output) + ".final.tmp")]
        if len(set(artifacts)) != len(artifacts):
            raise ValueError("Native predictions, CCT features and QC outputs must be distinct")
        if any(cct_output.resolve() in p.parents for p in artifacts):
            raise ValueError("CCT features/native predictions cannot be inside QC output directory")
        identity = runtime_identity(args.model_dir, args.config, options, args.device)
        if any(Path(p).exists() for p in (cct_features, str(cct_features) + ".receipt.json", args.output, cct_output)):
            raise FileExistsError("CCT output already exists; choose a new output path")
        if cct_enabled:
            cct_profile = load_profile(args.cct_profile, identity, args.cct_scope_id,
                                       args.psm_fdr, args.peptide_fdr)
    runtime = MNovoRuntime(
        args.model_dir, options, device=args.device, fdr_mode=fdr_mode
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    exporter = None
    if cct_features:
        from MNovo.cct.cct_runtime import CCTFeatureExporter
        Path(cct_features).parent.mkdir(parents=True, exist_ok=True)
        exporter = CCTFeatureExporter(runtime.backbone.decoder, runtime.config["residues"], cct_features)
        runtime.cct_exporter = exporter
    started = time.time()
    count = 0
    route_counts = {"r1": 0, "r2_long": 0, "r3_fragment": 0, "none": 0}
    temporary = output.with_suffix(output.suffix + ".tmp")
    with (
        exporter if exporter is not None else nullcontext(),
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
                public_row = output_row(
                            record,
                            prediction.dataset_index,
                            prediction.peptide,
                            prediction.confidence,
                            runtime.config["residues"],
                        )
                row = public_row + [prediction.status, prediction.route,
                                    str(prediction.dataset_index)]
                handle.write("\t".join(row) + "\n")
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
                                    "beam_width": runtime.beam_width,
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
    if exporter is not None:
        from MNovo.cct.cct_pipeline import write_receipt
        if exporter.complete_pairs + exporter.invalid_pairs != count:
            raise RuntimeError("CCT paired feature accounting differs from native predictions")
        write_receipt(cct_features, identity, args.cct_scope_id, exporter)
    temporary.replace(output)
    if cct_enabled:
        from MNovo.cct.cct_pipeline import apply_profile, _sha256
        cct_report = apply_profile(cct_features, cct_profile, cct_output)
        cct_report["profile_sha256"] = _sha256(args.cct_profile)
        cct_report["native_predictions"] = str(output.resolve())
        cct_report["native_predictions_sha256"] = _sha256(output)
        (cct_output / "report.json").write_text(json.dumps(cct_report, indent=2))
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
                "fdr_mode": requested_fdr_mode,
                "fdr_passed": cct_report["accepted_psms"] if cct_enabled else None,
                "output_rows": count,
                "seconds": elapsed,
                "spectra_per_second": count / elapsed,
                "candidate_generation": {
                    "beam_width": runtime.beam_width,
                    "pmc_mode": "union" if args.candidate_mode == "union" else "off",
                    "statistics": dict(runtime.candidate_statistics),
                    "dynamic_expansion": False,
                },
                "route_counts": route_counts,
                "input_type": "mgf" if input_files else "lmdb",
                "input_files": len(input_files) if input_files else 1,
                "config": args.config,
                **({"inference_overrides": args.inference_overrides}
                   if getattr(args, "inference_overrides", None) else {}),
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


def _check_inference_paths(args, metrics_output):
    output = Path(args.output)
    inputs = [args.config, args.indices, args.cct_profile, args.lmdb]
    if getattr(args, "cct_fit", None) and args.fdr_mode == "on":
        inputs.extend(args.cct_fit.get(k) for k in ("roles", "reference_table"))
    model = Path(args.model_dir).expanduser()
    if model.is_dir():
        inputs.extend([model / "verified.json", model / "config" / "inference.yaml"])
        inputs.extend((model / "components").glob("*"))
    else:
        inputs.append(model)
    outputs = [output, str(output) + ".tmp"]
    if args.input:
        try:
            inputs.extend(resolve_mgf_inputs(args.input))
        except FileNotFoundError:
            # The input stage records an unmatched pattern in the evaluation status.
            if Path(args.input).is_file():
                inputs.append(args.input)
        outputs.extend(InputAudit(output).paths.values())
    if args.task == "eval":
        outputs.append(metrics_output)
    features = args.cct_features_output
    if args.fdr_mode == "on":
        features = features or output.with_suffix(".cct_features.csv")
        outputs.append(output.with_suffix(".cct"))
        if not args.cct_profile or args.cct_fit.get("reference_table"):
            work = output.with_suffix(".cct-work")
            if work.exists():
                raise FileExistsError(f"CCT work directory already exists: {work}. Choose a fresh output.")
            outputs.append(work)
    if features:
        outputs.extend([features, str(features) + ".receipt.json", str(output) + ".final.tmp"])
    if args.gpus:
        outputs.extend([str(output) + ".workers", str(output) + ".partial"])
    check_output_paths(inputs, outputs)


def main() -> None:
    parsed = parse_args()
    metrics_output = parsed.metrics_output or str(Path(parsed.output).with_suffix(".metrics.json"))
    if parsed.max_samples < 0:
        raise ValueError("runtime.max_samples must be non-negative.")
    if parsed.task == "eval" and (parsed.indices or parsed.max_samples):
        raise ValueError(
            "eval requires all input spectra; subset parameters are unsupported."
        )
    if parsed.task != "train":
        _check_inference_paths(parsed, metrics_output)
    if parsed.task == "eval":
        write_metrics_state(metrics_output, dict(status="in_progress", peptide_recall=None,
                                                 aa_precision=None, aa_recall=None))
    phase = "initialization"
    try:
        if parsed.task != "train":
            parsed.model_dir = str(resolve_model_release(parsed.model_dir))
        config_was_explicit = parsed.config is not None
        args = apply_config(parsed)
        if args.task == "train":
            if args.gpus:
                raise ValueError("For training set trainer_devices in the YAML; runtime.gpus is the inference selector.")
            if args.lmdb:
                raise ValueError("--model train requires MGF input, not runtime.input_format=lmdb.")
            if not config_was_explicit:
                raise ValueError("--model train requires an explicit training --config.")
            if not args.validation_input:
                raise ValueError("--model train requires runtime.validation_input.")
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

        phase = "input"
        audit = None
        if args.lmdb:
            lmdb_path = str(Path(args.lmdb).expanduser())
            context = nullcontext((lmdb_path, [], lmdb_spectra_count(lmdb_path)))
        else:
            sources = resolve_mgf_inputs(args.input)
            if args.fdr_mode == "on" and args.cct_fit["reference_table"]:
                from MNovo.cct.workflow import annotate_sources
                sources = annotate_sources(sources, args.cct_fit, Path(args.output).with_suffix(".cct-work") / "labelled_mgf",
                                           yaml.safe_load(Path(args.config).read_text())["residues"])
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
                            preprocessing_config=resolve_settings(
                                yaml.safe_load(Path(args.config).read_text(encoding="utf-8")),
                                args.inference_overrides),
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
            phase = "prediction"
            try:
                from MNovo.inference.multi_gpu import run_selected_devices
                if args.fdr_mode == "on":
                    from MNovo.cct.workflow import run_workflow
                    predicted = run_workflow(args, lmdb, input_files, total_spectra, run_selected_devices)
                else:
                    predicted = run_selected_devices(args, lmdb, input_files, total_spectra)
            except Exception:
                if audit:
                    audit.save("prediction_failed")
                raise
            if audit:
                not_selected = total_spectra - predicted
                status = "complete_with_rejections" if audit.rejected else "complete"
                if not predicted:
                    status = "no_predictions"
                result = audit.save(status, predicted, not_selected)
                print(json.dumps({"input_accounting": result}, ensure_ascii=False, indent=2))
            if args.task == "eval":
                phase = "evaluation"
                result = evaluate_predictions(
                    args.output,
                    lmdb,
                    args.config,
                    metrics_output,
                    inference_overrides=args.inference_overrides,
                    input_counts=dict(original_input=audit.total, accepted=audit.accepted,
                                      rejected=audit.rejected) if audit else None,
                )
                print(json.dumps({"evaluation": result}, indent=2, sort_keys=True))
    except Exception as error:
        if parsed.task == "eval":
            write_metrics_state(metrics_output, dict(
                status=f"{phase}_failed", error_type=type(error).__name__, error=str(error),
                peptide_recall=None, aa_precision=None, aa_recall=None,
            ))
        raise


if __name__ == "__main__":
    main()
