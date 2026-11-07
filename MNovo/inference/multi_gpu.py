# Distribute complete batches; merge results by input indices before cohort filtering.
# 按完整批次分配任务，按输入索引合并后再做整批质控。
"""Process-per-device inference with whole-batch sharding and global FDR."""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


def batch_shards(indices, batch_size, workers):
    if batch_size < 1 or workers < 1:
        raise ValueError("Positive batch size and worker count required")
    pieces = [[] for _ in range(workers)]
    for n, start in enumerate(range(0, len(indices), batch_size)):
        pieces[n % workers].extend(indices[start:start + batch_size])
    return [np.asarray(p, dtype=np.int64) for p in pieces if p]


def selected_devices(spec, visible=None):
    values = spec.split(",")
    if not all(v.isdigit() for v in values) or len(set(values)) != len(values):
        raise ValueError("runtime.gpus requires distinct non-negative visible indices")
    indices = [int(v) for v in values]
    if visible is None:
        return [str(v) for v in indices]
    allowed = visible.split(",") if visible else []
    if any(v >= len(allowed) for v in indices):
        raise ValueError("runtime.gpus exceeds CUDA_VISIBLE_DEVICES")
    return [allowed[v] for v in indices]


def merge_predictions(paths, indices, output):
    records, header = {}, None
    for path in paths:
        with Path(path).open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if header is not None and reader.fieldnames != header:
                raise ValueError("Worker prediction headers differ")
            header = reader.fieldnames
            for row in reader:
                key = int(row["dataset_index"])
                if key in records:
                    raise ValueError(f"Duplicate prediction index {key}")
                records[key] = row
    if set(records) != set(map(int, indices)):
        raise ValueError("Worker predictions missing or contain unexpected indices")
    target = Path(output)
    temporary = target.with_suffix(target.suffix + ".partial")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(records[int(i)] for i in indices)
    temporary.replace(target)


def merge_features(paths, indices, output):
    import pandas as pd
    from MNovo.cct.cct_pipeline import SCHEMA, inference_contract
    from MNovo.cct.cct_scoring import _sha256
    receipts = [json.loads(Path(str(p) + ".receipt.json").read_text()) for p in paths]
    for receipt in receipts:
        receipt["runtime_identity"] = inference_contract(receipt["runtime_identity"])
    for p, r in zip(paths, receipts):
        if r["features_sha256"] != _sha256(p):
            raise ValueError("Worker feature checksum mismatch")
        if any(r[k] != receipts[0][k] for k in ("schema", "runtime_identity", "scope_id")):
            raise ValueError("Worker inference identities differ")
    rows = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    if rows.duplicated(["pair_id", "is_decoy"]).any():
        raise ValueError("Duplicate worker feature pair")
    if set(rows.pair_id) != set(map(int, indices)) or not rows.groupby("pair_id").is_decoy.agg(lambda x: set(x) == {0, 1}).all():
        raise ValueError("Missing worker feature views")
    # Restore global input order, not worker completion order.
    # 按原始输入索引还原顺序，而不按 worker 完成顺序写结果。
    order = {int(i): n for n, i in enumerate(indices)}
    rows["_order"] = rows.pair_id.map(order)
    rows.sort_values(["_order", "is_decoy"]).drop(columns="_order").to_csv(output, index=False)
    receipt = dict(receipts[0], schema=SCHEMA, features_sha256=_sha256(output))
    for key in ("complete_pairs", "invalid_pairs", "labelled_pairs"):
        receipt[key] = sum(r[key] for r in receipts)
    Path(str(output) + ".receipt.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def run_selected_devices(args, lmdb, input_files, total_spectra):
    from MNovo.cli import run
    from MNovo.inference.selection import validate_selection
    if not args.gpus:
        return run(args, lmdb, input_files, total_spectra)
    if args.device not in ("auto", "cuda", "cuda:0"):
        raise ValueError("runtime.gpus selects CUDA devices; do not combine with a different runtime.device")
    devices = selected_devices(args.gpus, os.environ.get("CUDA_VISIBLE_DEVICES"))
    indices = validate_selection(np.load(args.indices, allow_pickle=False) if args.indices else None,
                                 total_spectra, args.max_samples, args.task)
    if indices is None:
        indices = np.arange(total_spectra, dtype=np.int64)
    if not len(indices):
        return run(args, lmdb, input_files, total_spectra)
    output = Path(args.output).resolve()
    work = output.with_suffix(output.suffix + ".workers")
    features = args.cct_features_output
    if args.fdr_mode == "on":
        if not args.cct_profile or not args.cct_scope_id:
            raise ValueError("FDR requires runtime.cct_profile and runtime.cct_scope_id")
        features = features or str(output.with_suffix(".cct_features.csv"))
    elif args.cct_profile:
        raise ValueError("runtime.cct_profile requires runtime.fdr_mode=on")
    reserved = [output, work, output.with_suffix(".cct")]
    if features:
        reserved += [Path(features).resolve(), Path(str(features) + ".receipt.json").resolve()]
        if not args.cct_scope_id:
            raise ValueError("Feature export requires runtime.cct_scope_id")
    if len(set(reserved)) != len(reserved) or any(p.exists() for p in reserved):
        raise FileExistsError("Output artifacts must be distinct and absent")
    if any(output.with_suffix(".cct") in p.parents or work in p.parents for p in reserved):
        raise ValueError("Final outputs cannot be placed inside worker or FDR directories")
    if args.fdr_mode == "on":
        from MNovo.cct.cct_pipeline import load_profile, runtime_identity
        from MNovo.inference.runtime import RuntimeOptions
        options = RuntimeOptions(mode=args.mode, candidate_mode=args.candidate_mode,
                                 batch_size=args.batch_size, n_workers=args.n_workers,
                                 ctc_processes=args.ctc_processes,
                                 initial_beam=getattr(args, "beam_width", None),
                                 inference_overrides=getattr(args, "inference_overrides", None))
        load_profile(args.cct_profile, runtime_identity(args.model_dir, args.config, options, "cuda"),
                     args.cct_scope_id, args.psm_fdr, args.peptide_fdr)
    work.mkdir(parents=True)
    processes, handles, outputs, feature_paths = [], [], [], []
    try:
        for n, shard in enumerate(batch_shards(indices, args.batch_size, len(devices))):
            index_file = work / f"worker{n}.npy"
            np.save(index_file, shard)
            child = vars(args).copy()
            prediction = work / f"worker{n}.tsv"
            feature = work / f"worker{n}.csv" if features else None
            child.update(gpus=None, device="cuda:0", task="denovo", indices=str(index_file),
                         max_samples=0, output=str(prediction), fdr_mode="off", cct_profile=None,
                         cct_features_output=str(feature) if feature else None)
            spec = work / f"worker{n}.json"
            spec.write_text(json.dumps(dict(args=child, lmdb=lmdb, total=total_spectra)))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices[n])
            handle = (work / f"worker{n}.log").open("w")
            handles.append(handle)
            processes.append(subprocess.Popen([sys.executable, "-m", "MNovo.inference.multi_gpu", str(spec)],
                                              env=env, stdout=handle, stderr=subprocess.STDOUT,
                                              start_new_session=True))
            outputs.append(prediction)
            if feature:
                feature_paths.append(feature)
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError(f"GPU worker failed; inspect {work}")
            time.sleep(.2)
        if any(p.returncode for p in processes):
            raise RuntimeError(f"GPU worker failed; inspect {work}")
        merge_predictions(outputs, indices, output)
        if features:
            Path(features).parent.mkdir(parents=True, exist_ok=True)
            receipt = merge_features(feature_paths, indices, features)
            if args.fdr_mode == "on":
                from MNovo.cct.cct_pipeline import load_profile, apply_profile
                profile = load_profile(args.cct_profile, receipt["runtime_identity"], args.cct_scope_id,
                                       args.psm_fdr, args.peptide_fdr)
                apply_profile(features, profile, output.with_suffix(".cct"))
        (work / "complete.json").write_text(json.dumps(dict(rows=len(indices), devices=devices,
                                                            whole_batches=True, fdr_scope="global")))
        return len(indices)
    except BaseException as error:
        for p in processes:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        for p in processes:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
        (work / "failure.json").write_text(json.dumps(dict(error=str(error))))
        raise
    finally:
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    from MNovo.cli import run
    payload = json.loads(Path(sys.argv[1]).read_text())
    run(argparse.Namespace(**payload["args"]), payload["lmdb"], [], payload["total"])
