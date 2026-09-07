"""Audit nested checkpoint metadata and optionally write a tensor-identical candidate."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import sys

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from MNovo.release import _file_sha256

LOCAL_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|/(?:home|data|mnt|Users|tmp|root)/)")


def clean_metadata(value, findings, location="root"):
    if isinstance(value, str) and LOCAL_PATH.search(value):
        digest = hashlib.sha256(value.encode()).hexdigest()
        findings.append(dict(location=location, value_sha256=digest))
        return f"local-artifact-sha256:{digest}"
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            new_key = clean_metadata(key, findings, location + ".<key>")
            if new_key in output:
                raise ValueError("Metadata key collision during sanitization")
            output[new_key] = clean_metadata(
                item, findings, location + "." + str(new_key)
            )
        return output
    if isinstance(value, list):
        return [
            clean_metadata(item, findings, f"{location}[{i}]")
            for i, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        return tuple(
            clean_metadata(item, findings, f"{location}[{i}]")
            for i, item in enumerate(value)
        )
    return value


def tensor_inventory(value, prefix="root"):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from tensor_inventory(item, prefix + "." + str(key))
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            yield from tensor_inventory(item, f"{prefix}[{i}]")


def audit_checkpoint(source, output=None):
    source = Path(source).resolve()
    if output and (Path(output).resolve() == source or Path(output).exists()):
        raise ValueError(
            "Candidate output must be a new file; never overwrite a published asset."
        )
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    findings = []
    candidate = clean_metadata(checkpoint, findings)
    tensor_count = 0
    for name, payload in checkpoint["component_payloads"].items():
        if name.endswith((".yaml", ".yml")):
            original = yaml.safe_load(payload.decode("utf-8"))
            cleaned_yaml = clean_metadata(original, findings, name)
            if output:
                serialized = yaml.safe_dump(cleaned_yaml, sort_keys=False).encode(
                    "utf-8"
                )
                candidate["component_payloads"][name] = serialized
                metadata = candidate["manifest"]["components"][name]
                metadata["sha256"] = hashlib.sha256(serialized).hexdigest()
                for size_key in ("size", "bytes", "size_bytes"):
                    if size_key in metadata:
                        metadata[size_key] = len(serialized)
            continue
        if not name.endswith((".pt", ".ckpt", ".pth")):
            continue
        component = torch.load(
            io.BytesIO(payload), map_location="cpu", weights_only=True
        )
        cleaned = clean_metadata(component, findings, name)
        if output:
            buffer = io.BytesIO()
            torch.save(cleaned, buffer)
            serialized = buffer.getvalue()
            reloaded = torch.load(
                io.BytesIO(serialized), map_location="cpu", weights_only=True
            )
            before, after = (
                dict(tensor_inventory(component)),
                dict(tensor_inventory(reloaded)),
            )
            if before.keys() != after.keys():
                raise RuntimeError(f"Tensor keys changed: {name}")
            for key in before:
                a, b = before[key], after[key]
                if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
                    raise RuntimeError(f"Tensor changed: {name}:{key}")
            tensor_count += len(before)
            candidate["component_payloads"][name] = serialized
            metadata = candidate["manifest"]["components"][name]
            metadata["sha256"] = hashlib.sha256(serialized).hexdigest()
            for size_key in ("size", "bytes", "size_bytes"):
                if size_key in metadata:
                    metadata[size_key] = len(serialized)
    result = dict(
        source_sha256=_file_sha256(source),
        findings=findings,
        occurrences=len(findings),
        unique_values=len({f["value_sha256"] for f in findings}),
        status="AUDIT_ONLY",
    )
    if output:
        candidate["manifest"]["model_name"] = "pi-MNovo-metadata-candidate-20260907"
        candidate["manifest"]["status"] = "CANDIDATE"
        candidate["manifest"]["manuscript_eligible"] = False
        candidate["review_candidate"] = dict(
            status="CANDIDATE",
            manuscript_eligible=False,
            source_sha256=result["source_sha256"],
            change="recursive local-path metadata sanitization; tensors unchanged",
        )
        target = Path(output)
        torch.save(candidate, target)
        result.update(
            output_sha256=_file_sha256(target),
            tensor_count_verified=tensor_count,
            status="CANDIDATE_NOT_PROMOTED",
        )
        second = audit_checkpoint(target)
        if second["occurrences"]:
            raise RuntimeError("Sanitized candidate still contains local paths")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--output", help="Optional new, unpublished candidate checkpoint"
    )
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    result = audit_checkpoint(args.checkpoint, args.output)
    Path(args.report).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "findings"}, indent=2))


if __name__ == "__main__":
    main()
