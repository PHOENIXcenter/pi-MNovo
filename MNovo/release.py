"""Verify every cached byte; publish extraction only after complete validation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import torch
import yaml

REQUIRED = {
    "backbone.ckpt",
    "r1_ranker.pt",
    "r2_long_expert.pt",
    "r3_fragment_expert.pt",
    "length_predictor.pt",
    "MNovo_config.yaml",
}
ALIASES = {"router.pt": "observable_router.pt"}
INFERENCE_KEYS = {
    "PMC_enable",
    "mass_control_tol",
    "dim_model",
    "n_head",
    "dim_feedforward",
    "n_layers",
    "dropout",
    "dim_intensity",
    "custom_encoder",
    "max_length",
    "residues",
    "max_charge",
    "precursor_mass_tol",
    "isotope_error_range",
    "n_log",
    "n_peaks",
    "min_mz",
    "max_mz",
    "min_intensity",
    "remove_precursor_tol",
}


def _sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_manifest(manifest):
    components = manifest["components"]
    for name, metadata in components.items():
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or ".." in name:
            raise ValueError(f"Unsafe component filename: {name!r}")
        if not re.fullmatch(r"[a-f0-9]{64}", metadata["sha256"]):
            raise ValueError(f"Invalid component SHA256: {name}")
    if not REQUIRED.issubset(components):
        raise ValueError(
            f"Missing release components: {sorted(REQUIRED - components.keys())}"
        )
    if len({"router.pt", "observable_router.pt"} & components.keys()) != 1:
        raise ValueError("Release requires exactly one observable router component.")
    destinations = [ALIASES.get(name, name) for name in components]
    if len(set(destinations)) != len(destinations):
        raise ValueError("Duplicate component destinations.")
    return {name: item["sha256"] for name, item in components.items()}


def _inference_config(payload):
    original = yaml.safe_load(payload.decode("utf-8"))
    config = {key: value for key, value in original.items() if key in INFERENCE_KEYS}
    config["runtime"] = dict(
        mode="fast", batch_size=128, n_workers=16, ctc_processes=16
    )
    return yaml.safe_dump(config, sort_keys=False).encode("utf-8")


def _verify_directory(root, expected=None):
    marker = root / "verified.json"
    if not marker.is_file() or marker.is_symlink():
        raise ValueError("Missing verified runtime cache marker.")
    recorded = json.loads(marker.read_text(encoding="utf-8"))
    if recorded.get("cache_schema") != 2:
        raise ValueError(
            "Legacy cache cannot be trusted; reload the unified checkpoint to regenerate it."
        )
    hashes = _validate_manifest(recorded["manifest"])
    if expected is not None and hashes != expected:
        raise RuntimeError("Cached manifest does not match the source checkpoint.")
    for name, digest in hashes.items():
        target = root / "components" / ALIASES.get(name, name)
        if (
            target.is_symlink()
            or not target.is_file()
            or not target.resolve().is_relative_to(root.resolve())
            or _file_sha256(target) != digest
        ):
            raise RuntimeError(f"Cached component hash mismatch: {name}")
    config = root / "config" / "inference.yaml"
    original = (root / "components" / "MNovo_config.yaml").read_bytes()
    if (
        config.is_symlink()
        or not config.is_file()
        or not config.resolve().is_relative_to(root.resolve())
        or config.read_bytes() != _inference_config(original)
    ):
        raise RuntimeError("Cached inference config mismatch.")
    return root


def resolve_model_release(
    source: str | Path, cache_root: str | Path | None = None
) -> Path:
    """Directory caches detect corruption; use a pinned source SHA for authenticity."""
    source = Path(source).expanduser().resolve()
    if source.is_dir():
        return _verify_directory(source)
    if not source.is_file():
        raise FileNotFoundError(f"MNovo model release not found: {source}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "mnovo-unified-sequence-checkpoint":
        raise ValueError(f"Unsupported MNovo checkpoint format: {source}")
    manifest = checkpoint["manifest"]
    expected = _validate_manifest(manifest)
    payloads = checkpoint["component_payloads"]
    if set(payloads) != set(expected):
        raise ValueError("Component payload inventory does not match manifest.")
    for name, digest in expected.items():
        if _sha256(payloads[name]) != digest:
            raise RuntimeError(f"Unified checkpoint component hash mismatch: {name}")
    parent = (
        Path(cache_root).expanduser().resolve()
        if cache_root
        else Path.home() / ".cache" / "mnovo" / "releases"
    )
    parent.mkdir(parents=True, exist_ok=True)
    root = parent / (_file_sha256(source) + "-cache2")
    if root.exists():
        return _verify_directory(root, expected)
    with tempfile.TemporaryDirectory(prefix=".extract-", dir=parent) as staging:
        staged = Path(staging) / "release"
        (staged / "components").mkdir(parents=True)
        (staged / "config").mkdir()
        for name, payload in payloads.items():
            (staged / "components" / ALIASES.get(name, name)).write_bytes(payload)
        (staged / "config" / "inference.yaml").write_bytes(
            _inference_config(payloads["MNovo_config.yaml"])
        )
        (staged / "verified.json").write_text(
            json.dumps(
                {
                    "cache_schema": 2,
                    "manifest": manifest,
                    "source_sha256": _file_sha256(source),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        _verify_directory(staged, expected)
        try:
            os.rename(staged, root)
        except OSError:
            if not root.exists():
                raise
            # A concurrent extractor may have finished first; accept only verified bytes.
            _verify_directory(root, expected)
    return _verify_directory(root, expected)
