"""Install a verified unified MNovo checkpoint into a runtime cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import yaml


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_model_release(source: str | Path) -> Path:
    """Return a model directory, extracting unified checkpoints when needed."""
    source = Path(source).expanduser().resolve()
    if source.is_dir():
        marker = source / "verified.json"
        if not marker.is_file():
            raise ValueError(
                "pi-MNovo release inference requires the unified checkpoint "
                "or its verified runtime cache; arbitrary component "
                "directories such as models/final_components are unsupported."
            )
        return source
    if not source.is_file():
        raise FileNotFoundError(f"MNovo model release not found: {source}")

    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "mnovo-unified-sequence-checkpoint":
        raise ValueError(f"Unsupported MNovo checkpoint format: {source}")
    manifest = checkpoint["manifest"]
    payloads = checkpoint["component_payloads"]
    release_id = _file_sha256(source)[:16]
    root = Path.home() / ".cache" / "mnovo" / "releases" / release_id
    marker = root / "verified.json"
    expected = {
        name: metadata["sha256"] for name, metadata in manifest["components"].items()
    }
    if marker.is_file():
        recorded = json.loads(marker.read_text(encoding="utf-8"))
        if recorded.get("components") == expected:
            return root

    components = root / "components"
    config_dir = root / "config"
    components.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    aliases = {
        "router.pt": "observable_router.pt",
    }
    for name, metadata in manifest["components"].items():
        payload = payloads[name]
        digest = _sha256(payload)
        if digest != metadata["sha256"]:
            raise RuntimeError(f"Unified checkpoint component hash mismatch: {name}")
        if name == "MNovo_config.yaml":
            config = yaml.safe_load(payload.decode("utf-8")) or {}
            config.pop("load_file_name", None)
            config["runtime"] = {
                "mode": "fast",
                "batch_size": 128,
                "n_workers": 16,
                "ctc_processes": 16,
            }
            config.pop("fdr", None)
            (config_dir / "inference.yaml").write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            continue
        target = components / aliases.get(name, name)
        target.write_bytes(payload)
    marker.write_text(
        json.dumps(
            {"source": str(source), "manifest": manifest, "components": expected},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return root
