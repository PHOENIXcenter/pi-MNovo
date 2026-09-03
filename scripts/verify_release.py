#!/usr/bin/env python3
"""Verify the unified checkpoint and its frozen release settings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MNovo.release import resolve_model_release

EXPECTED_SHA256 = "1de589a887a7b7271aae794b90850ece6c3c68c6d085374d0b32e40ec18d617f"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    observed = file_sha256(checkpoint)
    if observed != EXPECTED_SHA256:
        raise RuntimeError(
            f"Checkpoint SHA256 mismatch: expected {EXPECTED_SHA256}, got {observed}"
        )
    with tempfile.TemporaryDirectory(prefix="pi_mnovo_verify_") as cache:
        previous_home = os.environ.get("HOME")
        os.environ["HOME"] = cache
        try:
            # Extraction verifies the hash of every embedded component.
            root = resolve_model_release(checkpoint)
        finally:
            if previous_home is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = previous_home
        config = yaml.safe_load(
            (root / "config" / "inference.yaml").read_text(encoding="utf-8")
        )
        required = {
            "backbone.ckpt",
            "r1_ranker.pt",
            "r2_long_expert.pt",
            "r3_fragment_expert.pt",
            "length_predictor.pt",
            "observable_router.pt",
        }
        present = {path.name for path in (root / "components").iterdir()}
        missing = sorted(required - present)
        if missing:
            raise RuntimeError(f"Unified checkpoint is missing: {missing}")
        runtime = config.get("runtime", {})
        if runtime.get("mode") != "fast":
            raise RuntimeError("Release mode must be fast.")
        print(
            json.dumps(
                {
                    "status": "verified",
                    "checkpoint": str(checkpoint),
                    "sha256": observed,
                    "components": sorted(required),
                    "runtime": runtime,
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
