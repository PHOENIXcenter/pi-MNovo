#!/usr/bin/env python3
"""Download and verify the pi-MNovo v0.1.0 unified checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import tempfile
import urllib.request
from pathlib import Path

MODEL_NAME = "pi-MNovo-v0.1.0.ckpt"
MODEL_URL = (
    f"https://github.com/PHOENIXcenter/pi-MNovo/releases/download/v0.1.0/{MODEL_NAME}"
)
MODEL_SHA256 = "1de589a887a7b7271aae794b90850ece6c3c68c6d085374d0b32e40ec18d617f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=f"models/{MODEL_NAME}")
    args = parser.parse_args()
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=output.name + ".",
                                     suffix=".part", delete=False) as handle:
        partial = Path(handle.name)
    try:
        print(f"Downloading {MODEL_URL}")
        urllib.request.urlretrieve(MODEL_URL, partial)
        observed = sha256(partial)
        if observed != MODEL_SHA256:
            raise RuntimeError(
                f"Checkpoint SHA256 mismatch: expected {MODEL_SHA256}, got {observed}"
            )
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)
    print(f"Verified checkpoint: {output.resolve()}")


if __name__ == "__main__":
    main()
