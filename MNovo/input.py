"""Resolve user-facing MGF inputs and materialize them for MNovo inference."""

from __future__ import annotations

import glob
import pickle
import re
from pathlib import Path
from typing import Iterable

import lmdb
import numpy as np


class LmdbTitleLookup:
    """Read original spectrum titles from one inference LMDB."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self.environment = None
        self.transaction = None

    def __enter__(self):
        self.environment = lmdb.open(
            self.path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
        )
        self.transaction = self.environment.begin()
        return self

    def spectrum(self, index: int) -> dict:
        buffer = self.transaction.get(str(int(index)).encode())
        if buffer is None:
            raise IndexError(f"Spectrum index is missing from LMDB: {index}")
        return pickle.loads(buffer)

    def title(self, index: int) -> str:
        data = self.spectrum(index)
        value = data.get("title", data.get("pep", f"index={index}"))
        return str(value)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.transaction is not None:
            self.transaction.abort()
        if self.environment is not None:
            self.environment.close()


def _case_insensitive_mgf_pattern(pattern: str) -> str:
    return re.sub(r"(?i)\.mgf", ".[mM][gG][fF]", pattern)


def _mgf_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix.lower() == ".mgf":
            files.append(path.resolve())
        elif path.is_dir():
            files.extend(
                child.resolve()
                for child in path.rglob("*")
                if child.is_file() and child.suffix.lower() == ".mgf"
            )
    return sorted(set(files), key=lambda path: (str(path).lower(), str(path)))


def resolve_mgf_inputs(value: str | Path) -> list[Path]:
    """Resolve one MGF file, a directory, or a glob, case-insensitively."""
    raw = str(value)
    path = Path(raw).expanduser()
    if path.exists():
        files = _mgf_files([path])
    elif glob.has_magic(raw):
        matches = [
            Path(match)
            for match in glob.glob(
                _case_insensitive_mgf_pattern(raw),
                recursive=True,
            )
        ]
        files = _mgf_files(matches)
    else:
        files = []
    if not files:
        raise FileNotFoundError(
            "No MGF files matched input. Accepted inputs are a single MGF "
            "file, a directory, or a glob such as '*.mgf' or '**/*.mgf'."
        )
    return files


def materialize_mgf_lmdb(
    sources: list[Path],
    lmdb: Path,
    max_charge: int,
    annotated: bool = False,
    audit=None,
    preprocessing_config=None,
) -> int:
    """Write resolved MGF spectra to a temporary inference LMDB."""
    from MNovo.denovo.spectrum_index import LmdbSpectrumIndex
    from MNovo.denovo.spectrum_parsers import MgfParser
    from MNovo.input_audit import spectra
    from MNovo.denovo.spectrum_dataset import SpectrumDataset

    config = preprocessing_config or {}
    preprocessor = SpectrumDataset([], **{
        key: config[key] for key in (
            "n_peaks", "min_mz", "max_mz", "min_intensity", "remove_precursor_tol"
        ) if key in config
    })

    database = LmdbSpectrumIndex(
        str(lmdb),
        None,
        ms_level=2,
        valid_charge=np.arange(1, max_charge + 1),
        annotated=annotated,
        lock=True,
    )
    try:
        for path in sources:
            parser = MgfParser(path, valid_charge=np.arange(1, max_charge + 1),
                               annotationsLabel=annotated)
            for spectrum in spectra(path, max_charge, audit, len(database), preprocessor):
                parser.parse_spectrum(spectrum)
            database.write_spectra(parser, len(parser.precursor_charge))
        count = len(database)
    finally:
        database.env.close()
    return count


def lmdb_spectra_count(path: str | Path) -> int:
    """Read the spectrum count without constructing a dataset."""
    environment = lmdb.open(
        str(path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
    )
    try:
        with environment.begin() as transaction:
            value = transaction.get(b"n_spectra")
            if value is None:
                raise RuntimeError(f"LMDB has no n_spectra metadata: {path}")
            return int(value.decode())
    finally:
        environment.close()
