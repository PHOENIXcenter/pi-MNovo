"""LMDB-backed storage for parsed mass spectra."""

from __future__ import annotations

import logging
import os
import pickle
from collections.abc import Iterable
from pathlib import Path

import lmdb

from .spectrum_parsers import MgfParser, MzmlParser, MzxmlParser

LOGGER = logging.getLogger(__name__)


def _as_list(value):
    """Return an iterable value as a list without splitting path strings."""
    if isinstance(value, (str, Path)) or not isinstance(value, Iterable):
        return [value]
    return list(value)


class LmdbSpectrumIndex:
    """Read, create, and manage one LMDB-backed spectrum index."""

    def __init__(
        self,
        db_path,
        filenames=None,
        ms_level=2,
        valid_charge=None,
        annotated=True,
        lock=True,
    ):
        self.db_path = Path(db_path)
        self.lock = bool(lock)
        self.filenames = filenames
        self.ms_level = int(ms_level)
        self.valid_charge = valid_charge
        self.annotated = bool(annotated)
        existed = self.db_path.exists()

        LOGGER.debug("Opening spectrum index %s with lock=%s", self.db_path, self.lock)
        self._init_db()

        if existed:
            with self.env.begin() as transaction:
                self.n_spectra = int(transaction.get(b"n_spectra").decode())
                stored_ms_level = int(transaction.get(b"ms_level").decode())
            if stored_ms_level != self.ms_level:
                raise ValueError(
                    f"{self.db_path} contains MS{stored_ms_level} spectra, "
                    f"but MS{self.ms_level} was requested."
                )
        else:
            with self.env.begin(write=True) as transaction:
                transaction.put(b"ms_level", str(self.ms_level).encode())
                transaction.put(b"n_spectra", b"0")
                transaction.put(b"n_peaks", b"0")
            self.n_spectra = 0

        if filenames is not None:
            files = _as_list(filenames)
            LOGGER.info("Reading %i files...", len(files))
            for spectrum_file in files:
                self.add_file(spectrum_file)

    def write_spectra(self, parser, count: int) -> None:
        """Append all spectra produced by a parser to this index."""
        if count != len(parser.precursor_charge):
            raise ValueError("Parser spectrum count does not match its metadata.")
        with self.env.begin(write=True) as transaction:
            for index in range(count):
                record = {
                    "precursor_mz": parser.precursor_mz[index],
                    "precursor_charge": parser.precursor_charge[index],
                    "mz_array": parser.mz_arrays[index],
                    "intensity_array": parser.intensity_arrays[index],
                    "pep": parser.annotations[index],
                }
                if hasattr(parser, "titles"):
                    record["title"] = parser.titles[index]
                elif hasattr(parser, "scan_id"):
                    record["title"] = str(parser.scan_id[index])
                transaction.put(
                    str(self.n_spectra).encode(),
                    pickle.dumps(record),
                )
                self.n_spectra += 1
            transaction.put(b"n_spectra", str(self.n_spectra).encode())

    def __getitem__(self, index):
        self._ensure_db_for_current_process()
        with self.env.begin() as transaction:
            payload = transaction.get(str(index).encode())
        if payload is None:
            raise IndexError(index)
        record = pickle.loads(payload)
        return (
            record["mz_array"],
            record["intensity_array"],
            record["precursor_mz"],
            record["precursor_charge"],
            record["pep"],
        )

    def __len__(self):
        return self.n_spectra

    def add_file(self, spectrum_file) -> None:
        parser = self._get_parser(Path(spectrum_file))
        parser.read()
        self.write_spectra(parser, parser.n_spectra)

    def _init_db(self) -> None:
        self.env = lmdb.open(
            str(self.db_path),
            map_size=4_099_511_627_776,
            subdir=False,
            readonly=not self.lock,
            lock=self.lock,
        )
        self._env_pid = os.getpid()

    def _ensure_db_for_current_process(self) -> None:
        """Reopen LMDB after DataLoader forks worker processes."""
        if getattr(self, "_env_pid", None) == os.getpid():
            return
        try:
            self.env.close()
        except Exception:
            pass
        self._init_db()

    def _get_parser(self, spectrum_file: Path):
        options = {
            "ms_level": self.ms_level,
            "valid_charge": self.valid_charge,
            "annotationsLabel": self.annotated,
        }
        parsers = {
            ".mzml": MzmlParser,
            ".mzxml": MzxmlParser,
            ".mgf": MgfParser,
        }
        try:
            parser_class = parsers[spectrum_file.suffix.lower()]
        except KeyError as error:
            raise ValueError(
                "Only mzML, mzXML, and MGF files are supported."
            ) from error
        return parser_class(spectrum_file, **options)
