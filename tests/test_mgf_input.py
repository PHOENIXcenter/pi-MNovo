from pathlib import Path
from types import SimpleNamespace

import pytest

from MNovo.denovo.parser2 import MgfParser
from MNovo.input import resolve_mgf_inputs
from predict import (
    PROTON_MASS,
    WATER_MASS,
    apply_config,
    output_row,
    scan_number,
    sequence_details,
)


def test_single_directory_and_case_insensitive_globs(tmp_path: Path) -> None:
    lower = tmp_path / "a.mgf"
    upper = tmp_path / "b.MGF"
    mixed = tmp_path / "nested" / "c.MgF"
    mixed.parent.mkdir()
    for path in (lower, upper, mixed):
        path.write_text("BEGIN IONS\nEND IONS\n", encoding="ascii")
    (tmp_path / "ignored.txt").write_text("", encoding="ascii")

    assert resolve_mgf_inputs(lower) == [lower.resolve()]
    assert resolve_mgf_inputs(upper) == [upper.resolve()]
    assert resolve_mgf_inputs(tmp_path) == [
        lower.resolve(),
        upper.resolve(),
        mixed.resolve(),
    ]
    assert resolve_mgf_inputs(str(tmp_path / "*.mgf")) == [
        lower.resolve(),
        upper.resolve(),
    ]
    assert resolve_mgf_inputs(str(tmp_path / "**" / "*.MGF")) == [
        lower.resolve(),
        upper.resolve(),
        mixed.resolve(),
    ]


def test_rejects_non_mgf_and_empty_matches(tmp_path: Path) -> None:
    text = tmp_path / "not_mgf.txt"
    text.write_text("", encoding="ascii")
    with pytest.raises(FileNotFoundError):
        resolve_mgf_inputs(text)
    with pytest.raises(FileNotFoundError):
        resolve_mgf_inputs(str(tmp_path / "*.mgf"))


def test_runtime_defaults_and_cli_overrides(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    config = model_dir / "config" / "inference.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        "max_charge: 8\n"
        "runtime:\n"
        "  mode: fast\n"
        "  batch_size: 128\n"
        "  n_workers: 16\n"
        "  ctc_processes: 16\n",
        encoding="ascii",
    )
    args = SimpleNamespace(
        model_dir=str(model_dir),
        config=None,
        mode=None,
        batch_size=None,
        n_workers=4,
        ctc_processes=None,
        temp_dir=None,
    )
    result = apply_config(args)
    assert result.mode == "fast"
    assert result.batch_size == 128
    assert result.n_workers == 4
    assert result.ctc_processes == 16
    assert result.max_charge == 8
    assert result.config == str(config.resolve())


def test_cli_defaults_to_automatic_device(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["pi-mnovo", "--input", "input.mgf", "--output", "output.tsv"],
    )
    from predict import parse_args

    assert parse_args().device == "auto"


@pytest.mark.parametrize("title_key", ["title", "TITLE", "TiTlE"])
def test_mgf_title_is_case_insensitive_and_preserved(
    tmp_path: Path,
    title_key: str,
) -> None:
    parser = MgfParser(tmp_path / "unused.mgf", annotationsLabel=True)
    parser.parse_spectrum(
        {
            "params": {
                "pepmass": (500.2,),
                "charge": [2],
                title_key: "sample scan=42",
                "SEQ": "PEPTIDE",
            },
            "m/z array": [100.0, 200.0],
            "intensity array": [10.0, 20.0],
        }
    )
    assert parser.titles == ["sample scan=42"]
    assert parser.annotations == ["PEPTIDE"]


def test_publication_output_row() -> None:
    residues = {
        "A": 71.037114,
        "C+57.021": 160.030649,
    }
    calc_mh, modification = sequence_details("AC+57.021", residues)
    assert calc_mh == pytest.approx(
        71.037114 + 160.030649 + WATER_MASS + PROTON_MASS
    )
    assert modification == "2,Carbamidomethyl[C];"
    record = {
        "title": "sample.42.42.2 scan=42",
        "precursor_mz": 500.2,
        "precursor_charge": 2,
    }
    row = output_row(record, 0, "AC+57.021", 0.87654, residues)
    assert row == [
        "sample.42.42.2 scan=42",
        "42",
        f"{500.2 * 2 - PROTON_MASS:.6f}",
        "2",
        "AC+57.021",
        f"{calc_mh:.6f}",
        f"{500.2 * 2 - PROTON_MASS - calc_mh:.6f}",
        "0.8765",
        "2,Carbamidomethyl[C];",
    ]


def test_scan_number_falls_back_to_one_based_index() -> None:
    assert scan_number("no scan identifier", 9) == 10


def test_scan_number_prefers_scan_over_earlier_index() -> None:
    title = (
        "Run: sample, Index: 2622, Scan: 2623, "
        "ActivationType: HCD"
    )
    assert scan_number(title, 0) == 2623


def test_modification_output_uses_named_position_format() -> None:
    residues = {
        "A": 71.037114,
        "C+57.021": 160.030649,
        "M+15.995": 147.0354,
        "+42.011": 42.010565,
    }
    _mass, modification = sequence_details(
        "+42.011AAAAAAAC+57.021AAAAAAAAAAAAAAAM+15.995",
        residues,
    )
    assert modification == (
        "0,Acetyl[AnyN-term];"
        "8,Carbamidomethyl[C];"
        "24,Oxidation[M];"
    )
