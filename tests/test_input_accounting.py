import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from MNovo.input import materialize_mgf_lmdb, LmdbTitleLookup
from MNovo.input_audit import InputAudit


def block(title, charge="2+", mass="500.2"):
    return ("BEGIN IONS\nTITLE=" + title + "\n"
            + (f"CHARGE={charge}\n" if charge is not None else "")
            + (f"PEPMASS={mass}\n" if mass is not None else "")
            + "100 10\n200 20\nEND IONS\n")


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_bad_blocks_do_not_hide_neighbouring_spectra(tmp_path):
    first = tmp_path / "a.mgf"
    second = tmp_path / "b.mgf"
    first.write_text(block("ok") + block("missing", None) + block("high", "11+")
                     + block("mass", mass=None) + block("broken", "nonsense")
                     + block("after"))
    second.write_text(block("other file"))
    with InputAudit(tmp_path / "pred.tsv") as audit:
        count = materialize_mgf_lmdb([first, second], tmp_path / "input.lmdb", 10, audit=audit)
    assert count == 3
    rows = read_rows(audit.paths["input_audit"])
    rejects = read_rows(audit.paths["rejected"])
    assert len(rows) == 7 and len(rejects) == 4
    assert rows[-1]["spectrum_number"] == "1"
    assert rows[-1]["source_file"] == str(second.resolve())
    assert rows[-1]["dataset_index"] == "2"
    assert rejects[1]["observed"] == "11+"
    assert "1..10" in rejects[1]["supported"]
    assert "11+" in rejects[1]["reason"]
    assert rejects[0]["observed"] == "missing"
    assert rejects[2]["field"] == "PEPMASS"
    with LmdbTitleLookup(tmp_path / "input.lmdb") as lookup:
        assert [lookup.title(i) for i in range(count)] == ["ok", "after", "other file"]
    result = audit.save("complete_with_rejections", count)
    assert result["count_verified"]
    assert result["original_input"] == result["predictions"] + result["rejected"]
    with pytest.raises(RuntimeError, match="accounting mismatch"):
        audit.save("complete", 2)


def test_all_rejected_publishes_zero_predictions_without_model(tmp_path, monkeypatch):
    from MNovo import cli
    source = tmp_path / "bad.mgf"
    source.write_text(block("missing", None) + block("too high", "12+"))
    output = tmp_path / "pred.tsv"
    with InputAudit(output) as audit:
        count = materialize_mgf_lmdb([source], tmp_path / "input.lmdb", 10, audit=audit)
    monkeypatch.setattr(cli, "MNovoRuntime", lambda *a, **k: pytest.fail("loaded model"))
    args = SimpleNamespace(indices=None, max_samples=0, task="denovo", output=str(output))
    assert cli.run(args, str(tmp_path / "input.lmdb"), [source], count) == 0
    assert read_rows(output) == []
    assert audit.save("no_predictions", 0)["count_verified"]


def test_incomplete_block_and_global_charge(tmp_path):
    source = tmp_path / "input.mgf"
    source.write_text("CHARGE=3+\n" + block("global", None)
                      + block("unfinished").replace("END IONS\n", "")
                      + block("after", "2+"))
    with InputAudit(tmp_path / "pred.tsv") as audit:
        count = materialize_mgf_lmdb([source], tmp_path / "input.lmdb", 10, audit=audit)
    assert count == 2 and audit.rejected == 1
    assert read_rows(audit.paths["rejected"])[0]["observed"] == "missing END IONS"
    result = audit.save("complete", 1, not_selected=1)
    assert result["count_verified"]
    assert result["not_selected"] == 1
    assert json.loads(audit.paths["summary"].read_text())["predictions"] == 1
