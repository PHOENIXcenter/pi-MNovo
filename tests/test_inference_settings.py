import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from MNovo.cli import apply_config
from MNovo.cct.cct_pipeline import runtime_identity
from MNovo.inference.config import resolve_settings
from MNovo.inference.runtime import RuntimeOptions


@pytest.mark.parametrize("task", ["denovo", "eval", "train"])
def test_optional_output_under_code_root(tmp_path, monkeypatch, task):
    from MNovo import cli
    monkeypatch.setattr(cli, "__file__", str(tmp_path / "MNovo/cli.py"))
    cfg = tmp_path / "config.yaml"
    cfg.write_text("runtime: {model_dir: weights.ckpt}")
    argv = ["pi-mnovo", "--model", task, "--config", str(cfg), "--input", "sample.mgf"]
    monkeypatch.setattr("sys.argv", argv + ["--output", "custom"])
    assert cli.parse_args().output == "custom"
    assert not (tmp_path / "outputs").exists()
    monkeypatch.setattr("sys.argv", argv)
    first, second = (Path(cli.parse_args().output) for _ in range(2))
    assert first != second
    run = first if task == "train" else first.parent
    assert run.parent == tmp_path / "outputs" and run.is_dir()
    assert task == "train" or first.name == "predictions.tsv"


def test_public_cli_requires_config_and_rejects_removed_flags(monkeypatch):
    from MNovo.cli import parse_args
    for extra in ([], ["--config", "unused.yaml", "--beam-width", "10"]):
        monkeypatch.setattr("sys.argv", ["pi-mnovo", "--model", "denovo", "--input", "sample.mgf",
                                       "--output", "pred.tsv", *extra])
        with pytest.raises(SystemExit):
            parse_args()


def test_yaml_beam_width_and_vocabulary_order(tmp_path):
    frozen = tmp_path / "config/inference.yaml"
    frozen.parent.mkdir()
    frozen.write_text("n_beams: 20\nresidues: {I: 113.084064, L: 113.084064}")
    cfg = tmp_path / "user.yaml"
    args = lambda: SimpleNamespace(model_dir=str(tmp_path), config=str(cfg), mode=None,
                                   batch_size=None, n_workers=None, ctc_processes=None, temp_dir=None)
    cfg.write_text("n_beams: 10")
    assert apply_config(args()).beam_width == 10
    cfg.write_text("residues: {L: 113.084064, I: 113.084064}")
    with pytest.raises(ValueError, match="order"):
        apply_config(args())


def test_training_config_dispatch_without_training(tmp_path, monkeypatch):
    from MNovo import cli
    cfg = tmp_path / "train.yaml"
    cfg.write_text("runtime: {validation_input: valid.mgf, checkpoint: init.ckpt}")
    monkeypatch.setattr("sys.argv", ["pi-mnovo", "--model", "train", "--config", str(cfg),
                                    "--input", "train.mgf", "--output", "out"])
    calls = []
    monkeypatch.setattr(cli.subprocess, "run", lambda command, check: calls.append(command))
    cli.main()
    command, = calls
    assert command[command.index("--peak_path_val") + 1] == "valid.mgf"
    assert command[command.index("--model") + 1] == "init.ckpt"
    assert command[command.index("--config") + 1] == str(cfg.resolve())


def test_lmdb_input_via_config(tmp_path, monkeypatch):
    from MNovo.cli import parse_args
    cfg = tmp_path / "eval.yaml"
    cfg.write_text("runtime: {model_dir: weights.ckpt, input_format: lmdb}")
    monkeypatch.setattr("sys.argv", ["pi-mnovo", "--model", "eval", "--config", str(cfg),
                                    "--input", "test.lmdb", "--output", "out.tsv"])
    args = parse_args()
    assert args.lmdb == "test.lmdb" and args.input is None


@pytest.mark.parametrize("overrides", [
    {"mass_control_tol": 0}, {"mass_control_tol": float("nan")},
    {"PMC_enable": "false"}, {"n_peaks": True}, {"n_peaks": -1},
    {"min_mz": 6500}, {"max_mz": 0}, {"min_intensity": 1.1},
    {"remove_precursor_tol": -1}, {"residues": {}},
])
def test_invalid_settings(overrides):
    with pytest.raises(ValueError):
        resolve_settings({"min_mz": 1, "max_mz": 6500}, overrides)


def test_effective_parameters_bind_cct_identity(tmp_path):
    base = {"PMC_enable": True, "mass_control_tol": 0.1, "n_peaks": 800}
    cfg = tmp_path / "config/inference.yaml"
    cfg.parent.mkdir()
    cfg.write_text(yaml.safe_dump(base))
    (tmp_path / "verified.json").write_text(json.dumps({"manifest": {"components": {}}}))
    external = tmp_path / "custom.yaml"
    def resolve(values):
        external.write_text(yaml.safe_dump(values))
        return apply_config(SimpleNamespace(config=str(external), model_dir=str(tmp_path),
                            mode=None, batch_size=None, n_workers=None,
                            ctc_processes=None, temp_dir=None))
    def identity(args):
        options = RuntimeOptions(inference_overrides=args.inference_overrides,
                                 candidate_mode=args.candidate_mode)
        return runtime_identity(tmp_path, cfg, options, "cpu")
    original = identity(resolve({}))
    assert original == identity(resolve(base))
    assert "inference_overrides" not in original["options"]
    custom = resolve({"PMC_enable": False, "mass_control_tol": 0.05, "n_peaks": 64})
    assert custom.candidate_mode == "beam-only"
    changed = identity(custom)
    assert changed["inference_config"] == custom.inference_overrides
    assert changed != original
