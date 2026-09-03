from pathlib import Path

from click.testing import CliRunner

from MNovo import backbone_cli


def test_training_outputs_are_scoped_to_run_directory(
    tmp_path: Path, monkeypatch
) -> None:
    captured = {}

    def fake_train(_train, _valid, _test, _model, config):
        captured.update(config)

    monkeypatch.setattr(backbone_cli.backbone_runner, "train", fake_train)
    run_dir = tmp_path / "run"
    config = Path(backbone_cli.__file__).with_name("config.yaml")
    result = CliRunner().invoke(
        backbone_cli.main,
        [
            "--peak_path",
            "train.mgf",
            "--peak_path_val",
            "valid.mgf",
            "--peak_path_test",
            "test.mgf",
            "--config",
            str(config),
            "--output",
            str(run_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["model_save_folder_path"] == str(run_dir / "checkpoints")
    assert captured["metrics_csv_path"] == str(run_dir / "metrics.csv")
    assert (run_dir / "train.log").is_file()
