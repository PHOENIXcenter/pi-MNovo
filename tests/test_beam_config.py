from argparse import Namespace

import pytest

from MNovo.cli import apply_config


@pytest.mark.parametrize("yaml_width,cli_width,expected", [
    (None, None, None), (20, None, None), (10, None, 10), (10, 20, None),
    (10, 8, 8), (0, None, ValueError), (True, None, ValueError),
])
def test_effective_width(tmp_path, yaml_width, cli_width, expected):
    cfg = tmp_path / "config"
    cfg.mkdir()
    (cfg / "inference.yaml").write_text("n_beams: 20\n")
    override = tmp_path / "user.yaml"
    override.write_text("runtime: {}" if yaml_width is None else
                        f"runtime: {{beam_width: {yaml_width}}}")
    args = Namespace(config=str(override), model_dir=str(tmp_path), mode=None,
                     beam_width=cli_width, batch_size=None, n_workers=None,
                     ctc_processes=None, temp_dir=None)
    if expected is ValueError:
        with pytest.raises(ValueError, match="positive integer"):
            apply_config(args)
    else:
        assert apply_config(args).beam_width == expected
