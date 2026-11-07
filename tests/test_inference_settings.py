import json
from types import SimpleNamespace

import pytest
import yaml

from MNovo.cli import apply_config
from MNovo.cct.cct_pipeline import runtime_identity
from MNovo.inference.config import resolve_settings
from MNovo.inference.runtime import RuntimeOptions


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
