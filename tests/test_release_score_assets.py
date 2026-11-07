# A valid archive digest does not imply the selected sequence can be scored.
# 即使外层哈希正确，缺少任一 Score 组件也不能报告权重检查通过。
import hashlib
import sys

import pytest

from scripts import verify_release


@pytest.mark.parametrize("missing", [
    "sequence_confidence_head_calibrated.pt",
    "long_peptide_confidence_expert.pt",
])
def test_verify_rejects_missing_score_component(tmp_path, monkeypatch, missing):
    checkpoint = tmp_path / "fixture.ckpt"
    checkpoint.write_bytes(b"fixture")
    args = ["verify_release", "--checkpoint", str(checkpoint),
            "--expected-sha256", hashlib.sha256(b"fixture").hexdigest()]
    monkeypatch.setattr(sys, "argv", args)
    (tmp_path / "config").mkdir()
    (tmp_path / "config/inference.yaml").write_text("runtime: {mode: fast}")
    components = tmp_path / "components"
    components.mkdir()
    for name in ("backbone.ckpt", "r1_ranker.pt", "r2_long_expert.pt",
                 "r3_fragment_expert.pt", "length_predictor.pt", "observable_router.pt",
                 "sequence_confidence_head_calibrated.pt", "long_peptide_confidence_expert.pt"):
        if name != missing:
            (components / name).touch()
    monkeypatch.setattr(verify_release, "resolve_model_release", lambda *a, **kw: tmp_path)
    with pytest.raises(RuntimeError, match=missing.replace(".", r"\.")):
        verify_release.main()
