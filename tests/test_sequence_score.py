# Check sequence confidence separately from CCT filtering.
# 单独验证序列置信评分，不把它当作 CCT 过滤结果。
import ast
from pathlib import Path

import torch

from MNovo.sequence_score import ctc_viterbi_features, local_summaries


def test_viterbi_features_finite_and_length_mask():
    torch.manual_seed(3407)
    probabilities = torch.randn(2, 12, 6).log_softmax(-1)
    features, mask = ctc_viterbi_features(probabilities, [[1, 2, 2], [3, 4]], 0)
    assert features.shape == (2, 3, 6)
    assert mask.sum(1).tolist() == [3, 2]
    assert torch.isfinite(features).all()
    assert local_summaries(features, mask).shape == (2, 24)


def test_score_module_does_not_filter_or_load_fdr_thresholds():
    from MNovo import sequence_score
    source = Path(sequence_score.__file__).read_text()
    assert 'pep_qvalues' not in source
    assert 'pep_thresholds' not in source
    assert 'fdr_runtime.json' not in source


def test_saved_scoring_has_no_decoding_or_selection_call():
    from MNovo import runtime
    tree = ast.parse(Path(runtime.__file__).read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and n.name == 'score_saved_batch')
    calls = {n.func.attr for n in ast.walk(method)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert not calls & {'decode_all', '_rank', '_observable_select',
                        '_observable_select_with_evidence', '_rows_from_decode'}
