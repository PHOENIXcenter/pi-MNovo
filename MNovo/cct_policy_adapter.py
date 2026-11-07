# Apply the feature repair policy around the reference feature extractor.
# 在参考特征提取器外应用修复规则。
"""Opt-in adapter; no monkey-patching or changes to frozen code."""
from MNovo.cct_feature_policy import validate_attention, repair_features


def paired_reference_features_with_policy(*args, reference, **kwargs):
    """Return the 33-column feature pair and separate per-side audit flags."""
    from MNovo.cct_adapter import paired_reference_features, FEATURE_COLUMNS
    flags = []

    class ReferenceProxy:
        def __getattr__(self, name):
            return getattr(reference, name)

        def compute_psm_features_with_attention(self, evaluated, matrix, theory, mz, intensity):
            validate_attention(matrix, mz)
            original = reference.compute_psm_features_with_attention(evaluated, matrix, theory, mz, intensity)
            repaired, audit = repair_features(original, matrix, mz, FEATURE_COLUMNS)
            flags.append(audit)
            return repaired

    pair = paired_reference_features(*args, reference=ReferenceProxy(), **kwargs)
    assert len(flags) == 2
    return pair, flags
