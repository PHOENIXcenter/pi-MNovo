# Validate attention and repair only explicitly supported feature failures.
# 检查 attention，只修复策略明确允许的特征异常。
"""Isolated CCT policy: repair only a verified constant-input Spearman NaN."""
import numpy as np

POLICY_ID = 'constant_spearman_zero_v1'


class FeaturePolicyError(ValueError):
    pass


def validate_attention(matrix, mz):
    matrix = np.asarray(matrix)
    mz = np.asarray(mz)
    if matrix.ndim != 2 or mz.ndim != 1 or matrix.shape[1] != len(mz):
        raise FeaturePolicyError('attention_dimension_mismatch')
    if matrix.shape[0] < 2 or matrix.shape[1] < 10:
        raise FeaturePolicyError('insufficient_attention_dimensions')
    if not np.isfinite(matrix).all() or not np.isfinite(mz).all():
        raise FeaturePolicyError('attention_or_mz_nonfinite')
    if np.any(matrix < 0) or np.any(matrix.sum(axis=1) <= 0):
        raise FeaturePolicyError('negative_or_zero_attention_row')


def repair_features(features, matrix, mz, expected_columns):
    validate_attention(matrix, mz)
    if tuple(features) != tuple(expected_columns):
        raise FeaturePolicyError('feature_schema_mismatch')
    result = {k: float(v) for k, v in features.items()}
    nonfinite = [k for k, v in result.items() if not np.isfinite(v)]
    flags = {'spearman_degenerate': False, 'nonfinite_replaced': [], 'policy': POLICY_ID}
    if not nonfinite:
        return result, flags
    name = 'fragment_alignment_consistency'
    if nonfinite != [name] or not np.isnan(result[name]):
        raise FeaturePolicyError('other_or_multiple_nonfinite_features:' + ','.join(nonfinite))
    peak_mz = np.asarray(mz)[np.argmax(matrix, axis=1)]
    if not np.all(peak_mz == peak_mz[0]):
        raise FeaturePolicyError('spearman_nan_without_constant_input')
    result[name] = 0.0
    flags['spearman_degenerate'] = True
    flags['nonfinite_replaced'] = [name]
    return result, flags


def self_test():
    keys = ('other', 'fragment_alignment_consistency')
    matrix = np.ones((4, 10)); mz = np.arange(10.)
    f, flags = repair_features(dict(other=2., fragment_alignment_consistency=np.nan), matrix, mz, keys)
    assert f == dict(other=2., fragment_alignment_consistency=0.) and flags['spearman_degenerate']
    original = dict(other=2., fragment_alignment_consistency=.4)
    assert repair_features(original, matrix, mz, keys)[0] == original
    cases = [
        (dict(other=np.nan, fragment_alignment_consistency=np.nan), matrix, mz),
        (dict(other=2., fragment_alignment_consistency=np.inf), matrix, mz),
        (dict(other=2., fragment_alignment_consistency=np.nan), np.zeros((4,10)), mz),
        (dict(other=2., fragment_alignment_consistency=np.nan), np.full((4,10),np.nan), mz),
        (dict(other=2., fragment_alignment_consistency=np.nan), matrix, mz[:5]),
        (dict(other=2., fragment_alignment_consistency=np.nan), np.eye(10)[:4], mz),
        (dict(other=2.), matrix, mz),
    ]
    for features, attention, masses in cases:
        try:
            repair_features(features, attention, masses, keys)
        except FeaturePolicyError:
            pass
        else:
            raise AssertionError('An unsafe feature vector was accepted')
    return {'status':'PASS','repair_cases':1,'unchanged_cases':1,'rejection_cases':len(cases)}

if __name__ == '__main__':
    print(self_test())
