# Test occurrence alignment and both views of the same spectrum.
# 检查重复残基的对齐以及同一谱图的两种视图。
import numpy as np
import pytest
import torch
from types import SimpleNamespace

from MNovo.cct.cct_adapter import (
    AttentionCapture, occurrence_frames, paired_attention, paired_reference_features,
)


def test_attention_capture_preserves_decoder_output():
    layer = torch.nn.TransformerDecoderLayer(8, 2, dropout=0, batch_first=True).eval()
    target = torch.randn(2, 4, 8)
    memory = torch.randn(2, 6, 8)
    with torch.inference_mode():
        native = layer(target, memory)
        with AttentionCapture([layer]) as capture:
            exported = layer(target, memory)
    assert torch.equal(native, exported)
    assert capture.attention.shape == (2, 4, 6)
    assert torch.isfinite(capture.attention).all()


def test_occurrence_frames_keep_repeated_residues_separate():
    probabilities = np.array([
        [.05, .90, .05],
        [.90, .05, .05],
        [.05, .90, .05],
        [.90, .05, .05],
    ])
    frames = occurrence_frames(np.log(probabilities), [1, 1], blank=0)
    assert frames == [0, 2]
    target, counterpart = paired_attention(np.arange(20).reshape(4, 5), frames)
    np.testing.assert_array_equal(target, np.arange(20).reshape(4, 5)[[0]])
    np.testing.assert_array_equal(counterpart, np.arange(20).reshape(4, 5)[[2]])


def test_occurrence_frames_fail_on_impossible_alignment():
    with pytest.raises(ValueError, match="CTC alignment"):
        occurrence_frames(np.log(np.full((2, 3), 1 / 3)), [1, 1], blank=0)


def test_pair_rejects_single_residue():
    with pytest.raises(ValueError, match="two residues"):
        paired_attention(np.ones((4, 5)), [1])


def test_reference_feature_pair_uses_same_peptide_and_encoder_columns():
    from MNovo.cct import cct_reference_features as reference

    peptide = "PEPTIDE"
    reverse = peptide[::-1]
    tokens = list(range(1, len(peptide) + 1))
    log_probs = np.full((40, 8), np.log(0.01))
    log_probs[:, 0] = np.log(0.9)
    for index, token in enumerate(tokens):
        frame = 2 + index * 4
        log_probs[frame, token] = np.log(0.9)
        log_probs[frame, 0] = np.log(0.01)
    theoretical = reference.get_theo_mz(peptide).reshape(-1)
    peaks = np.c_[theoretical[:12], np.linspace(0.1, 1.0, 12)]
    precursor_mz = reference.PeptideMass().mass(peptide, 2)
    rng = np.random.default_rng(454)
    attention = rng.random((40, 13))
    attention /= attention.sum(axis=1, keepdims=True)
    target, counterpart = paired_reference_features(
        attention, log_probs, tokens, peptide, peaks,
        np.array([precursor_mz * 2, 2, precursor_mz]),
        blank=0, symbols=dict(zip(tokens, reverse)),
        residues=reference.PeptideMass.canonical, reference=reference,
    )
    assert len(target) == len(counterpart) == 33
    assert target.keys() == counterpart.keys()
    assert target != counterpart


def test_exporter_keeps_two_views_and_reference_label(tmp_path):
    import pandas as pd
    from MNovo.cct import cct_reference_features as reference
    from MNovo.cct.cct_runtime import CCTFeatureExporter

    peptide = "PEPTIDE"
    tokens = list(range(1, len(peptide) + 1))
    log_probs = np.full((40, 8), np.log(0.01))
    log_probs[:, 0] = np.log(0.9)
    for index, token in enumerate(tokens):
        frame = 2 + index * 4
        log_probs[frame, token] = np.log(0.9)
        log_probs[frame, 0] = np.log(0.01)
    peaks = np.c_[reference.get_theo_mz(peptide).reshape(-1)[:12],
                  np.linspace(.1, 1, 12)]
    precursor_mz = reference.PeptideMass().mass(peptide, 2)
    rng = np.random.default_rng(454)
    attention = rng.random((1, 40, 13))
    attention /= attention.sum(axis=2, keepdims=True)
    decoder = SimpleNamespace(
        reverse=True, get_blank_idx=lambda: 0,
        _idx2aa=dict(zip(tokens, peptide[::-1])),
    )
    output = tmp_path / "views.csv"
    exporter = CCTFeatureExporter(decoder, reference.PeptideMass.canonical, output)
    prediction = SimpleNamespace(dataset_index=5, peptide=peptide, route="r1", status="complete")
    exporter.collect(
        [prediction], [{"tokens": [tokens]}], [0], [0], attention,
        log_probs[None], peaks[None],
        np.array([[precursor_mz * 2, 2, precursor_mz]]), np.array([peptide]),
    )
    exporter.close()
    rows = pd.read_csv(output)
    assert exporter.complete_pairs == exporter.labelled_pairs == 1
    assert len(rows) == 2 and set(rows.is_decoy) == {0, 1}
    assert rows.denovo_label.eq(1).all() and rows.valid.all()
    assert rows.filter(regex="^f_").shape[1] == 33
