# Pair attention views for the same final peptide using its aligned token occurrences.
# 根据 token 出现位置对齐，为同一最终肽段构造两种 attention 视图。
"""Align predicted CTC tokens and construct paired attention views for CCT."""

from __future__ import annotations

import re
import numpy as np
import torch


FEATURE_COLUMNS = (
    "aa_conf_mean", "aa_conf_var", "aa_conf_min", "aa_conf_max",
    "num_low_conf", "num_high_conf", "global_qc_score", "fragment_coverage",
    "explained_intensity", "precursor_error_ppm", "anchor_ratio",
    "aa_entropy_mean", "aa_entropy_std", "aa_entropy_min", "aa_entropy_max",
    "aa_concentration_mean", "aa_concentration_std", "aa_concentration_min", "aa_concentration_max",
    "aa_offset_mean", "aa_offset_std", "aa_offset_min", "aa_offset_max",
    "aa_center_drift_mean", "aa_center_drift_std", "aa_center_drift_min", "aa_center_drift_max",
    "aa_span_mean", "aa_span_std", "aa_span_min", "aa_span_max",
    "avg_attention_similarity", "fragment_alignment_consistency",
)


class AttentionCapture:
    """Recompute only frozen cross-attention weights without changing logits."""

    def __init__(self, layers):
        self.layers = list(layers)
        self.handles = []
        self.attention = None

    def _collect(self, module, args, kwargs, output):
        # Re-evaluate only this cross-attention operation for weights; the native
        # output remains untouched, so sequence logits follow the original path.
        weights = module.forward(
            *args, **{**kwargs, "need_weights": True, "average_attn_weights": False}
        )[1]
        if weights is None or weights.ndim != 4:
            raise RuntimeError("Decoder did not expose per-head cross-attention")
        reduced = weights.detach().float().amax(dim=1)
        self.attention = (
            reduced if self.attention is None else torch.maximum(self.attention, reduced)
        )

    def __enter__(self):
        self.attention = None
        for layer in self.layers:
            module = layer.multihead_attn
            self.handles.append(
                module.register_forward_hook(self._collect, with_kwargs=True)
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def occurrence_frames(log_probs: np.ndarray, tokens: list[int], blank: int) -> list[int]:
    """Pick one representative frame per occurrence along the best valid CTC path."""

    log_probs = np.asarray(log_probs, dtype=np.float64)
    if log_probs.ndim != 2 or not np.isfinite(log_probs).all():
        raise ValueError("CTC alignment needs finite time-by-vocabulary log probabilities")
    steps, vocabulary = log_probs.shape
    if not tokens or not 0 <= blank < vocabulary:
        raise ValueError("CTC alignment needs tokens and a valid blank index")
    if any(not 0 <= token < vocabulary or token == blank for token in tokens):
        raise ValueError("CTC alignment contains an invalid token")
    if len(tokens) + sum(a == b for a, b in zip(tokens, tokens[1:])) > steps:
        raise ValueError("CTC alignment is impossible in the decoder frame grid")

    extended = [blank]
    for token in tokens:
        extended.extend((token, blank))
    states = len(extended)
    scores = np.full(states, -np.inf)
    scores[0] = log_probs[0, blank]
    scores[1] = log_probs[0, tokens[0]]
    previous = np.zeros((steps, states), dtype=np.int32)
    for frame in range(1, steps):
        next_scores = np.full(states, -np.inf)
        for state, token in enumerate(extended):
            sources = [state]
            if state:
                sources.append(state - 1)
            if state > 1 and state % 2 and token != extended[state - 2]:
                sources.append(state - 2)
            source = max(sources, key=lambda index: scores[index])
            next_scores[state] = scores[source] + log_probs[frame, token]
            previous[frame, state] = source
        scores = next_scores
    terminal = max((states - 2, states - 1), key=lambda state: scores[state])
    if not np.isfinite(scores[terminal]):
        raise ValueError("CTC alignment has no terminal path")
    path = np.empty(steps, dtype=np.int32)
    path[-1] = terminal
    for frame in range(steps - 1, 0, -1):
        path[frame - 1] = previous[frame, path[frame]]
    representatives = []
    for occurrence, token in enumerate(tokens):
        aligned = np.flatnonzero(path == 2 * occurrence + 1)
        if not len(aligned):
            raise ValueError("CTC alignment omitted a predicted token occurrence")
        representatives.append(int(aligned[np.argmax(log_probs[aligned, token])]))
    return representatives


def paired_attention(attention: np.ndarray, frames: list[int]):
    """Return equal-length target and +1 occurrence-offset counterpart views."""

    attention = np.asarray(attention)
    if attention.ndim != 2 or not np.isfinite(attention).all():
        raise ValueError("Attention must be a finite frame-by-spectrum matrix")
    if len(frames) < 2:
        raise ValueError("CCT pairing needs at least two residues")
    if any(frame < 0 or frame >= len(attention) for frame in frames):
        raise ValueError("CTC frame lies outside the attention grid")
    if any(a >= b for a, b in zip(frames, frames[1:])):
        raise ValueError("CTC occurrence frames must increase strictly")
    # Shift by one aligned residue occurrence, not by one raw decoder frame.
    # 对照沿已对齐的残基移动一个位置，不是将原始时间步简单加一。
    return attention[frames[:-1]], attention[frames[1:]]


def paired_reference_features(
    attention: np.ndarray,
    log_probs: np.ndarray,
    tokens: list[int],
    peptide: str,
    spectrum: np.ndarray,
    precursor: np.ndarray,
    *,
    blank: int,
    symbols: dict[int, str],
    residues: dict[str, float],
    reference,
) -> tuple[dict[str, float], dict[str, float]]:
    """Extract the ordered CCT features for both views of the same peptide."""

    frames = occurrence_frames(log_probs, tokens, blank)
    residue_symbols = [symbols[token] for token in tokens if symbols[token] != "$"]
    peptide_frames = [frame for token, frame in zip(tokens, frames) if symbols[token] != "$"]
    if "".join(reversed(residue_symbols)) != peptide or len(peptide_frames) != len(
        re.split(r"(?<=.)(?=[A-Z])", peptide)
    ):
        raise ValueError("Predicted tokens do not match the final Top-1 peptide")
    target, counterpart = paired_attention(attention, peptide_frames)
    spectrum = np.asarray(spectrum)
    precursor = np.asarray(precursor)
    if spectrum.ndim != 2 or spectrum.shape[1] != 2 or len(precursor) < 3:
        raise ValueError("Invalid spectrum or precursor shape")
    used = np.flatnonzero(spectrum[:, 0] != 0)
    # Both views use the same real peaks; account for the encoder's leading position.
    # 两种视图使用相同的非填充峰；映射到编码器列时需计入首位偏移。
    columns = used + 1
    if len(columns) < 10 or columns[-1] >= attention.shape[1]:
        raise ValueError("Fewer than ten encoder spectrum positions are available")
    mz = spectrum[used, 0]
    intensity = spectrum[used, 1]
    if not np.isfinite(mz).all() or not np.isfinite(intensity).all():
        raise ValueError("Nonfinite spectrum peaks")
    reference._install_ptm242_masses(residues)
    theory = reference.get_theo_mz(peptide)
    labels = reference.get_aby_labels(peptide, mz, theory)
    masses = reference.PeptideMass(residues=residues)
    result = []
    for matrix in (target[:, columns], counterpart[:, columns]):
        evaluated = reference.evaluate_peptide_sequence(
            matrix, labels, peptide, mz, intensity, masses=masses,
            precursor_mass=float(precursor[2]), precursor_charge=int(precursor[1]),
            top_k=10,
        )
        features = reference.compute_psm_features_with_attention(
            evaluated, matrix, theory.reshape(-1), mz, intensity
        )
        if tuple(features) != FEATURE_COLUMNS or not np.isfinite(list(features.values())).all():
            raise ValueError("The reference 33-feature vector is incomplete or nonfinite")
        result.append({key: float(value) for key, value in features.items()})
    return result[0], result[1]
