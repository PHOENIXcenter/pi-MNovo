"""Candidate-specific b/y fragment-ion evidence."""

from __future__ import annotations

import re

import numpy as np


PROTON = 1.007276466812
WATER = 18.010564684


def tokenize_peptide(peptide: str) -> list[str]:
    peptide = str(peptide).replace("[", "").replace("]", "").replace("I", "L")
    return [token for token in re.split(r"(?<=.)(?=[A-Z])", peptide) if token]


def theoretical_ions(
    peptide: str,
    residue_masses: dict[str, float],
    max_charge: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tokens = tokenize_peptide(peptide)
    try:
        masses = np.asarray(
            [float(residue_masses[token]) for token in tokens], dtype=np.float64
        )
    except KeyError:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int8),
        )
    if masses.size <= 1:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int8),
        )
    prefix = np.cumsum(masses)[:-1]
    suffix = masses.sum() - prefix
    ion_mz, cut_ids, series_ids = [], [], []
    for charge in range(1, max(int(max_charge), 1) + 1):
        ion_mz.extend(((prefix + charge * PROTON) / charge).tolist())
        cut_ids.extend(range(1, int(masses.size)))
        series_ids.extend([0] * (int(masses.size) - 1))
        ion_mz.extend(((suffix + WATER + charge * PROTON) / charge).tolist())
        cut_ids.extend(range(1, int(masses.size)))
        series_ids.extend([1] * (int(masses.size) - 1))
    return (
        np.asarray(ion_mz, dtype=np.float64),
        np.asarray(cut_ids, dtype=np.int32),
        np.asarray(series_ids, dtype=np.int8),
    )


def fragment_features(
    peptide: str,
    spectrum_mz: np.ndarray,
    spectrum_intensity: np.ndarray,
    precursor_charge: int,
    residue_masses: dict[str, float],
    tolerance_da: float = 0.5,
    max_fragment_charge: int = 2,
) -> np.ndarray:
    valid = (spectrum_mz > 0) & (spectrum_intensity > 0)
    mz = np.asarray(spectrum_mz[valid], dtype=np.float64)
    intensity = np.asarray(spectrum_intensity[valid], dtype=np.float64)
    if mz.size == 0:
        return np.zeros(12, dtype=np.float32)
    order = np.argsort(mz)
    mz, intensity = mz[order], intensity[order]
    max_z = min(max(int(precursor_charge), 1), int(max_fragment_charge))
    ions, cuts, series = theoretical_ions(peptide, residue_masses, max_z)
    if ions.size == 0:
        return np.zeros(12, dtype=np.float32)

    right = np.searchsorted(mz, ions, side="left")
    left = np.maximum(right - 1, 0)
    right = np.minimum(right, mz.size - 1)
    left_error = np.abs(mz[left] - ions)
    right_error = np.abs(mz[right] - ions)
    choose_right = right_error < left_error
    peak_index = np.where(choose_right, right, left)
    error = np.where(choose_right, right_error, left_error)
    matched = error <= float(tolerance_da)
    if not matched.any():
        return np.zeros(12, dtype=np.float32)

    matched_peak = peak_index[matched]
    matched_intensity = intensity[matched_peak]
    matched_error = error[matched]
    matched_cuts = cuts[matched]
    matched_series = series[matched]
    unique_intensity = intensity[np.unique(matched_peak)].sum()
    n_cuts = max(int(cuts.max()), 1)
    cut_any = np.zeros(n_cuts + 1, dtype=bool)
    cut_b = np.zeros(n_cuts + 1, dtype=bool)
    cut_y = np.zeros(n_cuts + 1, dtype=bool)
    cut_any[matched_cuts] = True
    cut_b[matched_cuts[matched_series == 0]] = True
    cut_y[matched_cuts[matched_series == 1]] = True
    longest = run = 0
    for cut in range(1, n_cuts + 1):
        if cut_any[cut]:
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    terminal = (float(cut_any[1]) + float(cut_any[n_cuts])) / 2.0
    longest_b = longest_y = run_b = run_y = 0
    for cut in range(1, n_cuts + 1):
        run_b = run_b + 1 if cut_b[cut] else 0
        run_y = run_y + 1 if cut_y[cut] else 0
        longest_b = max(longest_b, run_b)
        longest_y = max(longest_y, run_y)
    b_coverage = float(cut_b.sum()) / n_cuts
    y_coverage = float(cut_y.sum()) / n_cuts
    missing_cleavage = 1.0 - float(cut_any.sum()) / n_cuts
    prefix_suffix_balance = 1.0 - abs(b_coverage - y_coverage)
    intensity_scale = max(float(intensity.max()), 1e-8)
    weighted = (
        (matched_intensity / intensity_scale)
        * np.maximum(0.0, 1.0 - matched_error / max(float(tolerance_da), 1e-8))
    ).sum() / max(int(ions.size), 1)
    return np.asarray(
        [
            float(matched.sum()) / max(int(ions.size), 1),
            float(unique_intensity) / max(float(intensity.sum()), 1e-8),
            float(longest) / n_cuts,
            terminal,
            float(weighted),
            float((cut_b & cut_y).sum()) / n_cuts,
            b_coverage,
            y_coverage,
            float(longest_b) / n_cuts,
            float(longest_y) / n_cuts,
            missing_cleavage,
            prefix_suffix_balance,
        ],
        dtype=np.float32,
    )
