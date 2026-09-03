"""Versioned, sharded candidate-cache helpers."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import torch

from MNovo.ranker.token_evidence import TOKEN_EVIDENCE_FEATURE_NAMES

CACHE_FORMAT_VERSION = 4
CORE_FEATURE_NAMES = [
    "ctc_length_normalized_score",
    "score_delta_to_top1",
    "reciprocal_rank",
]
MASS_FEATURE_NAMES = [
    "isotope_aware_mass_abs_ppm_scaled",
    "mass_error_improvement_vs_top1_scaled",
]
MASS0_FEATURE_NAMES = [
    "mass0_abs_ppm_scaled",
    "mass0_improvement_vs_top1_scaled",
]
MASS0_ABS_FEATURE_NAMES = [
    "mass0_abs_ppm_scaled",
]
MASS0_DELTA_FEATURE_NAMES = [
    "mass0_improvement_vs_top1_scaled",
]
MASS_ISO_ABS_FEATURE_NAMES = [
    "isotope_aware_mass_abs_ppm_scaled",
]
MASS_ISO_DELTA_FEATURE_NAMES = [
    "mass_error_improvement_vs_top1_scaled",
]
LENGTH_FEATURE_NAMES = [
    "candidate_length_log_probability",
    "candidate_length_deviation_scaled",
    "length_log_probability_improvement_vs_top1",
]
FRAGMENT_FEATURE_NAMES = [
    "fragment_matched_ion_coverage",
    "fragment_explained_intensity",
    "fragment_ladder_continuity",
    "fragment_terminal_coverage",
    "fragment_mass_tolerance_evidence",
    "fragment_complementary_cut_coverage",
]
LONG_EXPERT_FEATURE_NAMES = [
    "candidate_length_scaled",
    "candidate_length_delta_to_top1_scaled",
]
FRAGMENT_EXPERT_FEATURE_NAMES = [
    "fragment_matched_ion_coverage",
    "fragment_explained_intensity",
    "fragment_ladder_continuity",
    "fragment_terminal_coverage",
    "fragment_mass_tolerance_evidence",
    "fragment_complementary_cut_coverage",
    "fragment_b_ion_coverage",
    "fragment_y_ion_coverage",
    "fragment_longest_b_series",
    "fragment_longest_y_series",
    "fragment_missing_cleavage_fraction",
    "fragment_prefix_suffix_balance",
]
MASS_REGIME_FEATURE_NAMES = [
    "mass0_abs_ppm_scaled",
    "mass0_improvement_vs_top1_scaled",
    "mass0_excess_over_best_scaled",
    "mass0_reciprocal_rank",
    "mass0_signed_ppm_scaled",
    "precursor_charge_scaled",
    "candidate_is_pmc",
    "candidate_within_20ppm",
    "candidate_is_mass0_best",
]
FEATURE_GROUPS = {
    "R0": CORE_FEATURE_NAMES,
    "R1": CORE_FEATURE_NAMES + MASS_FEATURE_NAMES,
    "R1_MASS0": CORE_FEATURE_NAMES + MASS0_FEATURE_NAMES,
    "R1_MASS0_ABS": CORE_FEATURE_NAMES + MASS0_ABS_FEATURE_NAMES,
    "R1_TOKEN": (
        CORE_FEATURE_NAMES
        + MASS0_ABS_FEATURE_NAMES
        + TOKEN_EVIDENCE_FEATURE_NAMES
    ),
    "R1_MASS0_DELTA": CORE_FEATURE_NAMES + MASS0_DELTA_FEATURE_NAMES,
    "R1_ISO_ABS": CORE_FEATURE_NAMES + MASS_ISO_ABS_FEATURE_NAMES,
    "R1_ISO_DELTA": CORE_FEATURE_NAMES + MASS_ISO_DELTA_FEATURE_NAMES,
    "R2": CORE_FEATURE_NAMES + MASS_FEATURE_NAMES + LENGTH_FEATURE_NAMES,
    "R3": (
        CORE_FEATURE_NAMES
        + MASS_FEATURE_NAMES
        + LENGTH_FEATURE_NAMES
        + FRAGMENT_FEATURE_NAMES
    ),
    "R4": (
        CORE_FEATURE_NAMES
        + MASS_FEATURE_NAMES
        + LENGTH_FEATURE_NAMES
        + FRAGMENT_FEATURE_NAMES
    ),
    "R2_LONG_EXPERT": (
        CORE_FEATURE_NAMES
        + MASS0_ABS_FEATURE_NAMES
        + LENGTH_FEATURE_NAMES
        + LONG_EXPERT_FEATURE_NAMES
    ),
    "R3_FRAGMENT_EXPERT": (
        CORE_FEATURE_NAMES
        + MASS0_ABS_FEATURE_NAMES
        + FRAGMENT_EXPERT_FEATURE_NAMES
    ),
    "R4_MASS_REGIME_EXPERT": CORE_FEATURE_NAMES + MASS_REGIME_FEATURE_NAMES,
}
# Kept as an import-compatible alias for the core feature group.
FEATURE_NAMES = CORE_FEATURE_NAMES


def sha256_file(path: str | Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def mnovo_confidence(beam_nll: float) -> float:
    """MNovo's original exported score, exp(-CTC NLL), in [0, 1]."""
    if not math.isfinite(beam_nll):
        return 0.0
    return float(math.exp(-min(max(float(beam_nll), 0.0), 745.0)))


def candidate_core_features(
    beam_nlls: list[float],
    lengths: list[int],
    length_alpha: float,
) -> torch.Tensor:
    if not beam_nlls:
        raise ValueError("At least one candidate is required.")
    n = len(beam_nlls)
    if len(lengths) != n:
        raise ValueError("Candidate feature inputs must have equal lengths.")
    base_scores = [
        -float(nll) / (max(int(length), 1) ** max(float(length_alpha), 1e-6))
        for nll, length in zip(beam_nlls, lengths)
    ]
    top_score = base_scores[0]
    rows = []
    for idx in range(n):
        rows.append(
            [
                base_scores[idx],
                base_scores[idx] - top_score,
                1.0 / float(idx + 1),
            ]
        )
    return torch.tensor(rows, dtype=torch.float32)


def candidate_features(
    beam_nlls: list[float],
    lengths: list[int],
    mass_ppm: list[float],
    sources: list[str],
    length_alpha: float,
) -> torch.Tensor:
    """Compatibility wrapper for pre-v3 callers; returns v3 core features."""
    if not (len(beam_nlls) == len(lengths) == len(mass_ppm) == len(sources)):
        raise ValueError("Candidate feature inputs must have equal lengths.")
    return candidate_core_features(beam_nlls, lengths, length_alpha)


def write_manifest(cache_dir: str | Path, manifest: dict) -> None:
    path = Path(cache_dir) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_manifest(cache_dir: str | Path, require_complete: bool = True) -> dict:
    path = Path(cache_dir) / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != CACHE_FORMAT_VERSION:
        raise ValueError(f"Unsupported candidate cache version: {manifest.get('format_version')}")
    if require_complete and manifest.get("status") != "complete":
        raise RuntimeError(f"Candidate cache is not complete: {cache_dir}")
    if manifest.get("feature_names") != CORE_FEATURE_NAMES:
        raise ValueError("Candidate cache feature schema does not match this code.")
    return manifest


def iter_shards(cache_dir: str | Path) -> Iterable[dict[str, torch.Tensor]]:
    manifest = load_manifest(cache_dir)
    for relative_path in manifest["shards"]:
        yield torch.load(
            Path(cache_dir) / relative_path,
            map_location="cpu",
            weights_only=True,
        )


def load_tensors(cache_dir: str | Path) -> dict[str, torch.Tensor | dict]:
    manifest = load_manifest(cache_dir)
    shard_rows = list(iter_shards(cache_dir))
    max_candidates = max(int(row["mask"].size(1)) for row in shard_rows)
    tensor_keys = [
        "core_features",
        "labels",
        "mask",
        "beam_nll",
        "mnovo_score_raw",
        "mnovo_valid",
        "dataset_index",
        "precursor",
        "spectrum_embedding",
        "truth_length",
        "candidate_length",
        "candidate_mass",
        "candidate_is_pmc",
        "token_evidence",
    ]
    candidate_keys = {
        "core_features",
        "labels",
        "mask",
        "beam_nll",
        "candidate_length",
        "candidate_mass",
        "candidate_is_pmc",
        "token_evidence",
    }
    fill_values = {"beam_nll": float("inf")}

    def normalized(row: dict[str, torch.Tensor], key: str) -> torch.Tensor:
        if key == "token_evidence" and key not in row:
            mask = row["mask"]
            value = torch.zeros(
                (*mask.shape, len(TOKEN_EVIDENCE_FEATURE_NAMES)),
                dtype=torch.float32,
                device=mask.device,
            )
        else:
            value = row[key]
        if key not in candidate_keys or int(value.size(1)) == max_candidates:
            return value
        shape = list(value.shape)
        shape[1] = max_candidates
        padded = torch.full(
            shape,
            fill_values.get(key, 0),
            dtype=value.dtype,
            device=value.device,
        )
        padded[:, : value.size(1)] = value
        return padded

    tensors = {
        key: torch.cat([normalized(row, key) for row in shard_rows], dim=0)
        for key in tensor_keys
    }
    tensors["manifest"] = manifest
    return tensors


def isotope_aware_mass_features(
    candidate_mass: torch.Tensor,
    precursor_neutral_mass: torch.Tensor,
    isotope_errors: tuple[int, ...] = (0, 1),
    isotope_mass: float = 1.003354835,
) -> torch.Tensor:
    """Return absolute isotope-aware ppm and improvement over candidate zero."""
    target = precursor_neutral_mass.float().unsqueeze(1) - 18.01
    errors = []
    for isotope in isotope_errors:
        corrected = target - float(isotope) * float(isotope_mass)
        ppm = (
            (candidate_mass.float() - corrected).abs()
            / corrected.abs().clamp_min(1e-6)
            * 1e6
        )
        errors.append(ppm)
    best_ppm = torch.stack(errors, dim=-1).amin(dim=-1)
    top_ppm = best_ppm[:, :1]
    return torch.stack(
        [
            -best_ppm.clamp_max(1000.0) / 100.0,
            (top_ppm - best_ppm).clamp(min=-1000.0, max=1000.0) / 100.0,
        ],
        dim=-1,
    )


def mass_regime_features(
    candidate_mass: torch.Tensor,
    precursor: torch.Tensor,
    candidate_is_pmc: torch.Tensor,
) -> torch.Tensor:
    """Candidate mass features for mass-inconsistent and PMC rescue failures."""
    target = precursor[:, 0].float().unsqueeze(1) - 18.01
    safe_target = target.abs().clamp_min(1e-6)
    signed0 = (candidate_mass.float() - target) / safe_target * 1e6
    abs0 = signed0.abs()
    top0 = abs0[:, :1]
    best0 = abs0.amin(dim=1, keepdim=True)
    order = abs0.argsort(dim=1)
    rank = order.argsort(dim=1).float() + 1.0
    charge = precursor[:, 1].float().unsqueeze(1).expand_as(abs0)
    return torch.stack(
        [
            -abs0.clamp_max(1000.0) / 100.0,
            (top0 - abs0).clamp(min=-1000.0, max=1000.0) / 100.0,
            -(abs0 - best0).clamp(min=0.0, max=1000.0) / 100.0,
            1.0 / rank,
            signed0.clamp(min=-1000.0, max=1000.0) / 100.0,
            charge / 10.0,
            candidate_is_pmc.float(),
            (abs0 <= 20.0).float(),
            (abs0 == best0).float(),
        ],
        dim=-1,
    )
