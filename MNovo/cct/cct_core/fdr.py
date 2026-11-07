# Compute discovery curves and calibrate gamma without splitting tied scores.
# 计算发现曲线并校准 γ；阈值不能切开同分组。
"""Counterpart counts, complete score ties and reference-only gamma calibration.

PSM and peptide models calibrate separately. Application recounts pooled
Target/Counterpart scores; router scores are not FDR probabilities.
"""

from __future__ import annotations


import numpy as np
import pandas as pd



class FDRContractError(ValueError):
    """The requested FDR operation would weaken pairing or calibration scope."""


def combined_score(strict: pd.Series, relaxed: pd.Series) -> pd.Series:
    """Return the equally weighted mean of the strict and relaxed scores."""

    strict_values = pd.to_numeric(strict, errors="raise").astype(float)
    relaxed_values = pd.to_numeric(relaxed, errors="raise").astype(float)
    if not np.isfinite(strict_values).all() or not np.isfinite(relaxed_values).all():
        raise FDRContractError("strict/relaxed scores contain NaN or infinity")
    return 0.5 * strict_values + 0.5 * relaxed_values


def calculate_fdr(rows: pd.DataFrame, score_name: str = "score", *, require_truth=True) -> pd.DataFrame:
    """Calculate the established cumulative Target/Counterpart FDR curve.

    ``is_decoy=0`` means Target and ``is_decoy=1`` means Counterpart.  The name
    is retained for numeric compatibility with the upstream implementation;
    Counterpart is paired CCT evidence, not a biological/database decoy.
    ``observed_fdp`` is meaningful only where ``denovo_label`` is truth-known.
    """

    required = {score_name, "is_decoy"} | ({"denovo_label"} if require_truth else set())
    missing = sorted(required - set(rows.columns))
    if missing:
        raise FDRContractError(f"FDR rows are missing column(s): {missing}")
    work = rows.copy()
    work[score_name] = pd.to_numeric(work[score_name], errors="raise")
    if not np.isfinite(work[score_name].to_numpy(float)).all():
        raise FDRContractError("FDR scores contain NaN or infinity")
    if not work["is_decoy"].isin((0, 1)).all():
        raise FDRContractError("is_decoy must contain only 0/1 side identities")
    if "denovo_label" not in work:
        work["denovo_label"] = np.nan
    if require_truth and not work["denovo_label"].isin((0, 1)).all():
        raise FDRContractError(
            "denovo_label must be truth-known 0/1 for calibration; use a "
            "separate deployment curve for truth-unknown candidates"
        )

    work = work.sort_values(by=score_name, ascending=False, kind="mergesort").reset_index(drop=True)
    cumulative_target = work["is_decoy"].eq(0).cumsum()
    work["cum_decoy"] = work["is_decoy"].eq(1).cumsum()
    work["cum_decoy_true"] = (work["is_decoy"].eq(1) & work["denovo_label"].eq(1)).cumsum()
    work["cum_decoy_false"] = (work["is_decoy"].eq(1) & work["denovo_label"].eq(0)).cumsum()
    work["cum_true"] = (work["is_decoy"].eq(0) & work["denovo_label"].eq(1)).cumsum()
    work["cum_false"] = (work["is_decoy"].eq(0) & work["denovo_label"].eq(0)).cumsum()
    work["estimated_fdr"] = work["cum_decoy"] / np.maximum(cumulative_target, 1)
    work["observed_fdp"] = work["cum_false"] / np.maximum(work["cum_true"] + work["cum_false"], 1)
    if not work["denovo_label"].isin((0, 1)).all():
        # Unknown reference labels never become negative training labels or FDP.
        work["observed_fdp"] = np.nan
        work["cum_false"] = np.nan
        work["cum_true"] = np.nan
    return work


def gamma_fdr(
    curve: pd.DataFrame,
    *,
    threshold: float,
    method: str = "interval",
    interval_margin: float = 0.1,
    interval_indices: tuple[int, int] | None = None,
) -> tuple[pd.DataFrame, float, float, int, tuple[float, float]]:
    """SI Note 1, Theorems 3/4, evaluated only at complete score groups.

    Interval endpoints: nearest observed FDP to q*(1 +/- margin), ties choose
    the most stringent endpoint; include EVERY intervening score group. The
    rule is fixed before calibration. No clipping or silent method fallback.
    interval_indices, when supplied, index the complete-score-group curve.
    """
    if not 0 < threshold <= 1 or not np.isfinite(threshold):
        raise FDRContractError("threshold must be in (0, 1]")
    if method not in ("interval", "suffix"):
        raise FDRContractError("gamma method must be interval or suffix")
    grouped = complete_score_group_curve(curve)
    if grouped.empty:
        raise FDRContractError("Empty calibration curve")
    beta = grouped.estimated_fdr.to_numpy(float)
    fdp = grouped.observed_fdp.to_numpy(float)
    target = grouped.cum_target.to_numpy(int)
    if not np.isfinite(fdp).all() or not np.isfinite(beta).all():
        raise FDRContractError("Calibration requires known reference labels")
    valid = target > 0
    ratio = np.divide(fdp, beta, out=np.zeros_like(fdp), where=beta > 0)
    ratio[(beta == 0) & (fdp > 0)] = np.inf
    ratio[~valid] = 0
    if not valid.any():
        raise FDRContractError("No nonempty-target threshold")
    if method == "suffix":
        if interval_indices is not None:
            raise FDRContractError("interval_indices cannot be used with suffix")
        maxima = np.maximum.accumulate(ratio[::-1])[::-1]
        product = np.full_like(beta, np.inf)
        finite = np.isfinite(maxima)
        product[finite] = maxima[finite] * beta[finite]
        feasible = np.flatnonzero(valid & finite & (product <= threshold))
        if not len(feasible):
            raise FDRContractError("No feasible suffix calibration threshold")
        selected = int(feasible[-1])
        left, right = selected, len(grouped) - 1
        gamma = float(maxima[selected])
        if gamma == 0:
            gamma = threshold / (1 + beta[selected])
    else:
        if not np.isfinite(interval_margin) or interval_margin <= 0:
            raise FDRContractError("interval_margin must be finite and positive")
        if interval_indices is None:
            eligible = np.flatnonzero(valid)
            endpoints = [int(eligible[np.argmin(np.abs(fdp[eligible] - boundary))])
                         for boundary in (threshold * (1 - interval_margin),
                                          threshold * (1 + interval_margin))]
            left, right = sorted(endpoints)
        else:
            left, right = interval_indices
            if not (isinstance(left, int) and isinstance(right, int)
                    and 0 <= left <= right < len(grouped)):
                raise FDRContractError("Invalid fixed interval indices")
        gamma = float(np.max(ratio[left:right + 1]))
        if not np.isfinite(gamma):
            raise FDRContractError("No finite gamma: incorrect targets without counterparts in interval")
        if gamma == 0:
            gamma = threshold / (1 + np.max(beta[left:right + 1]))
        eligible = np.arange(left, right + 1)
        feasible = eligible[valid[eligible] & (gamma * beta[eligible] <= threshold)]
        if not len(feasible):
            raise FDRContractError("No feasible threshold inside fixed calibration interval")
        selected = int(feasible[-1])
    gamma = float(gamma)
    result = apply_frozen_gamma(curve, gamma)
    cutoff = float(grouped.score.iloc[selected])
    optimal_index = int(np.flatnonzero(curve.score.to_numpy() == cutoff)[-1])
    result.attrs["gamma_calibration"] = {
        "method": method, "threshold": float(threshold), "gamma": gamma,
        "interval_margin": interval_margin if method == "interval" else None,
        "interval_rule": "nearest_FDP_boundaries_first_tie_contiguous" if method == "interval" else "suffix_maximum",
        "group_indices": [int(left), int(right)],
        "score_bounds": [float(grouped.score.iloc[right]), float(grouped.score.iloc[left])],
        "selected_score": cutoff, "selected_group": selected,
        "selected_reference_fdp": float(fdp[selected]),
        "selected_estimated_fdr": float(gamma * beta[selected]),
        "scope": "reference_relative_calibration_range_only",
    }
    return result, gamma, float(fdp[selected]), optimal_index, (float(fdp[left]), float(fdp[right]))


def apply_frozen_gamma(curve: pd.DataFrame, gamma: float) -> pd.DataFrame:
    """Apply a positive, finite DB-calibrated γ without fitting on full data."""

    gamma = float(gamma)
    if not np.isfinite(gamma) or gamma <= 0:
        raise FDRContractError("frozen gamma must be finite and positive")
    if "estimated_fdr" not in curve:
        raise FDRContractError("curve lacks estimated_fdr")
    result = curve.copy()
    result["estimated_fdr_update"] = gamma * pd.to_numeric(
        result["estimated_fdr"], errors="raise"
    )
    return result


def complete_score_group_curve(curve: pd.DataFrame) -> pd.DataFrame:
    """Collapse tied scores only at group ends before choosing a cutoff."""

    if curve.empty:
        return curve.assign(cum_target=pd.Series(dtype=int)).copy()
    scores = pd.to_numeric(curve["score"], errors="raise").to_numpy(float)
    if len(scores) > 1 and np.any(scores[:-1] < scores[1:]):
        raise FDRContractError("FDR curve is not sorted by descending score")
    group_end = np.r_[scores[:-1] != scores[1:], True]
    result = curve.loc[group_end].copy().reset_index(drop=True)
    if "cum_decoy" in result:
        result["cum_target"] = (
            np.arange(1, len(curve) + 1)[group_end]
            - result["cum_decoy"].to_numpy(int)
        )
    return result


def accept_complete_score_groups(
    curve: pd.DataFrame,
    threshold: float,
    *,
    fdr_column: str = "estimated_fdr_update",
    score_bounds: tuple[float, float] | None = None,
) -> tuple[pd.DataFrame, float]:
    """Accept the lowest complete tied-score group that passes the FDR target."""

    grouped = complete_score_group_curve(curve)
    values = pd.to_numeric(grouped[fdr_column], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise FDRContractError(f"curve contains non-finite {fdr_column}")
    if not 0 < threshold <= 1:
        raise FDRContractError("threshold must be in (0, 1]")
    eligible = (values <= float(threshold)) & (grouped.cum_target.to_numpy() > 0)
    if score_bounds is not None:
        low, high = score_bounds
        if not np.isfinite([low, high]).all() or low > high:
            raise FDRContractError("Invalid calibrated score range")
        eligible &= grouped.score.between(low, high).to_numpy()
    passing = np.flatnonzero(eligible)
    if not len(passing):
        return curve.iloc[0:0].copy(), float("nan")
    cutoff = float(grouped.iloc[int(passing[-1])]["score"])
    return curve.loc[curve["score"].ge(cutoff)].copy(), cutoff


