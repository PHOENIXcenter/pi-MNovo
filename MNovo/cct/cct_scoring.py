# Score both views, then restore pairs before collapsing peptides.
# 为两种视图评分，并在肽段归并前恢复完整配对。
"""Train and evaluate paired CCT score models at PSM and peptide levels."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from MNovo.cct.cct_core.fdr import (
    FDRContractError, accept_complete_score_groups, apply_frozen_gamma,
    calculate_fdr, combined_score, gamma_fdr,
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class MLPClassifier(torch.nn.Module):
    """The user's frozen 33-512-256-128-1 score-model architecture."""

    def __init__(self, in_dim=33):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 512), torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512), torch.nn.Dropout(0),
            torch.nn.Linear(512, 256), torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256), torch.nn.Dropout(0),
            torch.nn.Linear(256, 128), torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128), torch.nn.Linear(128, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, features):
        return self.net(features).squeeze(-1)


def load_pairs(path, *, require_truth=True, allow_empty=False):
    rows = pd.read_csv(path)
    required = {"pair_id", "sequence", "denovo_label", "is_decoy", "valid"}
    missing = required - set(rows)
    if missing:
        raise ValueError(f"Feature table lacks {sorted(missing)}")
    # The trained scaler sees this sorted column order, not the CSV's physical order.
    # scaler 和 MLP 按此排序后的列顺序读数，不按 CSV 的物理列序。
    feature_columns = sorted(column for column in rows if column.startswith("f_"))
    if len(feature_columns) != 33:
        raise ValueError(f"Expected 33 reference features, found {len(feature_columns)}")
    if rows.pair_id.isna().any() or rows.duplicated(["pair_id", "is_decoy"]).any():
        raise ValueError("Each spectrum needs unique target/counterpart records")
    if not rows.groupby("pair_id").is_decoy.agg(lambda x: set(x) == {0, 1}).all():
        raise ValueError("Every spectrum needs both CCT views")
    rows["valid"] = rows.valid.astype(str).str.lower().eq("true")
    complete = rows.groupby("pair_id").valid.transform("all")
    excluded = int(rows.pair_id.nunique() - rows.loc[complete, "pair_id"].nunique())
    rows = rows.loc[complete].copy()
    if (rows.empty and not allow_empty) or (require_truth and not rows.denovo_label.isin([0, 1]).all()):
        raise ValueError("Truth-known complete CCT pairs are required")
    if rows.sequence.isna().any() or rows.sequence.astype(str).str.len().eq(0).any():
        raise ValueError("A complete pair has an empty predicted peptide")
    values = rows[feature_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("A complete pair has nonfinite reference features")
    if (not rows.empty and (rows.groupby("pair_id").sequence.nunique().max() != 1
            or rows.groupby("pair_id").denovo_label.nunique(dropna=False).max() != 1)):
        raise ValueError("Two views of one spectrum disagree on peptide or truth")
    rows["peptide_key"] = rows.sequence.str.strip().str.replace("I", "L", regex=False)
    return rows, feature_columns, excluded








def score_rows(rows, columns, models):
    result = rows.copy()
    if result.empty:
        for column in ("strict_score", "relaxed_score", "score"):
            result[column] = pd.Series(dtype=float)
        return result
    values = result[columns].to_numpy(dtype=np.float32)
    for name in ("strict", "relaxed"):
        model, scaler, _metrics = models[name]
        model.eval()
        with torch.no_grad():
            transformed = torch.tensor(
                scaler.transform(values), dtype=torch.float32,
                device=next(model.parameters()).device,
            )
            result[name + "_score"] = np.concatenate([
                model(batch).cpu().numpy() for batch in transformed.split(4096)
            ])
    result["score"] = combined_score(result.strict_score, result.relaxed_score)
    return result


def calibrate(rows, threshold, *, method="interval", interval_margin=0.1, return_audit=False):
    curve = calculate_fdr(rows)
    _adjusted, gamma, _point, _index, _interval = gamma_fdr(
        curve, threshold=threshold, method=method, interval_margin=interval_margin)
    return (gamma, _adjusted.attrs["gamma_calibration"]) if return_audit else gamma


def select_psms(rows, threshold, gamma, *, require_truth=True, score_bounds=None):
    curve = apply_frozen_gamma(calculate_fdr(rows, require_truth=require_truth), gamma)
    if score_bounds is not None:
        low, high = score_bounds
        if not np.isfinite([low, high]).all() or low > high:
            raise FDRContractError("Invalid calibrated score bounds")
        score_bounds = (low, np.finfo(float).max)
    accepted, cutoff = accept_complete_score_groups(curve, threshold, score_bounds=score_bounds)
    target = accepted.loc[accepted.is_decoy.eq(0)].copy()
    return target, curve, cutoff


def peptide_candidates(psm_rows, cutoff):
    """Restore either-side passing pairs; keep one entire representative pair."""
    # Either side can recover a pair; do not discard its partner before peptide collapse.
    # 任一侧达阈值即可恢复整对记录；肽段去重前不能丢弃另一侧。
    ids = psm_rows.loc[psm_rows.score.ge(float(cutoff)), "pair_id"].unique()
    recovered = psm_rows.loc[psm_rows.pair_id.isin(ids)].copy()
    winners = recovered.sort_values(
        ["score", "pair_id", "is_decoy"], ascending=[False, True, True],
        kind="mergesort").drop_duplicates("peptide_key")
    return recovered.loc[recovered.pair_id.isin(winners.pair_id)].copy()


def _pack_models(branch):
    packed = {}
    for name, (model, scaler, metrics) in branch.items():
        packed[name] = {
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "scaler_mean": None if scaler is None else torch.tensor(scaler.mean_, dtype=torch.float64),
            "scaler_scale": None if scaler is None else torch.tensor(scaler.scale_, dtype=torch.float64),
            "metrics": metrics,
        }
    return packed









