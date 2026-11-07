# Exercise score ties, calibration bounds and peptide pair restoration.
# 覆盖并列分数、校准边界和肽段配对恢复。
import json
import numpy as np
import pandas as pd
import pytest

from MNovo.cct.cct_core.fdr import (FDRContractError, calculate_fdr, gamma_fdr,
    complete_score_group_curve, accept_complete_score_groups, apply_frozen_gamma)
from MNovo.cct.cct_scoring import peptide_candidates, load_pairs, select_psms


def curve(sides, labels, scores=None):
    return calculate_fdr(pd.DataFrame({"is_decoy": sides, "denovo_label": labels,
        "score": scores if scores is not None else np.arange(len(sides), 0, -1, dtype=float)}))


def test_interval_interior_maximum_not_endpoints():
    c = curve([0, 1, 0, 0, 0, 1], [1, 1, 0, 0, 1, 0])
    a, g, _, idx, _ = gamma_fdr(c, threshold=.6, interval_indices=(2, 5))
    assert g == 2  # endpoint ratios 1, 1; interior ratio 2
    assert idx == 4  # 2 * 1/4=.5; later 2 * 2/4=1 is infeasible
    assert a.attrs["gamma_calibration"]["selected_score"] == c.score.iloc[4]
    assert np.all(a.estimated_fdr_update.iloc[2:6] >= a.observed_fdp.iloc[2:6])


@pytest.mark.parametrize("seed", range(20))
def test_suffix_matches_bruteforce(seed):
    rng = np.random.default_rng(seed)
    c = curve(rng.integers(0, 2, 80), rng.integers(0, 2, 80), rng.integers(0, 20, 80))
    groups = complete_score_group_curve(c)
    beta = groups.estimated_fdr.to_numpy()
    fdp = groups.observed_fdp.to_numpy()
    ratio = np.array([f/b if b else (np.inf if f else 0) for b, f in zip(beta, fdp)])
    ratio[groups.cum_target.to_numpy() == 0] = 0
    maxima = np.array([max(ratio[i:]) for i in range(len(ratio))])
    feasible = [i for i in range(len(ratio)) if groups.cum_target.iloc[i] > 0
                and np.isfinite(maxima[i]) and maxima[i]*beta[i] <= .5]
    if not feasible:
        with pytest.raises(FDRContractError, match="No feasible"):
            gamma_fdr(c, threshold=.5, method="suffix")
    else:
        adjusted, g, _, _, _ = gamma_fdr(c, threshold=.5, method="suffix")
        audit = adjusted.attrs["gamma_calibration"]
        i = feasible[-1]
        assert audit["selected_group"] == i
        assert audit["selected_estimated_fdr"] <= .5
        assert np.all(g * beta[i:] + 1e-14 >= fdp[i:])


def test_zero_ratio_uses_positive_document_branch():
    c = curve([0, 1, 0, 1], [1]*4)
    _, g, _, _, _ = gamma_fdr(c, threshold=.05, method="suffix")
    assert g == .025
    _, g, _, _, _ = gamma_fdr(c, threshold=.05, interval_indices=(0, 3))
    assert g == .025


def test_zero_counterparts_wrong_target_is_infeasible_no_clipping():
    c = curve([0, 0], [0, 1])
    with pytest.raises(FDRContractError, match="No finite"):
        gamma_fdr(c, threshold=.05, interval_indices=(0, 1))
    with pytest.raises(FDRContractError, match="No feasible"):
        gamma_fdr(c, threshold=.05, method="suffix")


def test_no_feasible_interval_does_not_silently_fallback():
    c = curve([0, 1], [0, 0])
    with pytest.raises(FDRContractError, match="No feasible"):
        gamma_fdr(c, threshold=.05, interval_indices=(1, 1))


def test_gamma_above_100_is_not_clipped():
    c = curve([0]*200 + [1], [0]*101 + [1]*100)
    _, g, _, _, _ = gamma_fdr(c, threshold=.6, interval_indices=(200, 200))
    assert g == 101


def test_tie_order_does_not_affect_calibration():
    c = curve([0, 1, 0, 1], [0, 1, 1, 0], [.9, .9, .8, .8])
    d = calculate_fdr(c.iloc[[1, 0, 3, 2]][["score", "is_decoy", "denovo_label"]])
    a = gamma_fdr(c, threshold=.5, method="suffix")
    b = gamma_fdr(d, threshold=.5, method="suffix")
    assert a[1:] == b[1:]


def test_empty_target_never_selected_and_empty_curve_supported():
    c = curve([1], [1])
    accepted, cutoff = accept_complete_score_groups(apply_frozen_gamma(c, .01), .05)
    assert accepted.empty and np.isnan(cutoff)
    c = calculate_fdr(pd.DataFrame(columns=["score", "is_decoy"]), require_truth=False)
    accepted, cutoff = accept_complete_score_groups(apply_frozen_gamma(c, 1), .05)
    assert accepted.empty


def paired_rows():
    records = []
    for pid, seq, scores in [(1, "PEPTIDE", (.9, .4)), (2, "MAGNET", (.4, .9)),
                              (3, "OTHER", (.1, .2)), (4, "PEPTIDE", (.85, .8))]:
        for side, score in enumerate(scores):
            records.append(dict(pair_id=pid, sequence=seq, peptide_key=seq, is_decoy=side,
                denovo_label=1, valid=True, score=score,
                **{f"f_{i:02d}": float(i) for i in range(33)}))
    return pd.DataFrame(records)


def test_counterpart_only_pair_restored_and_entire_winning_pair_kept():
    result = peptide_candidates(paired_rows(), .8)
    assert set(result.sequence) == {"PEPTIDE", "MAGNET"}
    assert len(result) == 4
    assert set(result.loc[result.sequence.eq("MAGNET"), "is_decoy"]) == {0, 1}
    assert result.loc[result.sequence.eq("PEPTIDE") & result.is_decoy.eq(1), "pair_id"].item() == 1
    assert peptide_candidates(paired_rows(), float("nan")).empty


def test_unknown_labels_do_not_change_application_cutoff(tmp_path):
    rows = paired_rows()
    unknown = rows.assign(denovo_label=np.nan)
    a = select_psms(rows, .05, .1, require_truth=False)
    b = select_psms(unknown, .05, .1, require_truth=False)
    assert a[2] == b[2]
    assert a[0].pair_id.tolist() == b[0].pair_id.tolist()
    assert b[1].observed_fdp.isna().all()
    path = tmp_path / "unknown.csv"
    unknown.to_csv(path, index=False)
    assert len(load_pairs(path, require_truth=False)[0]) == len(rows)
    with pytest.raises(ValueError, match="Truth-known"):
        load_pairs(path)


def test_profile_mismatch_fails_before_inference(tmp_path):
    import torch
    from MNovo.cct.cct_pipeline import SCHEMA, load_profile
    p = dict(schema=SCHEMA, scope_id="A", runtime_identity={"backbone": "sha", "source_sha256": {"a.py": "old"}},
        threshold_policy="recount_full_cohort_above_calibrated_score_floor",
        psm_q=.05, peptide_q=.05, psm_calibration={"gamma": 1., "score_bounds": [0., 1.]},
        peptide_calibration={"gamma": 1., "score_bounds": [0., 1.]})
    path = tmp_path / "profile.pt"
    torch.save(p, path)
    assert load_profile(path, p["runtime_identity"], "A", .05, .05)["schema"] == SCHEMA
    for ident in ({"backbone": "sha"}, {"backbone": "sha", "source_sha256": {"a.py": "changed"}}):
        assert load_profile(path, ident, "A", .05, .05)["runtime_identity"] == p["runtime_identity"]
    for identity, scope, q in [({}, "A", .05), ({"backbone": "different"}, "A", .05),
                                ({"backbone": "sha", "new_setting": True}, "A", .05),
                                (p["runtime_identity"], "B", .05),
                                (p["runtime_identity"], "A", .01)]:
        with pytest.raises(FDRContractError):
            load_profile(path, identity, scope, q, .05)


def test_apply_two_levels_with_unknown_truth(tmp_path):
    import torch
    from sklearn.preprocessing import StandardScaler
    from MNovo.cct.cct_scoring import MLPClassifier, _pack_models
    from MNovo.cct.cct_pipeline import apply_profile, SCHEMA
    rows = paired_rows().assign(denovo_label=np.nan)
    cols = sorted(c for c in rows if c.startswith("f_"))
    scaler = StandardScaler().fit(rows[cols].to_numpy())
    models = {name: (MLPClassifier(33).eval(), scaler, {}) for name in ("strict", "relaxed")}
    packed = _pack_models(models)
    profile = dict(schema=SCHEMA, scope_id="synthetic", feature_columns=cols,
        threshold_policy="recount_full_cohort_above_calibrated_score_floor",
        psm=packed, peptide=packed, psm_q=.05, peptide_q=.05,
        psm_calibration={"gamma": .01, "score_bounds": [0., 1.]},
        peptide_calibration={"gamma": .01, "score_bounds": [0., 1.]})
    path = tmp_path / "features.csv"
    rows.to_csv(path, index=False)
    report = apply_profile(path, profile, tmp_path / "result")
    assert report["accepted_psms"] == 4
    assert report["accepted_peptides"] == 3
    assert (tmp_path / "result/99_complete.done").exists()


def test_fit_save_reload_apply_profile_end_to_end(tmp_path):
    from MNovo.cct.cct_pipeline import fit_profile, load_profile, apply_profile, SCHEMA, _sha256
    rng = np.random.default_rng(31)
    paths = []
    for role_number, role in enumerate(("fit", "cal", "validation")):
        records = []
        for pid in range(64):
            for side in (0, 1):
                records.append(dict(pair_id=pid + role_number * 1000, group_id=f"{role}_{pid}", sequence=f"PEPTIDE{role}{pid}",
                    denovo_label=pid % 2, is_decoy=side, valid=True,
                    **{f"f_{i:02d}": float(rng.normal()) for i in range(33)}))
        path = tmp_path / (role + ".csv")
        pd.DataFrame(records).to_csv(path, index=False)
        Path = type(path)
        Path(str(path) + ".receipt.json").write_text(json.dumps(dict(
            schema=SCHEMA, features_sha256=_sha256(path), scope_id="synthetic",
            runtime_identity={"test_only": True})))
        paths.append(path)
    output = tmp_path / "fit_result"
    fit_profile(paths[0], paths[1], output, validation_path=paths[2], psm_q=1., peptide_q=1., method="suffix", epochs=1)
    profile = load_profile(output / "cct_profile.pt", {"test_only": True}, "synthetic", 1., 1.)
    application = tmp_path / "unknown.csv"
    pd.read_csv(paths[1]).assign(denovo_label=np.nan).to_csv(application, index=False)
    report = apply_profile(application, profile, tmp_path / "application")
    assert report["accepted_psms"] == 64
    assert report["accepted_peptides"] == 64
    assert (tmp_path / "application/99_complete.done").exists()


def test_local_gamma_cannot_escape_calibrated_interval():
    c = curve([0]*20 + [1]*20 + [0]*80 + [1]*80,
              [1]*40 + [0]*160, [.9]*20 + [.8]*20 + [.7]*80 + [.6]*80)
    adjusted, g, _, _, _ = gamma_fdr(c, threshold=.05)
    assert g == .05
    bad, _ = accept_complete_score_groups(adjusted, .05)
    assert bad.is_decoy.eq(0).sum() == 100
    accepted, cutoff = accept_complete_score_groups(
        adjusted, .05, score_bounds=adjusted.attrs["gamma_calibration"]["score_bounds"])
    assert cutoff == .9
    assert accepted.is_decoy.eq(0).sum() == 20
    assert accepted.denovo_label.eq(0).sum() == 0


def test_empty_test_scoring_is_valid_zero_discovery():
    from MNovo.cct.cct_scoring import score_rows
    empty = paired_rows().iloc[:0]
    scored = score_rows(empty, [], {})
    target, result, cutoff = select_psms(scored, .05, 1.)
    assert target.empty and result.empty and np.isnan(cutoff)


def test_full_precision_scaler_and_score_roundtrip():
    from sklearn.preprocessing import StandardScaler
    from MNovo.cct.cct_pipeline import _unpack
    from MNovo.cct.cct_scoring import MLPClassifier, _pack_models, score_rows
    rows = paired_rows()
    cols = sorted(c for c in rows if c.startswith("f_"))
    values = np.random.default_rng(77).normal(size=(200, 33)) * 1e-4 + .123456789
    scaler = StandardScaler().fit(values)
    models = {n: (MLPClassifier(33).eval(), scaler, {}) for n in ("strict", "relaxed")}
    loaded = _unpack(_pack_models(models), cols)
    np.testing.assert_array_equal(loaded["strict"][1].mean_, scaler.mean_)
    np.testing.assert_array_equal(loaded["strict"][1].scale_, scaler.scale_)
    a, b = score_rows(rows, cols, models), score_rows(rows, cols, loaded)
    np.testing.assert_array_equal(a.score, b.score)
