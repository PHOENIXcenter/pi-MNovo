# Fit or apply a scope-bound profile; neither path reroutes a peptide.
# 拟合或应用限定适用范围的 profile，不重新选择肽段。
"""Optional CCT fitting/deployment; never changes native sequence selection.

Only locally trusted profiles should be loaded. Profiles bind the two score
models, both gammas, input provenance, scope and exact inference identity.
This is a CTC adaptation, not an empirical FDR certification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from MNovo.cct.cct_core.fdr import FDRContractError, calculate_fdr, gamma_fdr
from MNovo.cct.cct_scoring import (
    MLPClassifier, _pack_models, _sha256, load_pairs,
    peptide_candidates, score_rows, select_psms,
)
from MNovo.cct.cct_training import train_fixed

SCHEMA = "mnovo-cct-fixed-roles-v2"
IDENTITY_POLICY = "full_modified_sequence_IL_merged_inherited_adapter"


def runtime_identity(model_dir, config_path, options, device="auto"):
    root = Path(model_dir)
    verified = json.loads((root / "verified.json").read_text())
    # The release resolver has already verified the bytes behind this manifest.
    import yaml
    config = yaml.safe_load(Path(config_path).read_text())
    config.pop("runtime", None)
    from MNovo.inference.config import resolve_settings
    option_fields = vars(options).copy()
    config = resolve_settings(config, option_fields.pop("inference_overrides", None) or {})
    device_class = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device).type
    return {
        "components": verified["manifest"]["components"],
        "inference_config": config,
        "options": option_fields,
        "device_class": device_class,
        "effective_precision": "bf16_autocast" if device_class == "cuda" and options.precision == "bf16" else "fp32",
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "feature_variant": "CTC_occurrence_offset_33_features",
        "peptide_identity": IDENTITY_POLICY,
    }


def inference_contract(identity):
    # rc2 also recorded package source bytes. All other fields remain binding.
    # 只忽略旧版的源码摘要，组件、配置以及未知的其他字段仍然参与比较。
    contract = {key: value for key, value in identity.items() if key != "source_sha256"}
    if not contract:
        raise FDRContractError("Missing CCT inference identity")
    return contract


def write_receipt(path, identity, scope_id, exporter):
    payload = {
        "schema": SCHEMA, "scope_id": scope_id, "runtime_identity": identity,
        "features_sha256": _sha256(path),
        "complete_pairs": exporter.complete_pairs, "invalid_pairs": exporter.invalid_pairs,
        "labelled_pairs": exporter.labelled_pairs,
    }
    Path(str(path) + ".receipt.json").write_text(json.dumps(payload, indent=2))


def _unpack(branch, columns):
    result = {}
    for name in ("strict", "relaxed"):
        item = branch[name]
        model = MLPClassifier(len(columns)).cpu()
        model.load_state_dict(item["state_dict"], strict=True)
        model.eval()
        scaler = StandardScaler()
        scaler.mean_ = item["scaler_mean"].numpy()
        scaler.scale_ = item["scaler_scale"].numpy()
        if (len(scaler.mean_) != len(columns) or not np.isfinite(scaler.mean_).all()
                or not np.isfinite(scaler.scale_).all() or np.any(scaler.scale_ <= 0)):
            raise FDRContractError("Invalid profile scaler")
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_features_in_ = len(columns)
        result[name] = model, scaler, item.get("metrics", {})
    return result


def load_profile(path, identity, scope_id, psm_q, peptide_q):
    profile = torch.load(path, map_location="cpu", weights_only=True)
    if profile.get("schema") != SCHEMA:
        raise FDRContractError("Unsupported CCT profile; refit with corrected protocol")
    if profile.get("threshold_policy") != "recount_full_cohort_above_calibrated_score_floor":
        raise FDRContractError("Profile lacks calibrated score-floor policy")
    if not scope_id or profile["scope_id"] != scope_id:
        raise FDRContractError("CCT scope differs from calibrated profile")
    if inference_contract(profile["runtime_identity"]) != inference_contract(identity):
        raise FDRContractError("Inference identity differs from calibrated profile")
    if profile["psm_q"] != psm_q or profile["peptide_q"] != peptide_q:
        raise FDRContractError("Requested FDR levels have no matching calibration")
    for name in ("psm", "peptide"):
        gamma = profile[name + "_calibration"]["gamma"]
        if not np.isfinite(gamma) or gamma <= 0:
            raise FDRContractError("Profile contains invalid gamma")
        bounds = profile[name + "_calibration"].get("score_bounds", [])
        if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
            raise FDRContractError("Profile contains invalid calibration range")
    return profile


class CalibrationFailure(FDRContractError):
    """The calibration data could not support the requested operating point."""


def calibrate_with_audit(rows, q, method, margin, level="CCT"):
    try:
        adjusted, gamma, _, _, _ = gamma_fdr(
            calculate_fdr(rows), threshold=q, method=method, interval_margin=margin)
    except FDRContractError as exc:
        reason = {
            "No finite gamma: incorrect targets without counterparts in interval":
                "the calibration interval contains incorrect predictions but lacks the counterpart evidence needed to estimate their error rate",
            "No feasible threshold inside fixed calibration interval":
                "no threshold in the calibration interval meets the requested FDR",
        }.get(str(exc), str(exc))
        raise CalibrationFailure(f"{level} FDR calibration failed at {q:.2%}: {reason}.") from exc
    return gamma, adjusted.attrs["gamma_calibration"]


def known_rows(rows):
    """Unknown references never become negative supervision."""
    return rows.loc[rows.denovo_label.isin([0, 1])].copy()


def validate_roles(roles):
    seen_groups, seen_pairs = set(), set()
    for name, rows in roles.items():
        if "group_id" not in rows or rows.group_id.isna().any():
            raise FDRContractError(f"{name}: explicit physical group_id required")
        if not rows.groupby("pair_id").group_id.nunique().eq(1).all():
            raise FDRContractError(f"{name}: pair crosses physical groups")
        groups, pairs = set(rows.group_id.astype(str)), set(rows.pair_id.astype(str))
        if seen_groups & groups or seen_pairs & pairs:
            raise FDRContractError(f"{name}: physical groups or pair IDs overlap roles")
        seen_groups.update(groups)
        seen_pairs.update(pairs)


def fit_profile(fit_path, cal_path, output, *, validation_path, psm_q=.05, peptide_q=.05,
                method="interval", margin=.1, epochs=50, seed=454):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    receipts = []
    for path in (fit_path, cal_path, validation_path):
        receipt = json.loads(Path(str(path) + ".receipt.json").read_text())
        if receipt.get("schema") != SCHEMA or receipt["features_sha256"] != _sha256(path):
            raise FDRContractError("Feature receipt does not match input")
        receipt["runtime_identity"] = inference_contract(receipt["runtime_identity"])
        receipts.append(receipt)
    if not receipts[0]["scope_id"] or any(
        receipts[0][key] != other[key]
        for other in receipts[1:] for key in ("runtime_identity", "scope_id")
    ):
        raise FDRContractError("Fit/calibration identities or scope differ")
    fit, columns, fit_excluded = load_pairs(fit_path, require_truth=False)
    cal, cal_columns, cal_excluded = load_pairs(cal_path, require_truth=False)
    validation, validation_columns, validation_excluded = load_pairs(validation_path, require_truth=False)
    if columns != cal_columns or columns != validation_columns:
        raise FDRContractError("Feature schemas differ")
    roles = {"train": fit, "early_stop": validation, "calibration": cal}
    validate_roles(roles)
    # Preserve failed attempt diagnostics, and never overwrite an older fit.
    output.mkdir(parents=True)
    try:
        # Calibrate with the SAME CPU scorer and full-precision scaler that
        # deployment reloads, not the training-device numerical path.
        psm_models = train_fixed(known_rows(fit), known_rows(validation), columns,
                                 output / "psm_models", seed, epochs)
        fit_scores = score_rows(fit, columns, psm_models)
        validation_scores = score_rows(validation, columns, psm_models)
        cal_scores = score_rows(cal, columns, psm_models)
        psm_gamma, psm_audit = calibrate_with_audit(known_rows(cal_scores), psm_q, method, margin, "PSM")
        def entry_cutoff(scores):
            return select_psms(scores, psm_q, psm_gamma, require_truth=False,
                               score_bounds=psm_audit["score_bounds"])[2]
        fit_cutoff = entry_cutoff(fit_scores)
        fit_peptides = peptide_candidates(fit_scores, fit_cutoff)
        validation_peptides = peptide_candidates(validation_scores, entry_cutoff(validation_scores))
        cal_peptides = peptide_candidates(cal_scores, entry_cutoff(cal_scores))
        if fit_peptides.empty or cal_peptides.empty:
            raise FDRContractError("No peptide candidates after paired restoration")
        peptide_models = train_fixed(known_rows(fit_peptides), known_rows(validation_peptides),
                                     columns, output / "peptide_models", seed, epochs,
                                     device="cpu")
        cal_peptide_scores = score_rows(cal_peptides, columns, peptide_models)
        _, peptide_audit = calibrate_with_audit(known_rows(cal_peptide_scores), peptide_q, method, margin, "Peptide")
        profile = {
            "schema": SCHEMA, "status": "CANDIDATE_NOT_EMPIRICALLY_VALIDATED",
            "scope_id": receipts[0]["scope_id"],
            "runtime_identity": receipts[0]["runtime_identity"],
            "feature_columns": columns, "psm_q": psm_q, "peptide_q": peptide_q,
            "psm": _pack_models(psm_models), "peptide": _pack_models(peptide_models),
            "psm_calibration": psm_audit, "peptide_calibration": peptide_audit,
            "source_sha256": {"fit": _sha256(fit_path), "calibration": _sha256(cal_path),
                              "early_stop": _sha256(validation_path)},
            "excluded_pairs": {"fit": fit_excluded, "calibration": cal_excluded,
                               "early_stop": validation_excluded},
            "physical_groups": {name: sorted(frame.group_id.astype(str).unique().tolist())
                                for name, frame in roles.items()},
            "seed": seed, "max_epochs": epochs,
            "fit_protocol": "explicit_scan_disjoint_roles_no_resplit",
            "pair_restoration": "target_OR_counterpart_ge_psm_cutoff",
            "threshold_policy": "recount_full_cohort_above_calibrated_score_floor",
        }
        torch.save(profile, output / "cct_profile.pt")
        report = {key: value for key, value in profile.items() if key not in ("psm", "peptide")}
        report["profile_sha256"] = _sha256(output / "cct_profile.pt")
        (output / "profile.json").write_text(json.dumps(report, indent=2))
        cal_scores.to_csv(output / "calibration_psm_scores.csv", index=False)
        cal_peptide_scores.to_csv(output / "calibration_peptide_scores.csv", index=False)
        (output / "99_complete.done").write_text("complete\n")
        return report
    except Exception as error:
        (output / "failure.json").write_text(json.dumps({"error": str(error)}, indent=2))
        raise


def apply_profile(features_path, profile, output, *, independent=False):
    """Recompute cutoffs on all valid pairs; unknown truths do not affect selection."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    rows, columns, excluded = load_pairs(features_path, require_truth=False, allow_empty=True)
    if columns != profile["feature_columns"]:
        raise FDRContractError("Profile feature order mismatch")
    if independent:
        if "group_id" not in rows or rows.group_id.isna().any():
            raise FDRContractError("Independent evaluation requires physical group_id")
        fitted = set().union(*[set(v) for v in profile["physical_groups"].values()])
        if fitted & set(rows.group_id.astype(str)):
            raise FDRContractError("Independent evaluation overlaps fitted/calibration groups")
    output.mkdir(parents=True)
    try:
        # Deployment reuses the fitted models and gamma but recounts the current cohort.
        # 应用时沿用已拟合的模型和 γ，阈值计数针对当前整批数据。
        psm_scores = (score_rows(rows, columns, _unpack(profile["psm"], columns))
                      if len(rows) else rows.assign(score=pd.Series(dtype=float)))
        target, curve, cutoff = select_psms(
            psm_scores, profile["psm_q"], profile["psm_calibration"]["gamma"], require_truth=False,
            score_bounds=profile["psm_calibration"]["score_bounds"])
        peptides = peptide_candidates(psm_scores, cutoff)
        peptide_scores = (score_rows(peptides, columns, _unpack(profile["peptide"], columns))
                          if len(peptides) else peptides)
        accepted_peptides, peptide_curve, peptide_cutoff = select_psms(
            peptide_scores, profile["peptide_q"], profile["peptide_calibration"]["gamma"],
            require_truth=False, score_bounds=profile["peptide_calibration"]["score_bounds"])
        for name, table in (("all_psm_scores", psm_scores), ("psm_curve", curve),
                            ("accepted_psms", target), ("peptide_candidates", peptides),
                            ("peptide_curve", peptide_curve), ("accepted_peptides", accepted_peptides)):
            table.to_csv(output / (name + ".tsv"), sep="\t", index=False)
        report = {
            "schema": SCHEMA, "scope_id": profile["scope_id"],
            "features_sha256": _sha256(features_path), "valid_pairs": int(rows.pair_id.nunique()),
            "excluded_invalid_pairs": excluded, "accepted_psms": len(target),
            "accepted_peptides": len(accepted_peptides),
            "psm_cutoff": float(cutoff) if np.isfinite(cutoff) else None,
            "peptide_cutoff": float(peptide_cutoff) if np.isfinite(peptide_cutoff) else None,
            "psm_q": profile["psm_q"], "peptide_q": profile["peptide_q"],
            "psm_gamma": profile["psm_calibration"]["gamma"],
            "peptide_gamma": profile["peptide_calibration"]["gamma"],
            "meaning": "estimated_CCT_FDR_not_empirical_certification_or_PEP",
            "threshold_policy": profile["threshold_policy"],
            "reference_evaluation": "independent_fixed_test" if independent else "full_application_not_independent",
            "psm_calibrated_score_bounds": profile["psm_calibration"]["score_bounds"],
            "peptide_calibrated_score_bounds": profile["peptide_calibration"]["score_bounds"],
        }
        for stage, accepted, values, threshold in (
            ("psm", target, psm_scores, cutoff),
            ("peptide", accepted_peptides, peptide_scores, peptide_cutoff),
        ):
            known = accepted.denovo_label.isin([0, 1])
            wrong = int(accepted.loc[known, "denovo_label"].eq(0).sum())
            counterparts = int((values.is_decoy.eq(1) & values.score.ge(threshold)).sum()) if "score" in values else 0
            report[stage + "_metrics"] = dict(
                retained=len(accepted), reference_known=int(known.sum()),
                reference_unknown=int((~known).sum()), reference_wrong=wrong,
                reference_fdp=wrong / int(known.sum()) if known.any() else None,
                counterparts=counterparts,
                estimated_fdr=profile[stage + "_calibration"]["gamma"] * counterparts / len(accepted) if len(accepted) else None)
            if "raw" in values:
                raw_rows = []
                for raw in sorted(values.raw.dropna().unique()):
                    subset = accepted.loc[accepted.raw.eq(raw)]
                    known_raw = subset.denovo_label.isin([0, 1])
                    errors = int(subset.loc[known_raw, "denovo_label"].eq(0).sum())
                    counters = int((values.raw.eq(raw) & values.is_decoy.eq(1) & values.score.ge(threshold)).sum())
                    raw_rows.append(dict(raw=raw, retained=len(subset), reference_known=int(known_raw.sum()),
                        reference_unknown=int((~known_raw).sum()), reference_wrong=errors,
                        reference_fdp=errors / int(known_raw.sum()) if known_raw.any() else None,
                        counterpart=counters, estimated_fdr=profile[stage + "_calibration"]["gamma"] * counters / len(subset) if len(subset) else None,
                        threshold_scope="pooled_no_RAW_retuning"))
                pd.DataFrame(raw_rows).to_csv(output / (stage + "_by_raw.tsv"), sep="\t", index=False)
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        (output / "99_complete.done").write_text("complete\n")
        return report
    except Exception as error:
        (output / "failure.json").write_text(json.dumps({"error": str(error)}, indent=2))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit")
    parser.add_argument("--calibration")
    parser.add_argument("--early-stop")
    parser.add_argument("--apply", help="Existing paired features; does not rerun sequencing")
    parser.add_argument("--profile")
    parser.add_argument("--independent", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--psm-fdr", type=float, default=.05)
    parser.add_argument("--peptide-fdr", type=float, default=.05)
    parser.add_argument("--gamma-method", choices=("interval", "suffix"), default="interval")
    parser.add_argument("--interval-margin", type=float, default=.1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=454)
    args = parser.parse_args()
    if args.apply:
        if not args.profile or any((args.fit, args.calibration, args.early_stop)):
            parser.error("--apply requires --profile and no fitting inputs")
        receipt = json.loads(Path(args.apply + ".receipt.json").read_text())
        if receipt["features_sha256"] != _sha256(args.apply):
            raise FDRContractError("Feature receipt mismatch")
        profile = load_profile(args.profile, receipt["runtime_identity"], receipt["scope_id"], args.psm_fdr, args.peptide_fdr)
        print(json.dumps(apply_profile(args.apply, profile, args.output_dir, independent=args.independent), indent=2))
        return
    if not all((args.fit, args.calibration, args.early_stop)) or args.profile or args.independent:
        parser.error("Fitting requires --fit, --early-stop and --calibration")
    print(json.dumps(fit_profile(args.fit, args.calibration, args.output_dir,
        validation_path=args.early_stop, psm_q=args.psm_fdr, peptide_q=args.peptide_fdr, method=args.gamma_method,
        margin=args.interval_margin, epochs=args.epochs, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
