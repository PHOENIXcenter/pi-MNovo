"""Orchestrate native prediction, optional local fitting and global CCT application."""
import copy
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from MNovo.data.input import LmdbTitleLookup
from . import cct_pipeline as pipeline
from .cct_roles import prepare


ROLES = ("train", "early_stop", "calibration", "test")


def fit_settings(values):
    defaults = dict(roles=None, save_profile=None, seed=454, fractions=[.48, .12, .20, .20], epochs=50, gamma_method="interval",
                    reference_table=None, title_column="TITLE", sequence_column="SEQ", fixed_carbamidomethyl_c=False)
    if values is not None and (not isinstance(values, dict) or values.keys() - defaults.keys()):
        raise ValueError("Invalid runtime.cct_fit mapping; see config.yaml for supported keys.")
    cfg = {**defaults, **(values or {})}
    if type(cfg["fixed_carbamidomethyl_c"]) is not bool:
        raise ValueError("fixed_carbamidomethyl_c must be true or false (use the actual search setting)")
    if cfg["gamma_method"] not in ("interval", "suffix"):
        raise ValueError("cct_fit.gamma_method must be interval or suffix")
    for name in ("roles", "reference_table"):
        if cfg[name] is not None and not Path(cfg[name]).is_file():
            raise ValueError(f"cct_fit.{name}: file not found: {cfg[name]}")
    if type(cfg["seed"]) is not int or not 0 <= cfg["seed"] < 2**32:
        raise ValueError("cct_fit.seed must be a nonnegative 32-bit integer")
    if type(cfg["epochs"]) is not int or not 1 <= cfg["epochs"] <= 50:
        raise ValueError("cct_fit.epochs must be between 1 and 50")
    f = np.asarray(cfg["fractions"], dtype=float)
    if f.shape != (4,) or not (f > 0).all() or not np.isclose(f.sum(), 1, rtol=0, atol=1e-9):
        raise ValueError("cct_fit.fractions needs four positive fractions summing to one")
    return cfg


def search_reference(row, kind, cfg):
    if kind == "generic":
        return row[cfg["title_column"]].strip(), row[cfg["sequence_column"]].strip()
    if kind == "pFind":
        title = row["File_Name"].strip()
        seq = list(row["Sequence"].strip())
        prefix = ""
        mods = {"Oxidation[M]": ("M", "+15.995"), "Carbamidomethyl[C]": ("C", "+57.021"),
                "Deamidated[N]": ("N", "+0.984"), "Deamidated[Q]": ("Q", "+0.984")}
        for item in filter(None, row["Modification"].strip().strip(";").split(";")):
            pos, mod = item.split(",", 1)
            pos = int(pos)
            if pos == 0 and mod in ("Acetyl[ProteinN-term]", "Acetyl[AnyN-term]") and not prefix:
                prefix = "+42.011"
            elif mod in mods and 1 <= pos <= len(seq) and seq[pos-1] == mods[mod][0]:
                seq[pos-1] += mods[mod][1]
            else:
                raise ValueError(f"Unsupported modification {item!r}; supply an accepted TITLE/SEQ TSV with explicit modified sequences.")
        if cfg["fixed_carbamidomethyl_c"]:
            seq = ["C+57.021" if aa == "C" else aa for aa in seq]
        return title, prefix+"".join(seq)
    key = (row["Raw file"].strip(), int(row["Scan number"]), int(row["Charge"]))
    seq = row["Modified sequence"].strip().strip("_")
    for old, new in (("M(ox)", "M+15.995"), ("M(Oxidation (M))", "M+15.995"),
                     ("(Acetyl (Protein N-term))", "+42.011"), ("(ac)", "+42.011")):
        seq = seq.replace(old, new)
    if cfg["fixed_carbamidomethyl_c"]:
        seq = seq.replace("C", "C+57.021")
    return key, seq


def annotate_sources(sources, cfg, out, residues):
    """Join accepted references by exact TITLE. Preserve unknown spectra and original MGF files."""
    source = Path(cfg["reference_table"])
    a, b = cfg["title_column"], cfg["sequence_column"]
    labels = {}
    tokens = re.compile("(?:" + "|".join(map(re.escape, sorted(residues, key=len, reverse=True))) + ")+")
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        cols = set(reader.fieldnames or ())
        if {a,b} <= cols:
            kind = "generic"
        elif {"File_Name","Sequence","Modification"} <= cols:
            kind = "pFind"
        elif {"Raw file","Scan number","Charge","Modified sequence"} <= cols:
            kind = "MaxQuant"
        else:
            raise ValueError(f"Expected accepted-PSM TSV columns {a!r}, {b!r}. Map the columns in cct_fit; use full modified SEQ.")
        for row in reader:
            if str(row.get("Target/Decoy", "target")).lower() == "decoy" or any(
                    row.get(c) == "+" for c in ("Reverse","Potential contaminant")):
                continue
            title, seq = search_reference(row, kind, cfg)
            if not title or not tokens.fullmatch(seq):
                raise ValueError(f"Invalid title or unsupported modified sequence at {source}:{reader.line_num}")
            if labels.setdefault(title, seq).replace("I","L") != seq.replace("I","L"):
                raise ValueError(f"Conflicting accepted sequences for TITLE={title}")
    out.mkdir(parents=True, exist_ok=False)
    result, seen = [], set()
    for i, path in enumerate(sources):
        dest = out / f"{i:05d}" / Path(path).name
        dest.parent.mkdir()
        title = seq = scan = charge = None
        with Path(path).open(encoding="utf-8-sig") as src, dest.open("w", encoding="utf-8") as dst:
            for line in src:
                key, _, val = line.strip().partition("=")
                if key.upper() == "BEGIN IONS":
                    title = seq = scan = charge = None
                elif key.upper() == "TITLE":
                    title = val
                elif key.upper() == "SEQ":
                    seq = val
                elif key.upper() == "SCANS":
                    scan = val
                elif key.upper() == "CHARGE":
                    charge = val.rstrip("+")
                elif key.upper() == "END IONS":
                    if kind == "MaxQuant":
                        m = re.fullmatch(r"(.+)\.(\d+)\.\2\.(\d+)(?:\..*)?", title or "")
                        if m and (charge or "").isdigit() and int(m[3]) == int(charge) and (not scan or scan == m[2]):
                            title = (m[1], int(m[2]), int(m[3]))
                        elif not m and (scan or "").isdigit() and (charge or "").isdigit():
                            title = (Path(path).stem, int(scan), int(charge))
                    if title not in labels:
                        dst.write(line)
                        continue
                    if title in seen or (seq is not None and seq.replace("I","L") != labels[title].replace("I","L")):
                        raise ValueError(f"Ambiguous TITLE or conflicting existing SEQ: {title}")
                    seen.add(title)
                    if seq is None:
                        dst.write(f"SEQ={labels[title]}\n")
                dst.write(line)
        result.append(dest)
    if not seen:
        raise ValueError("No exact TITLE match between MGF and reference table. Check run names, scans and charges.")
    print(f"CCT references: matched {len(seen)} of {len(labels)} distinct accepted titles.", flush=True)
    return result


def make_roles(ids, lmdb, cfg, output):
    # Keep all charge hypotheses of a scan and exact duplicate peak lists together.
    # 用连通分组防止同一个扫描或重复峰列表跨集合。
    parent, owner, scans = {}, {}, {}
    def root(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    with LmdbTitleLookup(lmdb) as lookup:
        for idx in sorted(map(int, ids)):
            record = lookup.spectrum(idx)
            title = str(record.get("title", ""))
            m = re.fullmatch(r"(.+)\.(\d+)\.\2\.\d+(?:\..*)?", title)
            if m:
                scan = f"{m[1]}:{int(m[2])}"
            elif record.get("source_run") and str(record.get("source_scan","")).isdigit():
                scan = f"{record['source_run']}:{int(record['source_scan'])}"
            else:
                raise ValueError(f"Cannot infer physical run and scan from TITLE={title!r}. Supply cct_fit.roles with pair_id, group_id, role.")
            scans[idx] = scan
            peaks = np.column_stack((record["mz_array"], record["intensity_array"])).astype("<f8")
            digest = hashlib.sha256(peaks.tobytes()).hexdigest()
            parent[idx] = idx
            for key in (("scan", scan), ("peaks", digest)):
                parent[root(idx)] = root(owner.setdefault(key, idx))
    groups = {}
    for idx in parent:
        groups.setdefault(root(idx), []).append(idx)
    # Sort before shuffling; never retry a split based on validation/test outcomes.
    groups = list(groups.values())
    if len(groups) < 4:
        raise ValueError(f"Only {len(groups)} independent scan groups; four nonempty roles are required. Add labelled spectra.")
    order = np.random.default_rng(cfg["seed"]).permutation(len(groups))
    cuts = np.rint(np.cumsum(cfg["fractions"][:-1])*len(groups)).astype(int)
    if np.min(np.diff([0, *cuts, len(groups)])) < 1:
        raise ValueError("Too few independent scans for the configured fractions. Add data or supply a predeclared roles TSV.")
    with output.open("x", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle, delimiter="\t")
        w.writerow(("pair_id", "group_id", "role"))
        for role, indexes in zip(ROLES, np.split(order, cuts)):
            for j in indexes:
                group_id = min(scans[idx] for idx in groups[j])
                for idx in groups[j]:
                    w.writerow((idx, group_id, role))


def summary(reports, output):
    rows = []
    for scope, report in reports:
        for stage, level in (("psm","PSM"), ("peptide","Distinct peptide")):
            m = report[stage + "_metrics"]
            pct = lambda x: "NA" if x is None else f"{100*x:.2f}"
            rows.append([scope, level, m["retained"], m["reference_unknown"], m["reference_known"],
                         m["reference_wrong"], pct(m["estimated_fdr"]), pct(m["reference_fdp"])])
    table = pd.DataFrame(rows, columns=["Scope","Level","Retained","Without reference","With reference",
                                        "Incorrect among referenced","Estimated FDR (%)","Observed error among referenced (%)"])
    table.to_csv(output, sep="\t", index=False)
    print(table.to_string(index=False), flush=True)


def run_workflow(args, lmdb, files, total, predict):
    """Predict once. Supply a trusted calibrated profile or fit on fixed roles, never on test."""
    output = Path(args.output)
    work, dest = output.with_suffix(".cct-work"), output.with_suffix(".cct")
    features = Path(args.cct_features_output or output.with_suffix(".cct_features.csv"))
    stage = "prediction and feature export"
    try:
        print(f"CCT: {stage}", flush=True)
        if dest.exists():
            raise FileExistsError(dest)
        if args.cct_profile:
            stage = "prediction and application of the supplied profile"
            count = predict(args, lmdb, files, total)
            if not count:
                raise ValueError("No accepted input spectra. Inspect the input audit.")
            report = json.loads((dest / "report.json").read_text())
            summary([("Full application", report)], dest / "summary.tsv")
            return count
        cfg = args.cct_fit
        native = copy.copy(args)
        native.fdr_mode, native.cct_features_output = "off", str(features)
        count = predict(native, lmdb, files, total)
        if not count:
            raise ValueError("No accepted input spectra. Inspect the input audit.")
        work.mkdir(parents=True, exist_ok=True)
        stage = "reference checks and group assignment"
        print(f"CCT: {stage}", flush=True)
        data = pd.read_csv(features, usecols=["pair_id","valid","is_decoy","denovo_label"])
        target = data.loc[data.is_decoy.eq(0)]
        valid = data.groupby("pair_id").valid.all()
        usable = target[target.pair_id.isin(valid.index[valid])]
        if not usable.denovo_label.isin([0,1]).any():
            raise ValueError("No usable reference-labelled predictions. Provide MGF SEQ= labels or an accepted reference TSV; unlabelled spectra are not negatives.")
        manifest = cfg["roles"]
        if not manifest:
            manifest = work / "roles.tsv"
            make_roles(target.pair_id, lmdb, cfg, manifest)
        roles = work / "roles"
        prepare(features, manifest, roles)
        stats = []
        for name in ROLES:
            part = pd.read_csv(roles / (name+".csv"), usecols=["pair_id","is_decoy","denovo_label"])
            ref = part.loc[part.is_decoy.eq(0) & part.pair_id.isin(usable.pair_id), "denovo_label"]
            stats.append(dict(role=name, valid=len(ref), correct=int(ref.eq(1).sum()), incorrect=int(ref.eq(0).sum())))
        pd.DataFrame(stats).to_csv(work/"role_counts.tsv", sep="\t", index=False)
        for s in stats:
            if (s["role"] != "test" and min(s["correct"], s["incorrect"]) == 0) or s["correct"] + s["incorrect"] == 0:
                raise ValueError(f"{s['role']}: {s['valid']} valid spectra, {s['correct']} reference-correct, {s['incorrect']} reference-incorrect. Insufficient reference evidence; add labelled spectra.")
        stage = "profile fitting and calibration (check label diversity and calibration support)"
        print(pd.DataFrame(stats).to_string(index=False), flush=True)
        fitted = work/"fitted"
        pipeline.fit_profile(roles/"train.csv", roles/"calibration.csv", fitted, validation_path=roles/"early_stop.csv",
                             psm_q=args.psm_fdr, peptide_q=args.peptide_fdr, seed=cfg["seed"], epochs=cfg["epochs"], method=cfg["gamma_method"])
        receipt = json.loads(Path(str(features)+".receipt.json").read_text())
        profile = pipeline.load_profile(fitted/"cct_profile.pt", receipt["runtime_identity"], args.cct_scope_id, args.psm_fdr, args.peptide_fdr)
        reports = []
        for role in ROLES:
            stage = f"filtering partition: {role}"
            print(f"CCT: {stage}", flush=True)
            report = pipeline.apply_profile(roles / (role + ".csv"), profile,
                                            work / ("report_" + role), independent=role == "test")
            reports.append((role, report))
        stage = "full application"
        print(f"CCT: {stage}", flush=True)
        full = pipeline.apply_profile(features, profile, dest)
        summary([*reports, ("Full application",full)], dest/"summary.tsv")
        return count
    except pipeline.CalibrationFailure:
        raise
    except Exception as exc:
        message = f"CCT failed during {stage}: {exc}"
        if output.is_file():
            message += " Native prediction completed."
        if work.is_dir():
            (work/"failure.json").write_text(json.dumps({"stage":stage, "message":message}, indent=2))
        print(message, file=sys.stderr)
        raise RuntimeError(message) from exc
