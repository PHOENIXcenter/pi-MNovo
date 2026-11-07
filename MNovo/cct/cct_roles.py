# Assign predeclared physical groups to roles; do not resample spectra here.
# 按预先定义的物理分组分配角色，不在此重新抽样。
"""Attach a predeclared scan-disjoint manifest; never resample or split here."""
import argparse
import json
from pathlib import Path
import pandas as pd
from MNovo.cct.cct_pipeline import SCHEMA, validate_roles
from MNovo.cct.cct_scoring import _sha256


def prepare(features, manifest, output):
    output = Path(output)
    receipt = json.loads(Path(str(features) + ".receipt.json").read_text())
    if receipt["schema"] != SCHEMA or receipt["features_sha256"] != _sha256(features):
        raise ValueError("Feature receipt mismatch")
    rows = pd.read_csv(features)
    roles = pd.read_csv(manifest, sep="\t", dtype={"group_id": str})
    required = {"pair_id", "group_id", "role"}
    if not required.issubset(roles) or roles[list(required)].isna().any().any() or roles.pair_id.duplicated().any():
        raise ValueError("Manifest needs one pair_id/group_id/role per spectrum")
    names = {"train", "early_stop", "calibration", "test"}
    if set(roles.role) != names or not roles.groupby("group_id").role.nunique().eq(1).all():
        raise ValueError("Manifest must contain four scan-disjoint roles")
    if set(rows.pair_id) != set(roles.pair_id):
        raise ValueError("Manifest must cover all exported pairs exactly")
    if "group_id" in rows or "role" in rows:
        raise ValueError("Feature table already has role assignments")
    if (set(rows) & set(roles)) - {"pair_id"}:
        raise ValueError("Manifest metadata columns collide with feature columns")
    rows = rows.merge(roles, on="pair_id", validate="many_to_one", sort=False)
    parts = {name: rows.loc[rows.role.eq(name)].copy() for name in sorted(names)}
    validate_roles(parts)
    output.mkdir(parents=True, exist_ok=False)
    for name, part in parts.items():
        path = output / (name + ".csv")
        part.to_csv(path, index=False)
        audit = dict(receipt, features_sha256=_sha256(path), role=name,
                     manifest_sha256=_sha256(manifest), parent_features_sha256=_sha256(features))
        pair_valid = part.assign(_valid=part.valid.astype(str).str.lower().eq("true")).groupby("pair_id")._valid.all()
        audit.update(complete_pairs=int(pair_valid.sum()), invalid_pairs=int((~pair_valid).sum()),
                     labelled_pairs=int(part.loc[part.is_decoy.eq(0), "denovo_label"].isin([0, 1]).sum()))
        Path(str(path) + ".receipt.json").write_text(json.dumps(audit, indent=2))
    (output / "manifest.tsv").write_text(roles.to_csv(sep="\t", index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    prepare(args.features, args.manifest, args.output_dir)
