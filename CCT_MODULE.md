# Counterpart FDR

Implements FDR-20261027-01, not the removed PEP filter. The inherited CTC
occurrence-offset construction and legacy L-1 feature convention are retained.

## Features

There are 33 features. Only a verified constant-input Spearman NaN is set to
zero with audit flags; other invalid pairs are excluded. Labels belong to the
spectrum/PSM, not peptide membership in a protein database. Matching preserves
modifications and merges I/L. Unknown references remain unknown on both views.

```bash
# In the YAML runtime section set fdr_mode: "off", cct_scope_id: study-v1,
# and cct_features_output: pairs.csv.
pi-mnovo --model denovo --config MNovo/config.yaml --input mixed.mgf --output native.tsv
```

## Fixed roles

Prepare a TSV with pair_id, group_id, role. Pair IDs are exported dataset_index
values. Group IDs join hypotheses of one physical scan and known exact
duplicates. Roles: train, early_stop, calibration, test. The adopted study used
approximately 48/12/20/20. Physical groups cannot cross roles; predicted peptides
on different scans may. No implicit resampling occurs.

```bash
python -m MNovo.cct.cct_roles --features pairs.csv --manifest roles.tsv --output-dir roles
pi-mnovo-fdr --fit roles/train.csv --early-stop roles/early_stop.csv \
  --calibration roles/calibration.csv --output-dir fitted --psm-fdr .05 --peptide-fdr .05
```

Fitting never accepts test inputs. Only known references supervise MLPs and gamma.
Strict/relaxed models use train-only scaling, AdamW (1e-3, weight decay 1e-5),
batch 1024, up to 50 epochs, validation-AUC early stopping (patience 8), seed 454.
Scores average the two models. Peptide fitting keeps role identities and seed
but uses separate models and gamma.

Gamma uses complete score ties and the maximum FDP/(C/T) ratio throughout the
calibration interval. No clipping or automatic fallback rescues failures.
The calibrated lower bound is a score floor, not an upper rejection boundary.
Application recounts T and C in its population.

Either passing PSM side restores the entire pair. Within each role, each full
modified I/L-equivalent peptide keeps its highest-scoring representative pair,
with deterministic pair/side tie breaking. The label comes from that spectrum.
Target/Counterpart features never come from different representative scans.

## Validation and application

```bash
pi-mnovo-fdr --apply roles/test.csv --profile fitted/cct_profile.pt \
  --independent --output-dir test_result --psm-fdr .05 --peptide-fdr .05
pi-mnovo-fdr --apply pairs.csv --profile fitted/cct_profile.pt \
  --output-dir full_result --psm-fdr .05 --peptide-fdr .05
```

Independent mode rejects fitted/calibration physical groups. Full application
includes those groups and is not independent validation. Reference FDP is
wrong retained targets / known retained targets; unknowns are separate.
Peptides are unique within an application. Per-RAW reporting uses pooled
thresholds, never retuned RAW cutoffs. Estimated FDR is not a guarantee on
unknown labels or every RAW.

Profiles bind inference components and configuration, scope, features and FDR levels.
An rc2 profile with the same schema and inference configuration can be used
without rewriting it. Its historical Python source digests are not consulted.
Load only trusted profiles. Failures preserve diagnostics without overwriting
outputs. Reference alignment and prior test exposure must be documented.
