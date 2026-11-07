# Counterpart FDR

Implements FDR-20261027-01, not the removed PEP filter. The inherited CTC
occurrence-offset construction and legacy L-1 feature convention are retained.

## Features

### One-command workflow

Edit the runtime section in the supplied config, then run the usual command:

```bash
pi-mnovo --model denovo --config my-config.yaml --input './runs/*.mgf' --output native.tsv
```

* Set `fdr_mode: "on"` and `cct_scope_id` to your study identifier. A supplied
  `cct_profile` is checked and applied without refitting. A missing or incompatible
  supplied file is an error, not permission to fit another one.
* With `cct_profile: null`, provide partial or complete MGF `SEQ=` labels, or
  set `cct_fit.reference_table` to a UTF-8 tab-separated table of **accepted PSMs**.
  The portable format is exact MGF TITLE and complete modified SEQ in the model's
  token notation. Column names are configurable. The importer also recognizes
  pFind File_Name/Sequence/Modification (exact full title) and MaxQuant Raw file/
  Scan number/Charge/Modified sequence (run, scan and charge). Common supported
  modification encodings are translated; unrecognized encodings require conversion
  to the portable format. Set fixed_carbamidomethyl_c only if confirmed by the
  actual search settings. Do not supply unfiltered search output. The importer
  does not perform database-search FDR filtering. Original MGF files are unchanged;
  labelled copies remain under native.cct-work/labelled_mgf. Unmatched spectra
  still receive predictions; they do not become negative training examples.
* By default the workflow groups charge hypotheses of the same run/scan and
  exact duplicate peak lists, shuffles groups once (seed 454), and partitions
  groups approximately 48/12/20/20 into train/early_stop/calibration/test.
  Fractions refer to groups, not spectrum quotas. For unconventional scan titles
  or biological-sample holdout, supply a predeclared `cct_fit.roles` manifest.
  A held-out spectrum test does not establish performance on independent subjects.
* Auto-fit writes roles.tsv (or uses your manifest), role_counts.tsv,
  fitted/cct_profile.pt and held_out/ under native.cct-work/. Native predictions
  remain in native.tsv even if downstream fitting fails. Rerun with a fresh output
  path after correcting errors. Calibration can fail even when both classes exist:
  do not retry seeds or relax the FDR level to obtain a passing result.
* Full-cohort outputs are native.cct/accepted_psms.tsv, accepted_peptides.tsv and
  summary.tsv. The summary separates estimated FDR from errors / referenced retained
  targets (NA for an empty denominator). Only auto-fit reports a held-out test.
  Full application includes the fitting partitions and is not independent validation.

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
