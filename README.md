# π-MNovo

π-MNovo predicts peptide sequences from tandem mass spectra. Use it for **de novo sequencing** without reference labels, **evaluation** against known sequences, or optional **Counterpart FDR (CCT)** filtering. Backbone training is a separate task. This guide describes the current **local development branch based on v0.2.1**, not the interface of the published v0.2.1 tag.

**中文快速开始：** 按第 1 节安装，第 2 节下载一个适合样本的权重并复制配置。只做预测时，直接用第 3 节的命令，不需要搜库结果。如果要做 FDR，第 4 节有三条路径：已有 profile、MGF 自带标签、另有已接受的搜库结果。三种情况使用同一条预测命令，不需要自己串联脚本。各输出文件见第 5 节，报错见第 7 节。

## 1. Install from the repository root

The complete pipeline uses Linux x86-64, Python 3.10, CUDA 12 and a BF16-capable NVIDIA GPU. The bundled CTC decoder wheel targets CPython 3.10 and PyTorch 2.5.1; it is not a Windows wheel. From the directory containing this README:

```bash
conda env create -f environment.yml
conda activate pi-mnovo
python -m pip install -e . --no-deps
python -m pip check
pi-mnovo --help
```

## 2. Choose a checkpoint and prepare your config

Download **one** of the following v0.2.0 release assets. Keep it on disk; you do not need to unpack it. Use only trusted checkpoints, profiles and LMDBs.

| Sample type | Download | Expected SHA-256 |
| --- | --- | --- |
| Microbial isolates | [pi-MNovo-isolate.ckpt](https://github.com/PHOENIXcenter/pi-MNovo/releases/download/v0.2.0/pi-MNovo-isolate.ckpt) | `b332315f9b76839b8f58fb42b7379dc4626120148ef711d1e28a08355d0f5bb3` |
| Community metaproteomes | [pi-MNovo-metaproteome.ckpt](https://github.com/PHOENIXcenter/pi-MNovo/releases/download/v0.2.0/pi-MNovo-metaproteome.ckpt) | `6497cfa0fb3f7933d4215158288bb8fa720e98c8498860f29ee52bfbdbcd98f5` |

For example, place the isolate checkpoint in the repository root, then run:

```bash
sha256sum pi-MNovo-isolate.ckpt
cp MNovo/config.yaml my-config.yaml
```

Compare the digest with the table. In `my-config.yaml`, set **`runtime.model_dir` to the checkpoint file**, e.g. `./pi-MNovo-isolate.ckpt`. Relative paths are relative to the **working directory**, not the YAML file. Edit the existing keys in this copy; do not replace the entire config with a fragment.

| Setting | When to change it |
| --- | --- |
| `n_beams: 20`, `PMC_enable: true`, `mass_control_tol: 0.1` | Beam width, precursor-mass-constrained candidate generation and its mass tolerance (Da). |
| `n_peaks`, `min_mz`, `max_mz`, `min_intensity`, `remove_precursor_tol` | Spectrum preprocessing. Keep the supplied values to reproduce default inference. |
| `residues` and architecture settings | Document the checkpoint's token vocabulary, masses and architecture; these must match the checkpoint, including token order. |
| `runtime.batch_size`, `runtime.n_workers`, `runtime.ctc_processes`, `runtime.gpus` | Runtime resources; e.g. `gpus: "0,1"` selects two GPUs. Keep batch size and precision fixed for reproduction. |
| `runtime.input_format` | Default: `mgf`. Set to `lmdb` only for a compatible existing index. |

## 3. Predict sequences

Supply an MGF file, directory or **quoted** glob. MGF blocks should contain precursor m/z (`PEPMASS`), a single supported `CHARGE` and fragment peaks. Preserve original spectrum titles. **Reference sequences are not required for ordinary prediction.**

```bash
pi-mnovo --model denovo --config my-config.yaml --input './runs/*.mgf' --output native.tsv
```

Here `--model` selects the **task**, not the weight file. Only `--model`, `--config` and `--input` are required. Without `--output`, a fresh directory is created under `outputs/` beside the source package and the destination is printed. All output names below assume you chose `native.tsv`; otherwise use your own output stem.

For predictions only, leave `runtime.fdr_mode: "off"`. Read `native.tsv` and the input audit. The model generates candidates, ranks and routes them, then scores the selected sequence. **Score is not an FDR or q-value.** CCT filters predictions; it does not replace the selected sequence.

## 4. Optional: run FDR filtering in the same command

Edit the existing **runtime** section, then run the command from step 3 with a **fresh output path**. Set `fdr_mode: "on"` (quoted), set `cct_scope_id` to your study identifier, and choose **one** path:

| What you have | What to set | What happens |
| --- | --- | --- |
| A trusted profile calibrated for this scope and inference setup | Set `cct_profile` to its `.pt` path and use its matching scope ID and FDR levels. | Predict and filter. No fitting, labels or holdout required. An incompatible profile is an error, not permission to silently refit. |
| MGF with reference labels | Leave `cct_profile: null` and `cct_fit.reference_table: null`. Labelled blocks contain `SEQ=<full modified reference sequence>`. | Read labels, export paired features, assign roles, fit, calibrate, test on held-out scans, then filter the full cohort. |
| MGF plus accepted search PSMs | Leave `cct_profile: null`; set `cct_fit.reference_table` to a UTF-8 TSV of **already accepted** PSMs. | Match labels, write annotated MGF copies without changing the originals, then follow the same fit/test/filter workflow. |

**Which columns?** The portable TSV format uses the exact MGF title and the complete modified reference sequence in checkpoint token notation, for example:

```tsv
TITLE	SEQ
runA.123.123.2	AM+15.995K
```

Different header names are fine: set `cct_fit.title_column` and `sequence_column`. Recognized native headers include pFind `File_Name/Sequence/Modification` (exact full-title match) and MaxQuant `Raw file/Scan number/Charge/Modified sequence` (run, scan and charge). Only supported modification encodings are converted. Enable `fixed_carbamidomethyl_c` only if this was the actual native search setting. A protein FASTA, an unfiltered list of candidate PSMs, or an unmodified sequence column that omits known modifications is **not** a substitute for accepted reference PSMs. Unlabelled spectra remain in the application; unknown labels are not counted as incorrect predictions.

**What gets fitted?** Not the sequencing backbone. CCT builds target/counterpart views of the same prediction, with 33 features per view. Known references supervise the strict and relaxed scoring networks; their scores are averaged. The PSM and peptide stages fit separate models and calibrations. Peptide filtering restores a pair when either PSM side passes and chooses one representative per modified, I/L-equivalent sequence.

**Default split.** `cct_fit.roles: null` groups charge hypotheses of one run/scan and exact duplicate peak lists before splitting. The default fractions are 48% training, 12% early stopping, 20% calibration and 20% held-out testing **by scan group**, with seed 454, up to 50 epochs and interval calibration. It does not split by biological subject. To impose your own split, provide a TSV with `pair_id` (exported dataset index), `group_id` and `role` (`train`, `early_stop`, `calibration`, `test`); cover every exported pair exactly once and keep each physical group in one role.

**Default targets:** `runtime.psm_fdr: 0.05` and `runtime.peptide_fdr: 0.05`. A supplied profile must match the requested levels, scope, model components and inference configuration. Changing decoding, preprocessing, CCT features or scoring may require revalidation or recalibration. The two downloadable model checkpoints do **not** include a universal FDR profile.

**Advanced: reuse exported features without decoding again.** Skip this for the one-command workflow. To export features only, set `fdr_mode: "off"`, `cct_scope_id` and `cct_features_output: pairs.csv` before prediction. Keep the receipt alongside the CSV. With a predeclared `roles.tsv`, the equivalent standalone fitting and application steps are:

```bash
python -m MNovo.cct.cct_roles --features pairs.csv --manifest roles.tsv --output-dir roles
pi-mnovo-fdr --fit roles/train.csv --early-stop roles/early_stop.csv --calibration roles/calibration.csv --output-dir fitted --psm-fdr .05 --peptide-fdr .05
pi-mnovo-fdr --apply roles/test.csv --profile fitted/cct_profile.pt --independent --output-dir test_result --psm-fdr .05 --peptide-fdr .05
pi-mnovo-fdr --apply pairs.csv --profile fitted/cct_profile.pt --output-dir full_result --psm-fdr .05 --peptide-fdr .05
```

## 5. Find and interpret your results

| File (for `--output native.tsv`) | What it contains |
| --- | --- |
| `native.tsv` | Unfiltered predictions for accepted inputs: spectrum title, scan, charge, sequence, modification, Score, status, route and dataset index. An empty prediction remains explicit. |
| `native.input-audit.tsv`, `native.rejected.tsv`, `native.input-summary.json` | For MGF input: per-spectrum accounting, rejected inputs and reasons, and totals. Check these when output rows differ from input spectra. |
| `native.cct_features.csv` and its `.receipt.json` | Paired CCT features and provenance. Keep them together. The feature path can be overridden. |
| `native.cct-work/labelled_mgf/` | Annotated copies when a reference table was supplied. |
| `native.cct-work/roles.tsv`, `roles/`, `role_counts.tsv` | Auto-fit: generated assignments (if automatic), role-specific feature files with receipts, and valid/correct/incorrect counts by role. Paths in this row are relative to the work directory after the first entry. |
| `native.cct-work/fitted/cct_profile.pt`, `native.cct-work/held_out/` | Reusable fitted profile and separate held-out test results. Created only during auto-fit. |
| `native.cct/accepted_psms.tsv` | Retained peptide–spectrum matches at the PSM threshold. |
| `native.cct/accepted_peptides.tsv` | Retained distinct peptides at the peptide threshold; this is **not** just a deduplicated copy of the accepted PSM table. |
| `native.cct/summary.tsv`, `native.cct/report.json` | English summary and detailed machine-readable full-application report. |

The summary reports retained counts, counts with/without reference, incorrect referenced targets, estimated FDR, and observed error **among retained targets with a reference**. A missing denominator is `NA`, not 0%. An auto-fit run reports both held-out test and full application; the latter includes fitting partitions and is not independent validation. A supplied-profile run reports full application only.

## 6. Evaluate or train instead of predicting

**Evaluation.** Use labelled MGF (`SEQ=`) and the same inference YAML:

```bash
pi-mnovo --model eval --config my-config.yaml --input labelled.mgf --output eval.tsv
```

Read `eval.metrics.json` for evaluation metrics and `eval.tsv` for predictions. Recall includes accepted inputs with empty predictions; rejected inputs are reported separately. For LMDB evaluation, records need explicit `reference_sequence`; a title is never a reference label.

**Backbone training** is optional and is not how you obtain a CCT profile. Copy the training template, provide labelled training and validation MGFs, and set `runtime.validation_input`. For fine-tuning set `runtime.checkpoint` to a compatible **backbone** checkpoint. To initialize from scratch explicitly set `train_from_scratch: true`.

```bash
cp docs/training_config.example.yaml my-training.yaml
# Edit my-training.yaml before running:
pi-mnovo --model train --config my-training.yaml --input './train/*.mgf' --output training_run
```

The training directory contains `train.log` and, when checkpoint saving is enabled, `checkpoints/`. Set `trainer_devices` in the training YAML for distributed training. Batch size is per device. CTC loss excludes infeasible or nonfinite samples; inspect `train/ctc_seen`, `train/ctc_effective` and `train/ctc_excluded`. Evaluation metrics do not drop samples merely because a CTC target is infeasible.

## 7. Troubleshooting

| Symptom | What to do |
| --- | --- |
| No profile and no usable reference-labelled predictions | Check that MGF blocks contain real `SEQ=` labels, or supply an accepted PSM table. Without labels or a matching profile, use FDR off. |
| No exact title match, conflicting labels or unrecognized modification | Compare one MGF title and precursor charge with its accepted search row. Correct the run mapping or use a portable TSV with explicit modified sequences. Do not strip modifications to force a match. |
| Physical scans cannot be grouped | Supply run/scan metadata or a verified `cct_fit.roles` manifest that keeps duplicate spectra together. |
| Too few correct/incorrect examples or calibration cannot estimate a threshold | Inspect `role_counts.tsv` and the failure message. Add suitable labelled data; do not select a new seed or relax the threshold just to make the run pass. Completed native predictions remain available. |
| Incompatible profile | Check scope, checkpoint, inference settings and FDR levels against the calibration run. Do not rename a scope just to bypass a mismatch. |
| Output already exists or is not writable | Use a fresh `--output` path in a writable directory. |
| Native decoder will not import | Check OS, Python and PyTorch against the bundled wheel and install via `environment.yml`. |

## Repository and checks

| Location | Purpose |
| --- | --- |
| `environment.yml`, `requirements.txt`, `requirements-dev.txt`, `wheels/` | Environment definition, pinned runtime dependencies, developer tools and Linux decoder wheel. |
| `MNovo/config.yaml`, `docs/training_config.example.yaml` | Editable inference/evaluation and backbone-training templates. |
| `MNovo/cli.py`, `MNovo/data/` | Task entry point, MGF parsing, indexing and input accounting. |
| `MNovo/components/`, `MNovo/denovo/`, `MNovo/training/` | Model layers, decoder, backbone training and CTC loss. |
| `MNovo/inference/`, `MNovo/ranker/`, `MNovo/evaluation.py` | Candidate generation, ranking/routing, sequence confidence and evaluation. |
| `MNovo/cct/` | CCT input preparation, scoring, profile fitting, held-out testing and filtering. |
| `scripts/`, `tests/` | Verification utilities and regression fixtures. The historical downloader is not the source of the two weights above. |
| `MODEL_CARD.md`, `docs/validation.md`, `LICENSE`, `NOTICE`, `CITATION.cff` | Model information, historical validation record, license, credits and citation metadata. |

To run the regression suite after installation:

```bash
pip install -r requirements-dev.txt
pytest -q
```
