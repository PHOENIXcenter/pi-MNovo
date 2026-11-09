# π-MNovo v0.2.2

π-MNovo predicts peptide sequences from tandem mass spectra of microbial isolates and community metaproteomes. Use it to sequence peptides, evaluate predictions against reference sequences, and filter results at the spectrum and peptide levels.

**中文快速开始：** 从仓库根目录运行以下命令。使用默认单菌模型时，将权重下载到根目录即可开始预测。第 3–6 节依次介绍预测、评估、训练和 FDR 过滤。

## 1. Install

Run these commands from the repository root on Linux x86-64 with Python 3.10, CUDA 12 and a BF16-capable NVIDIA GPU. The environment includes PyTorch 2.5.1 and the CTC decoder.

```bash
conda env create -f environment.yml
conda activate pi-mnovo
python -m pip install -e . --no-deps
pi-mnovo --help
```

## 2. Download a model

Choose the checkpoint for your samples:

- **Microbial isolates:** [pi-MNovo-isolate.ckpt](https://github.com/PHOENIXcenter/pi-MNovo/releases/download/v0.2.0/pi-MNovo-isolate.ckpt)
- **Community metaproteomes:** [pi-MNovo-metaproteome.ckpt](https://github.com/PHOENIXcenter/pi-MNovo/releases/download/v0.2.0/pi-MNovo-metaproteome.ckpt)

Place `pi-MNovo-isolate.ckpt` in the repository root to use the default [MNovo/config.yaml](MNovo/config.yaml) directly. For community samples, set `runtime.model_dir` to `./pi-MNovo-metaproteome.ckpt`. If you store the checkpoint elsewhere, set `runtime.model_dir` to its path.

## 3. Predict peptide sequences

Run the included [two-spectrum example](examples/tiny.mgf) with the isolate checkpoint and the default configuration. Each spectrum block should contain a precursor m/z (`PEPMASS`), charge (`CHARGE`) and fragment peaks. Retain the original spectrum titles to link predictions to scans.

```bash
pi-mnovo --model denovo \
  --config MNovo/config.yaml \
  --input examples/tiny.mgf \
  --output results/denovo.tsv
```

**Output:** `results/denovo.tsv` contains one row per accepted spectrum. The terminal reports input counts and rejection reasons. Preview of the [complete 12-column TSV](examples/expected/denovo.tsv):

| Scan_No | Charge | Sequence | Score | Modification |
| ---: | ---: | --- | ---: | --- |
| 3988 | 2 | HGNLAENAK | 0.9241 | |
| 4508 | 2 | AGSNM+15.995EVSR | 0.8499 | 5,Oxidation[M]; |

`Score` is the model's confidence score for the selected sequence. An empty modification field means no modification was reported.

**Other inputs:** `--input` accepts an MGF file, a directory or a quoted glob. Omit `--output` to create a fresh run under `outputs/` in the code directory. The program prints the path.

## 4. Evaluate predictions

Use the same checkpoint and inference configuration as for prediction.

```bash
pi-mnovo --model eval \
  --config MNovo/config.yaml \
  --input examples/tiny.mgf \
  --output results/eval.tsv
```

**Example result:** 2 of 2 reference peptides recovered (`peptide_recall: 1.0`).

- `results/eval.tsv`: per-spectrum predictions, identical to the prediction example above.
- `results/eval.metrics.json`: evaluation metrics. Compare with the [expected JSON](examples/expected/eval.metrics.json).

Recall includes all accepted inputs. The input audit lists rejected spectra separately. To evaluate an existing LMDB, set `runtime.input_format: lmdb` and supply records with an explicit `reference_sequence`.

## 5. Train a backbone

For training on your own dataset, prepare separate labelled training and validation MGF files and copy the training template:

```bash
cp docs/training_config.example.yaml my-training.yaml
```

Set `runtime.validation_input` to the validation data. For fine-tuning, set `runtime.checkpoint` to a compatible backbone checkpoint. For training from scratch, set `train_from_scratch: true`. Start training with:

```bash
pi-mnovo --model train \
  --config my-training.yaml \
  --input './train/*.mgf' \
  --output training_run
```

Read `training_run/train.log` for progress and `training_run/checkpoints/` for saved weights.

## 6. Filter predictions with CCT

### Configure the profile

Set `runtime.cct_scope_id` to your study identifier and choose one of the following options. The default FDR target is 5% at both the PSM and peptide levels.

**Use an existing profile.** Set `runtime.cct_profile` to your fitted `cct_profile.pt`. Use the study identifier, checkpoint, inference settings and FDR levels used for calibration. π-MNovo applies the profile to the full set of predictions.

**To fit a new FDR profile, provide enough reference-labelled spectra to assess both correct and incorrect de novo predictions. Calibration also needs sufficient counterpart evidence in the score range used to estimate the error rate.**

**Fit a profile from labelled MGF.** Leave `runtime.cct_profile: null`. Include the full modified reference peptide in the `SEQ=` field of each labelled spectrum. π-MNovo predicts the sequences, groups the scans, fits and calibrates a profile, evaluates it on held-out scans and filters the full dataset. Unlabelled spectra are included in the full application.

**Fit a profile from database-search results.** Set `runtime.cct_profile: null` and point `runtime.cct_fit.reference_table` to a tab-separated file of accepted peptide–spectrum matches (PSMs). A portable input file looks like this:

```tsv
TITLE	SEQ
FAIMS_4802_DDA_75min_300nL_LLY_singleStrain_ATCC17744_F1_241014.3988.3988.2	HGNLAENAK
```

Use the exact MGF title and encode the complete modified sequence with the tokens listed in the config. For a table with different column names, set `title_column` and `sequence_column` under `runtime.cct_fit`. The importer also recognizes pFind tables with `File_Name`, `Sequence` and `Modification`, and MaxQuant tables with `Raw file`, `Scan number`, `Charge` and `Modified sequence`. It writes labelled MGF copies and continues with profile fitting.

**Fitting partitions.** Fitting requires enough labelled scan groups for four partitions. Automatic fitting assigns scan groups to training (48%), early stopping (12%), calibration (20%) and held-out testing (20%). Charge hypotheses for the same run and scan and exact duplicate peak lists stay together. To supply your own assignments, set `runtime.cct_fit.roles` to a TSV with `pair_id` (exported dataset index), `group_id` and `role`. Use the role names `train`, `early_stop`, `calibration` and `test`.

To keep the fitted profile for reuse, set `runtime.cct_fit.save_profile` before prediction.

### Run filtering

Place your study MGF files in `runs/`. Run prediction and CCT filtering together with a fresh output path:

```bash
pi-mnovo --model denovo \
  --config MNovo/config.yaml \
  --input './runs/*.mgf' \
  --output results/native.tsv \
  --fdr_mode on
```

The optional `--fdr_mode on|off` overrides `runtime.fdr_mode` for this run. Omit the flag to use the YAML setting (default: `"off"`).

### Read the outputs

The command writes **four result tables** in `results/`:

| File | What it contains |
| --- | --- |
| `native.tsv` | All predictions before FDR filtering. |
| `native.accepted_psms.tsv` | One retained target peptide–spectrum match per row. |
| `native.accepted_peptides.tsv` | One retained peptide representative per row after peptide-level filtering. |
| `native.fdr_summary.tsv` | PSM and peptide counts, estimated FDR, and observed error among entries with a reference, by scope. |

In the PSM and peptide tables, `pair_id` identifies the spectrum by its native `dataset_index`, and `score` is the CCT score. The peptide table retains one representative per I/L-equivalent `peptide_key`. The summary reports the estimated FDR and the observed error among retained predictions with known references.

When fitting a profile, the summary covers `train`, `early_stop`, `calibration`, `test` and `Full application`. The frozen profile is applied to each partition separately; only `test` is held out from fitting and calibration. Full application filters the complete input together and deduplicates peptides across it. When loading an existing profile, the summary reports the current full application. Rates with an empty denominator are shown as `NA`.

## Project files

- `environment.yml`, `requirements.txt`, `wheels/`: runtime environment, Python dependencies and decoder wheel.
- `MNovo/config.yaml`, `docs/training_config.example.yaml`: prediction/evaluation and backbone-training templates.
- `MNovo/data/`, `MNovo/inference/`, `MNovo/ranker/`: spectrum input, candidate generation, ranking and sequence scoring.
- `MNovo/cct/`: reference preparation, profile fitting, calibration and FDR filtering.
- `MNovo/components/`, `MNovo/denovo/`, `MNovo/training/`: model layers, decoding and backbone training.
- `MODEL_CARD.md`, `CITATION.cff`, `LICENSE`, `NOTICE`: model description, citation, license and acknowledgments.
- `tests/`: regression tests. Install `requirements-dev.txt` and run `pytest -q` to execute the suite.
