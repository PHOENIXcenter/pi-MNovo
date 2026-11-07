# pi-MNovo

CTC peptide sequencing with Beam20 + PMC, three experts and an observable router.

The main command requires `--model` (denovo, eval or train), `--config` and
`--input`. Optional `--output` sets the prediction TSV or training run directory.
When omitted, a new run directory is created under `outputs/` beside the
`MNovo/` source directory. The command prints the destination; prediction and
evaluation write `predictions.tsv` inside that directory. For prediction or evaluation, copy
[MNovo/config.yaml](MNovo/config.yaml) and set `runtime.model_dir` to the downloaded
checkpoint. The file lists all checkpoint configuration fields, including the
ordered residue dictionary, plus runtime settings. Change `n_beams: 20` to try
another width or set `PMC_enable: false` to disable PMC. Changes to decoding or
peak filters require a matching profile when applying FDR.
Optional Counterpart FDR is post-processing, not a change to sequence selection.

**Version 0.2.1** · [中文使用说明与代码导读](docs/README.zh-CN.md)

Use the isolate checkpoint for microbial isolates and the metaproteome checkpoint
for real-community metaproteomes. Neither checkpoint supplies a cohort-specific fitted CCT profile.

## Repository map

MGF input → accepted spectra → CTC beam and PMC candidates → R1/R2/R3 ranking
and routing → confidence for the selected sequence. Optional CCT filtering
consumes paired evidence; it does not choose the sequence.

| Location | Responsibility |
| --- | --- |
| `MNovo/` | CLI, checkpoint loading and evaluation |
| `MNovo/data/` | MGF input resolution, spectrum accounting and LMDB construction |
| `MNovo/inference/` | Candidate generation, sequence selection, final Score and single-/multi-GPU execution |
| `MNovo/training/` | Training-time CTC loss and distributed training helpers |
| `MNovo/components/`, `MNovo/denovo/` | Transformer layers, spectrum data, decoding and backbone training |
| `MNovo/ranker/` | Candidate features, length and fragment evidence, and routing |
| `MNovo/cct/cct_*.py`, `MNovo/cct/cct_core/` | Paired features, fixed-role fitting, calibration and filtering |
| `scripts/` | Checkpoint and label audits, historical downloader and training smoke test |
| `tests/`, `tests/data/` | Regression tests and synthetic MGF input |
| `docs/`, `wheels/` | Usage and validation notes; native Linux decoder wheel |

## Installation

Linux x86-64, Python 3.10, CUDA 12 and a BF16-capable NVIDIA GPU are required
for the complete sequencing pipeline. The bundled decoder wheel targets
PyTorch 2.5.1; other platforms require rebuilding the native extension.

```bash
conda env create -f environment.yml
conda activate pi-mnovo
pip install -e . --no-deps
```

## Download a checkpoint

Choose a checkpoint for your sample type from the v0.2.0 release assets:

| Sample type | Model checkpoint |
| --- | --- |
| Microbial isolates | [pi-MNovo-isolate.ckpt](https://github.com/PHOENIXcenter/pi-MNovo/releases/download/v0.2.0/pi-MNovo-isolate.ckpt) |
| Community metaproteomes | [pi-MNovo-metaproteome.ckpt](https://github.com/PHOENIXcenter/pi-MNovo/releases/download/v0.2.0/pi-MNovo-metaproteome.ckpt) |

SHA-256 checksums for the two downloads:

```text
b332315f9b76839b8f58fb42b7379dc4626120148ef711d1e28a08355d0f5bb3  pi-MNovo-isolate.ckpt
6497cfa0fb3f7933d4215158288bb8fa720e98c8498860f29ee52bfbdbcd98f5  pi-MNovo-metaproteome.ckpt
```

Use trusted checkpoints and LMDB files only. The bundle must supply the
backbone, rankers, length predictor, router and both final-Score components.
`scripts/download_model.py` targets the historical preprint checkpoint, not the
two manuscript assets. Pass their paths explicitly. The environment file
installs the native decoder and pinned requirements; resolving the dependency
ranges in `pyproject.toml` alone does not reproduce that environment.

## Prediction

```bash
pi-mnovo --model denovo --config MNovo/config.yaml --input sample.mgf --output native.tsv
```

Input accepts MGF files, directories or quoted globs. Whole batches are assigned
to workers and merged in input order. Preserve batch size and precision for
reproduction. Native TSV includes sequence, Score, Route, Status and dataset_index.
**Score is not an FDR or q-value.** Empty predictions remain explicit failures.

The software computes the frozen C1/long-peptide sequence Score
independently of the YAML FDR setting. CCT filtering and its fitted profiles are unchanged.
The Score uses the original confidence head and calibrator shipped with the unified
checkpoint, not the router probability or a replacement CTC score. A new backbone
does not establish calibration validity in a new domain. Missing scoring components
are an error rather than a silently exported NaN. Previous rc1 sequence-only tables
may contain NaN Score values; sequence and recall results remain separate from this
score repair. `MNovoRuntime.score_saved_batch` can score fixed trace predictions
without decoding or changing their sequence, route, or status.
Input-audit sidecars reconcile original, accepted and rejected spectra.

## Optional FDR

Set `runtime.fdr_mode: "on"` and a study-specific `runtime.cct_scope_id` in
your YAML, then run the same prediction command. Supply a trusted, matching
`runtime.cct_profile` to apply it directly. With `cct_profile: null`, MNovo
uses reference-labelled spectra to fit, test and apply a study-specific profile.
MGF `SEQ=` labels are read automatically; an accepted PSM table can be supplied
instead. See the [input contract and one-command workflow](CCT_MODULE.md#one-command-workflow).

FDR is off by default.
Changes to CCT feature definitions or scoring algorithms may invalidate a fitted
profile; validate the calibration again or refit it before reporting FDR results.
Native predictions remain in native.tsv; filtered results go to native.cct/.
Multi-GPU features are merged before one global FDR calculation. Legacy
annotate/filter modes are unsupported. See [FDR workflow](CCT_MODULE.md).

## Evaluation and training

```bash
pi-mnovo --model eval --config MNovo/config.yaml --input labelled.mgf --output predictions.tsv
pi-mnovo --model train --config docs/training_config.example.yaml --input './train/*.mgf' --output training_run
```

Evaluation references use SEQ. CLI recall includes all accepted inputs, even
empty predictions; rejected spectra are reported separately. This is not
automatically a study-specific all-original-input benchmark denominator.
When evaluating an existing LMDB, each record must have an explicit
`reference_sequence`. Rebuild older unlabelled caches from SEQ-labelled MGF;
titles and legacy `pep` values are not evidence of a reference label.

Training YAML selects trainer_accelerator: gpu and trainer_devices: 1 or [0, 1].
train_batch_size is per device; accumulate_grad_batches controls accumulation.
See [training safety](docs/training_safety.md). FDR MLP fitting uses one device
for PSM and CPU for peptide fitting; multi-GPU sequencing/backbone training
does not alter the statistical FDR protocol.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
python tests/test_lightning_training_fix.py 2 /tmp/tiny-ddp.ckpt
TEST_OUTPUT=/tmp/mnovo-cli-test python scripts/check_training_cli.py
```

CPU DDP tests do not replace physical multi-GPU validation.
See [validation](docs/validation.md). Installation does not promote weights or
replace historical results. MIT license; retain NOTICE and CITATION.cff.
