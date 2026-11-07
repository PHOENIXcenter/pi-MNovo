# Training target applicability and CTC safety

Training target applicability is not annotation quality. Original spectra and
evaluation membership must not be deleted when a reference cannot be aligned
by a particular decoder. `MNovo.training.training_ctc.training_target_status` uses the
decoder tokenizer, a configurable residue length interval, and the actual CTC
time dimension. Adjacent repeated tokens require additional blank time steps.

The model excludes infeasible targets and nonfinite sample logits/losses from
the training loss, computes CTC in FP32, and averages length-normalized losses
over retained samples only. It does not turn infinite losses into silent zeros.
Normal valid targets retain PyTorch CTCLoss's mean reduction semantics.
Validation loss uses the same applicable subset; peptide/AA metrics still
include every supplied evaluation sample, including unalignable references.

`train/ctc_seen`, `train/ctc_effective`, and `train/ctc_excluded` expose per-epoch
counts (also available with `valid/`). Warnings identify excluded samples by the
optional fourth batch field, or by within-batch position and sequence if no ID
is supplied. Experiment loaders should provide stable spectrum IDs.

An optimizer accumulation window with no effective samples leaves AdamW
parameters, momentum, weight decay, and the step scheduler unchanged. Lightning
may still advance its framework global step for an attempted window; it is not
an effective-sample count. A window with earlier valid accumulated gradients
still updates normally. Nonfinite model gradients stop before an update.

The following safeguards are configurable model/CLI configuration keys:

| Key | Default | Meaning |
| --- | --- | --- |
| `ctc_max_invalid_fraction` | 0.05 | Maximum cumulative training exclusion fraction |
| `ctc_guard_min_samples` | 1000 | Samples before the fraction guard is evaluated |
| `ctc_max_empty_batches` | 10 | Consecutive all-excluded batches before stopping |

Accounting and the consecutive-empty counter are saved in checkpoints and
restored on resume. These safety defaults are operational settings, not
statistical thresholds selected using test performance. Single-device training
and multi-device DDP training use global valid-count weighting and synchronized
empty-batch handling. A rank with no usable samples contributes a graph-connected
zero loss; an entirely empty accumulation window does not update optimizer or
scheduler state. CPU multi-process regression tests exercise these cases;
physical multi-GPU validation is reported separately.

The default learning-rate schedule remains `cosine`. `lr_schedule:
warmup_constant` selects linear warmup followed by constant learning rate.

This change does not modify the sequencing, candidate ranking, or confidence
inference path. It is a new code revision, not a byte-identical copy of the
source release archive.
