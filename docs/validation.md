# Validation scope

Historical validation of 0.2.0rc1 / CODE-20261027-clean-candidate, 2026-10-27.
These results describe that historical snapshot.
Original frozen sources and weights are not modified. No GitHub push.

## Sequencing compatibility

128 real spectra from an existing regression fixture, REL-20261003-01 weights,
Beam20 + PMC, BF16, batch 64. Native predictions are byte-identical to the
saved CODE-20261025-01 baseline with FDR off, feature export, FDR on, and the
worker path. Normal 33-feature values are unchanged. This is regression
evidence, not a rerun of every historical benchmark.

Input SHA256: affdf35ed3dc5ec2635bfbb892b294c6db8338c8a996d1f5d389a830aa0fc2e8.
Native TSV SHA256: 608039320b6d850e84253f3b82c5050772259a879c6dfcf08efd824d19954b35.

FDR on/off entry-point tests use a clearly marked synthetic test profile;
they validate software plumbing, not scientific FDR performance.

## Adopted FDR parity

Seven historical single-strain smoke sets (512 each; 3,584 spectra total)
also reproduce sequence, route, status and metadata exactly with FDR off.
Historical annotate-mode Score values used legacy PEP calibration and are
deliberately excluded from this comparison; native Score parity is covered
by the separate byte-identical 128-spectrum test above.

Read-only rescoring of the existing 2,800,785 scoreable MIX12 pairs with frozen
PSM/peptide scorers and gamma reproduced all 169,369 retained PSM identities
and 33,700 retained peptide identities. Scores agree within 1e-15; cutoffs and
Counterpart counts match. No sequencing or model training was repeated.
This validates implementation equivalence, not a new independent experiment.

## Devices

One physical A100 was available. Two CPU DDP processes passed normal-data,
one-empty-rank, all-empty-batch and training-CLI checks. Two inference workers
sharing that one A100 preserved native output and global FDR selection.
**Physical two-GPU/NCCL execution remains untested.** Run that hardware
acceptance test before claiming a measured multi-GPU speedup or parity.

## Statistical boundaries

Only explicit SEQ metadata provides references in MGF exports. TITLE is never
supervision. Old LMDBs without reference_sequence remain unlabelled for FDR.
Known/unlabelled mixed inputs, unsafe nonfinite rejection, safe Spearman repair,
complete pair recovery, score floors, tied scores, fixed-role isolation and
profile identity mismatches have regression tests.

Independent tests and full-cohort application are distinct. Full application
may contain fitting/calibration rows; its reference FDP is not independent.
No guarantee of unknown-label or per-RAW FDR is inferred from software PASS.
