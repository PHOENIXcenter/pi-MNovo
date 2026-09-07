# Release provenance

## Current release

| Field | Value |
|---|---|
| Release | `pi-MNovo-v0.1.0` |
| Status | `CURRENT` |
| Public checkpoint | `pi-MNovo-v0.1.0.ckpt` |
| Public SHA256 | `1de589a887a7b7271aae794b90850ece6c3c68c6d085374d0b32e40ec18d617f` |
| Candidate generation | Beam5 + PMC union |
| Sequence selection | R1 default, R2 long expert, R3 fragment expert, observable router |
| Score threshold | approximately 0.92 |

The public checkpoint was repacked from the frozen manuscript checkpoint to
remove some machine-local paths. This historical statement did not cover all nested component metadata; the recursive review audit supersedes the path-cleanliness claim. All 286 backbone state tensors were compared with
`torch.equal`; no tensor changed. Component payload hashes are verified during
checkpoint extraction.

## Evaluation claim

| Value | Data role | Source release | Status |
|---|---|---|---|
| External seven-species peptide recall: 0.663018 | Final independent evaluation | `pi-MNovo-v0.1.0` | Current |
| Correct spectra: 134,728 / 203,204 | Final independent evaluation | `pi-MNovo-v0.1.0` | Current |

Historical exploratory FDR, dynamic Beam20, Soft MoE, and residual confidence
head experiments are intentionally excluded from this software release.
