# Release decisions

## DEC-20260903-PUBLIC-RELEASE

- **Date:** 2026-09-03
- **Candidate/release:** `pi-MNovo-v0.1.0`
- **Decision:** Adopt as the current public manuscript software and model release.
- **Evidence reviewed:** Frozen manuscript component manifest; SHA256 and embedded-component verification; exact equality of all 286 backbone tensors before and after path sanitization; clean-environment installation; static checks; 16 unit tests; and a two-spectrum end-to-end synthetic-MGF smoke test.
- **Reason:** This snapshot packages the adopted alpha-0.25 backbone, Beam5 plus PMC union candidate generation, R1/R2/R3 sequence-ranking paths, conservative observable router, and monotonic Score calibration in one checkpoint and one inference entry point. Abandoned FDR, residual confidence, Soft MoE, and dynamic Beam20 experiments are excluded.
- **Previous current release:** None in the public repository.
- **New current release:** `pi-MNovo-v0.1.0`.
- **Required manuscript updates:** None; this release is intended to match the frozen manuscript model.
- **Approver or user instruction:** User explicitly designated this model as the publishable version and requested a public GitHub release.
