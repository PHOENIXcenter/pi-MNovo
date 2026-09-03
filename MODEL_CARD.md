# Model card: pi-MNovo v0.1.0

## Model

pi-MNovo predicts peptide sequences directly from MS/MS spectra. The v0.1.0
checkpoint contains a microbial-domain-adapted non-autoregressive CTC
backbone, Beam5 plus PMC-union candidate generation, a default ranker, a
long-peptide expert, a fragment-evidence expert, a length predictor, an
observable hard router, and score calibration.

## Intended use

- De novo peptide sequencing of DDA tandem mass spectra in MGF format.
- Evaluation of annotated MGF files containing reference peptide sequences.
- Research use in microbial proteomics and metaproteomics.

The model is not a database-search engine, taxonomic classifier, protein
inference method, or clinical diagnostic system.

## Inputs and outputs

The model accepts precursor m/z, precursor charge, and centroided fragment
peaks. It supports peptide outputs up to 40 residues and the residue/modification
vocabulary embedded in the checkpoint. The exported `Score` lies in `[0, 1]`;
larger values indicate greater empirical confidence. A score near `0.92` was
used as the manuscript high-confidence operating threshold. The score is not
an estimated false discovery rate.

## Validation snapshot

The frozen manuscript checkpoint obtained a peptide recall of 0.6630 on the
pooled seven-species external evaluation set (134,728 correct predictions from
203,204 spectra). These data were not used to train model parameters. See the
associated manuscript and supplementary tables for per-dataset results and
the complete evaluation design.

## Limitations

- Performance depends on spectrum quality, acquisition conditions, peptide
  length, charge state, and modification coverage.
- Peptides longer than 40 residues cannot be emitted by this release.
- The confidence threshold should be rechecked when acquisition or sample
  domains differ substantially from the validation data.
- High-confidence output is not a replacement for target-decoy FDR control.

## Integrity

Checkpoint SHA256:

```text
1de589a887a7b7271aae794b90850ece6c3c68c6d085374d0b32e40ec18d617f
```
