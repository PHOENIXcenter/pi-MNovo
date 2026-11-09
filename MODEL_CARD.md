# Model and FDR scope

Version 0.2.2 loads the
backbone, experts, length predictor and router from a verified unified checkpoint.
Architecture, vocabulary and decoding settings belong to that checkpoint.
Record its SHA256 and resolved configuration with every result.

The regression reference for this integration is REL-20261003-01 sequencing,
using Beam20 + PMC union and the existing single-bacterium weights. No new
benchmark claim is implied by the code version. Historical preprint weights
and metaproteome weights are separate experimental choices.

Inputs are centroided DDA MS/MS MGF spectra with precursor m/z and charge.
Output Score is not a q-value, PEP or FDR. The optional CCT module fits separate
PSM and peptide scoring/calibration models; a profile is cohort/scope-specific.
It must not be transferred to new conditions without validation.

Feature rejection and input preprocessing rejection are different accounting
categories. The safe Spearman repair retains 33 features and records flags.
Unknown database labels do not prove correctness or incorrectness.
Reference FDP applies only to reference-known retained predictions.

Research use only; not a taxonomic classifier, protein-inference system or
clinical diagnostic. Acquisition, modification coverage, peptide length and
label-selection bias can affect sequencing and FDR performance.
