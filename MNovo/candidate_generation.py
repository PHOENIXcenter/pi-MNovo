"""Candidate-generation helpers shared by the fused inference runtime."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import torch

from MNovo.denovo import mass_con
from MNovo.denovo.model import Spec2Pep, ctc_post_processing

LOGGER = logging.getLogger(__name__)


def model_parameters(config: dict, args: SimpleNamespace) -> dict:
    """Translate the release YAML and runtime options into model arguments."""
    return {
        "PMC_enable": bool(config["PMC_enable"]),
        "mass_control_tol": float(config["mass_control_tol"]),
        "dim_model": int(config["dim_model"]),
        "n_head": int(config["n_head"]),
        "dim_feedforward": int(config["dim_feedforward"]),
        "n_layers": int(config["n_layers"]),
        "dropout": float(config["dropout"]),
        "dim_intensity": config.get("dim_intensity"),
        "custom_encoder": config.get("custom_encoder"),
        "max_length": int(config["max_length"]),
        "residues": config["residues"],
        "max_charge": int(config["max_charge"]),
        "precursor_mass_tol": float(config["precursor_mass_tol"]),
        "isotope_error_range": tuple(config["isotope_error_range"]),
        "n_beams": int(args.beam_size),
        "n_log": int(config.get("n_log", 1)),
        "ctc_dic": {
            "beam": int(args.beam_size),
            "cutoff_top_n": int(args.cutoff_top_n),
            "cutoff_prob": 1.0,
            "num_processes": int(args.ctc_processes),
        },
    }


def _token_mass(model: Spec2Pep, token_id: int) -> float:
    symbol = model.decoder._idx2aa[int(token_id)]
    return float(model.decoder._peptide_mass.masses.get(symbol, 0.0))


def pmc_candidates(
    model: Spec2Pep,
    log_probs: torch.Tensor,
    precursors: torch.Tensor,
    top_tokens: list[list[int]],
    enabled: bool,
) -> list[list[int]]:
    """Generate one precise-mass-control proposal per eligible spectrum."""
    candidates: list[list[int]] = [[] for _ in top_tokens]
    if not enabled or not model.PMC_enable:
        return candidates
    for idx, ctc_tokens in enumerate(top_tokens):
        target = float(precursors[idx, 0].detach().cpu()) - 18.01
        predicted = sum(_token_mass(model, token) for token in ctc_tokens)
        if abs(target - predicted) < float(model.mass_control_tol):
            continue
        try:
            tokens = mass_con.knapDecode(
                log_probs[idx : idx + 1],
                precursors[idx : idx + 1, 0],
                model.mass_control_tol,
            )
            candidates[idx] = [
                int(token) for token in ctc_post_processing(tokens) if int(token) >= 0
            ]
        except (RuntimeError, ValueError) as error:
            LOGGER.warning("PMC candidate failed at batch item %d: %s", idx, error)
    return candidates


def ctc_nll_for_batch(
    model: Spec2Pep,
    log_probs: torch.Tensor,
    token_lists: list[list[int]],
) -> list[float]:
    """Calculate per-candidate forced CTC negative log likelihood."""
    valid = [
        idx
        for idx, tokens in enumerate(token_lists)
        if tokens and len(tokens) <= int(log_probs.size(1))
    ]
    output = [float("inf")] * len(token_lists)
    if not valid:
        return output
    targets = torch.tensor(
        [token for idx in valid for token in token_lists[idx]],
        device=log_probs.device,
        dtype=torch.long,
    )
    target_lengths = torch.tensor(
        [len(token_lists[idx]) for idx in valid],
        device=log_probs.device,
        dtype=torch.long,
    )
    input_lengths = torch.full(
        (len(valid),),
        int(log_probs.size(1)),
        device=log_probs.device,
        dtype=torch.long,
    )
    loss = torch.nn.CTCLoss(
        blank=int(model.decoder.get_blank_idx()),
        reduction="none",
        zero_infinity=True,
    )(
        log_probs[valid].transpose(0, 1).contiguous(),
        targets,
        input_lengths,
        target_lengths,
    )
    for idx, value in zip(valid, loss.detach().cpu().tolist()):
        output[idx] = float(value)
    return output


def merge_pmc_candidate(
    candidates: list[list[int]],
    nlls: list[float],
    pmc_tokens: list[int],
    pmc_nll: float,
    beam_size: int,
    mode: str,
) -> tuple[list[list[int]], list[float], list[str], bool]:
    """Append a unique PMC proposal without replacing Beam candidates."""
    sources = ["ctc"] * len(candidates)
    if not pmc_tokens or mode == "off":
        return candidates, nlls, sources, False
    matching_index = next(
        (
            idx
            for idx, candidate in enumerate(candidates)
            if tuple(candidate) == tuple(pmc_tokens)
        ),
        None,
    )
    if mode != "union":
        raise ValueError("The release runtime supports only PMC union or off.")
    if matching_index is not None:
        return candidates, nlls, sources, False
    return (
        candidates[:beam_size] + [pmc_tokens],
        nlls[:beam_size] + [pmc_nll],
        sources[:beam_size] + ["pmc_union"],
        True,
    )
