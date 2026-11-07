# Report which labels fit the available CTC time axis before aggregating loss.
# 在汇总损失前判断哪些标注能在给定时间轴上对齐。
"""Auditable, sample-wise CTC exclusion for training and validation loss."""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .ctc import required_steps


@dataclass
class CTCLossResult:
    loss: Optional[torch.Tensor]
    valid: torch.Tensor
    reasons: list


def masked_ctc_loss(logits, targets, lengths, blank):
    """Compute FP32 CTC on feasible samples; preserve CTCLoss mean semantics.

    logits is T,N,C. Invalid losses are identified with detached checks, and
    the backward graph is rebuilt without any nonfinite CTC samples.
    The normal finite path requires only one CTC evaluation.
    """
    time, size, classes = logits.shape
    if targets.ndim != 2 or targets.shape[0] != size or lengths.numel() != size:
        raise ValueError("Inconsistent CTC batch dimensions")
    reasons = []
    labels_cpu = targets.detach().cpu()
    lengths_cpu = lengths.detach().cpu().long()
    for target, length in zip(labels_cpu, lengths_cpu.tolist()):
        labels = target[:length].tolist()
        if length <= 0 or length > targets.shape[1]:
            reasons.append("invalid_target_length")
        elif any(x == blank or x < 0 or x >= classes for x in labels):
            reasons.append("invalid_target_token")
        elif required_steps(labels) > time:
            reasons.append("unalignable_target")
        else:
            reasons.append("")
    valid = torch.tensor([not r for r in reasons], device=logits.device)
    finite = torch.isfinite(logits.detach()).all(dim=0).all(dim=-1)
    for index in (~finite).nonzero().flatten().tolist():
        if not reasons[index]:
            reasons[index] = "nonfinite_logits"
    valid &= finite
    indices = valid.nonzero().flatten()
    if not indices.numel():
        return CTCLossResult(None, valid, reasons)

    with torch.autocast(device_type=logits.device.type, enabled=False):
        log_probs = logits[:, indices].float().log_softmax(-1)
        finite_probs = torch.isfinite(log_probs.detach()).all(dim=0).all(dim=-1)
        for index in indices[~finite_probs].tolist():
            reasons[index] = "nonfinite_log_probabilities"
            valid[index] = False
        indices = indices[finite_probs]
        log_probs = log_probs[:, finite_probs]
        if not indices.numel():
            return CTCLossResult(None, valid, reasons)

        def evaluate(chosen_probs, chosen_indices):
            selected_lengths = lengths_cpu[chosen_indices.cpu()]
            return F.ctc_loss(
                chosen_probs, targets[chosen_indices],
                torch.full((len(chosen_indices),), time, dtype=torch.long),
                selected_lengths, blank=blank, reduction="none", zero_infinity=False,
            )

        losses = evaluate(log_probs, indices)
        finite_loss = torch.isfinite(losses.detach())
        if not finite_loss.all():
            for index in indices[~finite_loss].tolist():
                reasons[index] = "nonfinite_ctc_loss"
                valid[index] = False
            indices = indices[finite_loss]
            if not indices.numel():
                return CTCLossResult(None, valid, reasons)
            # Rebuild CTC's graph without the failing samples, not loss * mask.
            losses = evaluate(log_probs[:, finite_loss], indices)
            if not torch.isfinite(losses).all():
                raise FloatingPointError("CTC remains nonfinite after sample isolation")
        normalized = losses / lengths_cpu[indices.cpu()].to(losses.device)
        return CTCLossResult(normalized.mean(), valid, reasons)


def training_target_status(sequence, decoder, time_steps, min_length=6, max_length=40):
    """Assess training applicability, not the quality of the reference label.

    Use the model tokenizer: terminal modifications count as CTC tokens but
    not residues; residue modifications do not add amino acids.
    """
    import re

    residue_count = len(re.findall(r"[A-Z]", sequence))
    if not min_length <= residue_count <= max_length:
        return "outside_training_length"
    try:
        tokens = decoder.tokenize(sequence).tolist()
    except (KeyError, ValueError):
        return "unsupported_training_token"
    if decoder.get_blank_idx() in tokens or not tokens:
        return "invalid_target_token"
    if required_steps(tokens) > time_steps:
        return "unalignable_target"
    return ""
