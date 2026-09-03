"""Vectorized CTC token-evidence features for MNovo candidate sequences."""

from __future__ import annotations

from collections.abc import Sequence

import torch


TOKEN_EVIDENCE_FEATURE_NAMES = [
    "forced_ctc_score",
    "forced_ctc_length_normalized_score",
    "viterbi_length_normalized_score",
    "residue_posterior_mean",
    "residue_posterior_min",
    "residue_posterior_q10",
    "residue_posterior_q25",
    "residue_entropy_mean",
    "residue_top2_margin_mean",
    "longest_low_confidence_run_scaled",
    "blank_path_fraction",
    "alignment_ambiguity_per_token",
    "prefix_residue_posterior_mean",
    "suffix_residue_posterior_mean",
]


def _pad_candidates(
    candidates: Sequence[Sequence[Sequence[int]]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = len(candidates)
    width = max((len(rows) for rows in candidates), default=0)
    max_length = max(
        (len(tokens) for rows in candidates for tokens in rows),
        default=0,
    )
    if width == 0 or max_length == 0:
        return (
            torch.empty((0, 0), dtype=torch.long, device=device),
            torch.empty((0,), dtype=torch.long, device=device),
            torch.empty((0,), dtype=torch.long, device=device),
            torch.empty((batch_size, width), dtype=torch.bool, device=device),
        )
    mask = torch.zeros((batch_size, width), dtype=torch.bool, device=device)
    flat_tokens = []
    lengths = []
    batch_indices = []
    for batch_index, rows in enumerate(candidates):
        for candidate_index, tokens in enumerate(rows):
            if not tokens:
                continue
            mask[batch_index, candidate_index] = True
            padded = list(tokens) + [0] * (max_length - len(tokens))
            flat_tokens.append(padded)
            lengths.append(len(tokens))
            batch_indices.append(batch_index)
    return (
        torch.tensor(flat_tokens, dtype=torch.long, device=device),
        torch.tensor(lengths, dtype=torch.long, device=device),
        torch.tensor(batch_indices, dtype=torch.long, device=device),
        mask,
    )


def _viterbi_states(
    selected_log_probs: torch.Tensor,
    tokens: torch.Tensor,
    lengths: torch.Tensor,
    blank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    count, time_steps, _vocabulary = selected_log_probs.shape
    max_length = tokens.size(1)
    max_states = 2 * max_length + 1
    state_positions = torch.arange(max_states, device=tokens.device)
    expanded = torch.full(
        (count, max_states),
        int(blank),
        dtype=torch.long,
        device=tokens.device,
    )
    expanded[:, 1::2] = tokens
    state_counts = 2 * lengths + 1
    valid_states = state_positions.unsqueeze(0) < state_counts.unsqueeze(1)
    emissions = selected_log_probs.gather(
        2,
        expanded.unsqueeze(1).expand(-1, time_steps, -1),
    )
    emissions = emissions.masked_fill(~valid_states.unsqueeze(1), -torch.inf)

    score = torch.full(
        (count, max_states),
        -torch.inf,
        dtype=selected_log_probs.dtype,
        device=tokens.device,
    )
    score[:, 0] = emissions[:, 0, 0]
    score[:, 1] = emissions[:, 0, 1]
    backpointers = torch.zeros(
        (time_steps, count, max_states),
        dtype=torch.int8,
        device=tokens.device,
    )
    skip_allowed = (
        (state_positions.unsqueeze(0) >= 2)
        & (state_positions.unsqueeze(0) % 2 == 1)
        & (expanded != torch.roll(expanded, shifts=2, dims=1))
        & valid_states
    )
    for time_index in range(1, time_steps):
        step_one = torch.roll(score, shifts=1, dims=1)
        step_one[:, 0] = -torch.inf
        step_two = torch.roll(score, shifts=2, dims=1)
        step_two[:, :2] = -torch.inf
        step_two = step_two.masked_fill(~skip_allowed, -torch.inf)
        options = torch.stack((score, step_one, step_two), dim=0)
        best, transition = options.max(dim=0)
        score = (best + emissions[:, time_index]).masked_fill(
            ~valid_states,
            -torch.inf,
        )
        backpointers[time_index] = transition.to(torch.int8)

    rows = torch.arange(count, device=tokens.device)
    final_blank = state_counts - 1
    final_label = state_counts - 2
    final_options = torch.stack(
        (score[rows, final_blank], score[rows, final_label]),
        dim=1,
    )
    final_choice = final_options.argmax(dim=1)
    state = torch.where(final_choice == 0, final_blank, final_label)
    viterbi_score = final_options.gather(1, final_choice[:, None]).squeeze(1)
    path = torch.empty(
        (count, time_steps),
        dtype=torch.long,
        device=tokens.device,
    )
    path[:, -1] = state
    for time_index in range(time_steps - 1, 0, -1):
        transition = backpointers[time_index, rows, state].long()
        state = state - transition
        path[:, time_index - 1] = state
    return path, viterbi_score


def token_evidence_features(
    log_probs: torch.Tensor,
    candidates: Sequence[Sequence[Sequence[int]]],
    blank: int,
    *,
    length_alpha: float = 0.7,
    low_confidence_threshold: float = 0.5,
    ignored_residue_tokens: set[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return padded candidate features and a candidate-validity mask.

    `log_probs` has shape ``[batch, time, vocabulary]``. Candidate tokens must
    use the same CTC decoding direction as the model.
    """

    if log_probs.ndim != 3:
        raise ValueError("log_probs must have shape [batch, time, vocabulary].")
    device = log_probs.device
    tokens, lengths, batch_indices, candidate_mask = _pad_candidates(
        candidates,
        device,
    )
    batch_size, width = candidate_mask.shape
    feature_count = len(TOKEN_EVIDENCE_FEATURE_NAMES)
    output = torch.zeros(
        (batch_size, width, feature_count),
        dtype=torch.float32,
        device=device,
    )
    if tokens.numel() == 0:
        return output, candidate_mask

    selected = log_probs.float().index_select(0, batch_indices)
    count, time_steps, _vocabulary = selected.shape
    if torch.any(lengths > time_steps):
        raise ValueError("A candidate is longer than the CTC time axis.")
    flat_targets = torch.cat(
        [tokens[row, : lengths[row]] for row in range(count)],
        dim=0,
    )
    input_lengths = torch.full(
        (count,),
        time_steps,
        dtype=torch.long,
        device=device,
    )
    forced_nll = torch.nn.functional.ctc_loss(
        selected.transpose(0, 1).contiguous(),
        flat_targets,
        input_lengths,
        lengths,
        blank=int(blank),
        reduction="none",
        zero_infinity=True,
    )
    path, viterbi_score = _viterbi_states(selected, tokens, lengths, blank)

    max_length = tokens.size(1)
    positions = torch.arange(max_length, device=device)
    residue_valid = positions.unsqueeze(0) < lengths.unsqueeze(1)
    if ignored_residue_tokens:
        ignored = torch.zeros_like(residue_valid)
        for token_id in ignored_residue_tokens:
            ignored |= tokens == int(token_id)
        residue_valid &= ~ignored
    effective_lengths = residue_valid.sum(dim=1).clamp_min(1)
    residue_states = 2 * positions + 1
    alignment_mask = (
        path.unsqueeze(1) == residue_states.view(1, max_length, 1)
    ) & residue_valid.unsqueeze(2)
    alignment_count = alignment_mask.sum(dim=2).clamp_min(1)

    token_log_probs = selected.gather(
        2,
        tokens.unsqueeze(1).expand(-1, time_steps, -1),
    ).transpose(1, 2)
    token_probabilities = token_log_probs.exp()
    residue_posterior = token_probabilities.masked_fill(
        ~alignment_mask,
        -torch.inf,
    ).amax(dim=2)
    residue_posterior = residue_posterior.masked_fill(~residue_valid, 1.0)

    timestep_entropy = -(selected.exp() * selected).sum(dim=2)
    top_two = selected.exp().topk(2, dim=2).values
    timestep_margin = top_two[:, :, 0] - top_two[:, :, 1]
    residue_entropy = (
        timestep_entropy.unsqueeze(1)
        .expand(-1, max_length, -1)
        .masked_fill(~alignment_mask, 0.0)
        .sum(dim=2)
        / alignment_count
    )
    residue_margin = (
        timestep_margin.unsqueeze(1)
        .expand(-1, max_length, -1)
        .masked_fill(~alignment_mask, 0.0)
        .sum(dim=2)
        / alignment_count
    )

    valid_float = residue_valid.float()
    posterior_mean = (residue_posterior * valid_float).sum(dim=1) / effective_lengths
    posterior_min = residue_posterior.amin(dim=1)
    sorted_posterior = residue_posterior.sort(dim=1).values
    q10_index = torch.floor((effective_lengths - 1).float() * 0.10).long()
    q25_index = torch.floor((effective_lengths - 1).float() * 0.25).long()
    q10 = sorted_posterior.gather(1, q10_index[:, None]).squeeze(1)
    q25 = sorted_posterior.gather(1, q25_index[:, None]).squeeze(1)
    entropy_mean = (residue_entropy * valid_float).sum(dim=1) / effective_lengths
    margin_mean = (residue_margin * valid_float).sum(dim=1) / effective_lengths

    run = torch.zeros(count, dtype=torch.long, device=device)
    longest_run = torch.zeros_like(run)
    low = (residue_posterior < low_confidence_threshold) & residue_valid
    for position in range(max_length):
        run = torch.where(low[:, position], run + 1, torch.zeros_like(run))
        longest_run = torch.maximum(longest_run, run)

    first_count = effective_lengths.clamp_max(3)
    residue_rank = residue_valid.long().cumsum(dim=1) - 1
    prefix_mask = (residue_rank < first_count.unsqueeze(1)) & residue_valid
    suffix_mask = (
        residue_rank >= (effective_lengths - first_count).unsqueeze(1)
    ) & residue_valid
    prefix_mean = (
        (residue_posterior * prefix_mask.float()).sum(dim=1) / first_count
    )
    suffix_mean = (
        (residue_posterior * suffix_mask.float()).sum(dim=1) / first_count
    )

    length_scale = effective_lengths.float().pow(max(float(length_alpha), 1e-6))
    total_log_probability = -forced_nll
    ambiguity = (total_log_probability - viterbi_score).clamp_min(0.0)
    blank_fraction = (path % 2 == 0).float().mean(dim=1)
    features = torch.stack(
        (
            total_log_probability,
            total_log_probability / length_scale,
            viterbi_score / length_scale,
            posterior_mean,
            posterior_min,
            q10,
            q25,
            entropy_mean,
            margin_mean,
            longest_run.float() / effective_lengths,
            blank_fraction,
            ambiguity / effective_lengths,
            prefix_mean,
            suffix_mean,
        ),
        dim=1,
    )
    output[candidate_mask] = features
    return output, candidate_mask
