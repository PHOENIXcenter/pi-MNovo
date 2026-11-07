# Score the sequence already selected by the router; this is not CCT calibration.
# 对路由已经选定的序列评分，不在这里做 CCT 校准。
"""Historical sequence Score, independent of CCT calibration and filtering."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F


class SequenceConfidenceHead(torch.nn.Module):
    """Architecture used by the frozen REL-20260920-01 confidence model."""

    def __init__(
        self,
        vocab_size: int,
        global_dim: int,
        memory_dim: int = 128,
        token_dim: int = 6,
        hidden: int = 192,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token = torch.nn.Embedding(vocab_size, memory_dim)
        self.position = torch.nn.Embedding(40, memory_dim)
        self.cross = torch.nn.MultiheadAttention(
            memory_dim, 4, dropout=dropout, batch_first=True
        )
        self.local_norm = torch.nn.LayerNorm(memory_dim)
        self.residue = torch.nn.Sequential(
            torch.nn.Linear(memory_dim + token_dim, 96),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(96, 1),
        )
        self.boundary = torch.nn.Sequential(
            torch.nn.Linear(memory_dim * 2, 96),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(96, 1),
        )
        self.global_head = torch.nn.Sequential(
            torch.nn.LayerNorm(memory_dim * 2 + global_dim),
            torch.nn.Linear(memory_dim * 2 + global_dim, hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, 96),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(96, 1),
        )
        self.c2_weight = torch.nn.Parameter(torch.tensor(0.0))
        self.c3_weight = torch.nn.Parameter(torch.zeros(6))


class LongPeptideConfidenceExpert(torch.nn.Module):
    """Frozen residual confidence expert for peptides of length 26--40."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, 128),
            torch.nn.GELU(),
            torch.nn.Dropout(0.15),
            torch.nn.Linear(128, 64),
            torch.nn.GELU(),
            torch.nn.Dropout(0.10),
            torch.nn.Linear(64, 1),
        )

    def forward(
        self, features: torch.Tensor, base_logit: torch.Tensor
    ) -> torch.Tensor:
        return base_logit + 2.0 * torch.tanh(self.net(features).squeeze(1))


def ctc_viterbi_features(
    log_probs: torch.Tensor,
    targets: list[list[int]],
    blank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the six frozen token-evidence features for selected peptides."""
    device = log_probs.device
    # B × T × V: these are decoder frames, not measured peak positions.
    # 三维分别是谱图批次、解码时间步和词表；时间步不是实验峰的编号。
    batch, steps, vocab = log_probs.shape
    lengths = torch.tensor([len(row) for row in targets], device=device)
    max_len = int(lengths.max())
    target = torch.full((batch, max_len), blank, dtype=torch.long, device=device)
    for index, row in enumerate(targets):
        target[index, : len(row)] = torch.tensor(row, device=device)
    states = 2 * max_len + 1
    extended = torch.full((batch, states), blank, dtype=torch.long, device=device)
    extended[:, 1::2] = target
    valid = torch.arange(states, device=device)[None] < (2 * lengths + 1)[:, None]
    emissions = log_probs.gather(2, extended[:, None, :].expand(-1, steps, -1))
    negative = torch.tensor(-1e30, device=device)
    dynamic = torch.full((batch, states), negative, device=device)
    dynamic[:, 0] = emissions[:, 0, 0]
    dynamic[:, 1] = emissions[:, 0, 1]
    back = torch.zeros((steps, batch, states), dtype=torch.int8, device=device)
    for step in range(1, steps):
        previous_one = torch.cat(
            [negative.expand(batch, 1), dynamic[:, :-1]], dim=1
        )
        previous_two = torch.cat(
            [negative.expand(batch, 2), dynamic[:, :-2]], dim=1
        )
        skip = torch.zeros_like(valid)
        skip[:, 2:] = (
            (torch.arange(states, device=device)[None, 2:] % 2 == 1)
            & (extended[:, 2:] != extended[:, :-2])
        )
        previous_two = previous_two.masked_fill(~skip, negative)
        best, choice = torch.stack(
            [dynamic, previous_one, previous_two], dim=-1
        ).max(dim=-1)
        dynamic = (best + emissions[:, step]).masked_fill(~valid, negative)
        back[step] = choice.to(torch.int8)
    last_blank = 2 * lengths
    last_token = last_blank - 1
    choose_token = dynamic.gather(1, last_token[:, None]) > dynamic.gather(
        1, last_blank[:, None]
    )
    state = torch.where(choose_token[:, 0], last_token, last_blank)
    alignment = torch.zeros((batch, steps), dtype=torch.long, device=device)
    rows = torch.arange(batch, device=device)
    for step in range(steps - 1, 0, -1):
        alignment[:, step] = state
        state = state - back[step, rows, state].long()
    alignment[:, 0] = state
    token_state = alignment % 2 == 1
    residue = ((alignment - 1) // 2).clamp_min(0)
    aligned_token = target.gather(1, residue.clamp_max(max_len - 1))
    selected_lp = log_probs.gather(2, aligned_token[:, :, None]).squeeze(2)
    probabilities = log_probs.exp()
    entropy = -(probabilities * log_probs).sum(2) / math.log(vocab)
    top_two = log_probs.topk(2, dim=2).values
    margin = top_two[:, :, 0] - top_two[:, :, 1]
    blank_probability = probabilities[:, :, blank]
    flat_size = batch * max_len
    flat_index = (rows[:, None] * max_len + residue).reshape(-1)
    active = token_state.reshape(-1)
    index = flat_index[active]
    count = torch.zeros(flat_size, device=device).scatter_add_(
        0, index, torch.ones_like(index, dtype=torch.float)
    )
    values = []
    for source, mode in (
        (selected_lp.exp(), "mean"),
        (selected_lp.exp(), "min"),
        (entropy, "mean"),
        (margin, "mean"),
        (blank_probability, "mean"),
    ):
        flat = source.reshape(-1)[active]
        if mode == "min":
            output = torch.full((flat_size,), float("inf"), device=device)
            output.scatter_reduce_(
                0, index, flat, reduce="amin", include_self=True
            )
            output[output.isinf()] = 0
        else:
            output = torch.zeros(flat_size, device=device).scatter_add_(
                0, index, flat
            ) / count.clamp_min(1)
        values.append(output.view(batch, max_len))
    values.append((count / steps).view(batch, max_len))
    features = torch.stack(values, dim=2)
    mask = torch.arange(max_len, device=device)[None] < lengths[:, None]
    return features, mask


def local_summaries(
    token_features: torch.Tensor, token_mask: torch.Tensor
) -> torch.Tensor:
    rows = []
    for features, valid in zip(token_features.float(), token_mask):
        values = features[valid]
        ordered = values.sort(0).values
        q10 = ordered[max(0, math.ceil(len(values) * 0.1) - 1)]
        rows.append(
            torch.cat(
                [
                    values.mean(0),
                    values.min(0).values,
                    q10,
                    values.max(0).values,
                ]
            )
        )
    return torch.stack(rows)


class SequenceScoreRuntime:
    """Apply the original C1 and long-peptide Score without any FDR decision."""

    def __init__(self, components: Path, device: torch.device) -> None:
        self.device = device
        base = torch.load(
            components / "sequence_confidence_head_calibrated.pt",
            map_location="cpu",
            weights_only=False,
        )
        self.base = SequenceConfidenceHead(**base["model_args"]).to(device).eval()
        self.base.load_state_dict(base["model_state"])
        self.base.requires_grad_(False)
        if base.get("selected_variant") != "C1":
            raise RuntimeError("Unified runtime is frozen to confidence variant C1.")
        self.base_mean = base["feature_mean"].to(device).float()
        self.base_std = base["feature_std"].to(device).float()
        self.base_calibrator = base["calibrator"]

        long_checkpoint = torch.load(
            components / "long_peptide_confidence_expert.pt",
            map_location="cpu",
            weights_only=False,
        )
        self.long = LongPeptideConfidenceExpert(
            int(long_checkpoint["input_dim"])
        ).to(device).eval()
        self.long.load_state_dict(long_checkpoint["model_state"])
        self.long.requires_grad_(False)
        self.long_mean = long_checkpoint["feature_mean"].to(device).float()
        self.long_std = long_checkpoint["feature_std"].to(device).float()
        self.long_calibrator = long_checkpoint["calibrator"]

    def score(
        self,
        spectrum_embedding: torch.Tensor,
        global_features: torch.Tensor,
        token_ids: torch.Tensor,
        token_features: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        token_ids = token_ids.to(self.device).long().clamp_min(0)
        token_mask = token_mask.to(self.device).bool()
        token_features = token_features.to(self.device).float()
        lengths = token_mask.sum(1)
        positions = torch.arange(token_ids.shape[1], device=self.device)[None]
        query = self.base.token(token_ids) + self.base.position(positions)
        mask = token_mask.float()
        sequence_embedding = (query * mask[:, :, None]).sum(1) / mask.sum(
            1, keepdim=True
        ).clamp_min(1)
        normalized = (
            global_features.to(self.device).float() - self.base_mean
        ) / self.base_std
        base_logit = self.base.global_head(
            torch.cat(
                [
                    spectrum_embedding.to(self.device).float(),
                    sequence_embedding,
                    normalized,
                ],
                dim=1,
            )
        ).squeeze(1)
        bucket = torch.where(lengths < 16, 0, torch.where(lengths <= 25, 1, 2))
        length_bias = torch.tensor(
            self.base_calibrator["length_bias"], device=self.device
        )
        z = (
            float(self.base_calibrator["scale"]) * base_logit
            + float(self.base_calibrator["bias"])
            + length_bias[bucket]
        )
        score = torch.sigmoid(z)
        # The long-sequence correction is applied after base calibration, at the frozen boundary.
        # 先做基础校准，再对达到冻结长度边界的序列应用长肽修正。
        long = lengths >= 26
        if long.any():
            local = local_summaries(
                token_features[long], token_mask[long]
            ).to(self.device)
            features = torch.cat(
                [
                    normalized[long],
                    local,
                    base_logit[long, None],
                    lengths[long, None].float() / 40.0,
                ],
                dim=1,
            )
            normalized_long = (features - self.long_mean) / self.long_std
            long_logit = self.long(normalized_long, base_logit[long])
            long_z = (
                float(self.long_calibrator["scale"]) * long_logit
                + float(self.long_calibrator["bias"])
            )
            # Autocast can keep the long-peptide expert in BF16 while the
            # calibrated score tensor is FP32. Normalize the assignment dtype
            # without changing the score calculation or operating threshold.
            score[long] = torch.sigmoid(long_z).to(score.dtype)
        if not torch.isfinite(score).all():
            raise FloatingPointError("Non-finite sequence Score; refusing invalid export.")
        return score


def build_score_inputs(rows, selected, evidence, log_probs, blank):
    """Reproduce the released confidence features for already selected peptides."""
    device = log_probs.device
    indices = selected.detach().cpu().tolist()
    tokens = [row['tokens'][i] for row, i in zip(rows, indices)]
    token_features, token_mask = ctc_viterbi_features(log_probs.float(), tokens, blank)
    lengths = token_mask.sum(1).long()
    summaries = []
    for features, length in zip(token_features, lengths.tolist()):
        values = features[:length]
        ordered = values[:, 0].sort().values
        q10 = min(max(int(math.ceil(length * .1)) - 1, 0), length - 1)
        edge = max(1, length // 4)
        summaries.append(torch.stack([
            values[:, 0].mean(), values[:, 0].min(), ordered[q10],
            values[:, 2].mean(), values[:, 2].max(),
            values[:, 3].mean(), values[:, 3].min(),
            values[:, 4].mean(), values[:, 5].mean(),
            values[:edge, 0].mean(), values[-edge:, 0].mean(),
            (values[:, 0] < .5).float().mean(),
        ]))
    nll = torch.tensor([r['nll'][i] for r, i in zip(rows, indices)], device=device)
    trailing = torch.stack([lengths.float() / 40., evidence['selected_is_pmc'],
                            -nll, -nll / lengths.clamp_min(1)], dim=1)
    global_features = torch.cat([
        evidence['router_features'], evidence['route_probabilities'],
        evidence['route_advantage'][:, None], evidence['selected_core'],
        evidence['selected_fragment'].float(), evidence['selected_mass0'],
        trailing, torch.stack(summaries),
    ], dim=1).float()
    ids = torch.zeros((len(tokens), int(lengths.max())), dtype=torch.long, device=device)
    for i, row in enumerate(tokens):
        ids[i, :len(row)] = torch.tensor(row, device=device)
    return global_features, ids, token_features, token_mask
