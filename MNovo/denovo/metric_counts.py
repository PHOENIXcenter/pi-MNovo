"""Truth-first integer counts; failed predictions remain in the denominator."""

import torch
from torchmetrics import Metric

from MNovo.evaluation import tokens
from .evaluate import aa_match_batch


def match_counts(truths, predictions, masses):
    if len(truths) != len(predictions):
        raise ValueError("Prediction count must equal the number of spectra.")
    truth_tokens = [tokens(value) for value in truths]
    if any(not row for row in truth_tokens):
        raise ValueError("Metrics require a non-empty truth for every spectrum.")
    if any(token not in masses for row in truth_tokens for token in row):
        raise ValueError(
            "Truth contains a token missing from the configured residue masses."
        )
    predicted_tokens = []
    for value in predictions:
        value = value if isinstance(value, str) else "".join(value)
        value = value.removeprefix("$")
        # Internal stop and unsupported predictions are failures, never exclusions.
        try:
            parsed = tokens(value)
            predicted_tokens.append(
                parsed if all(token in masses for token in parsed) else []
            )
        except ValueError:
            predicted_tokens.append([])
    matches, true_aa, predicted_aa = aa_match_batch(
        truth_tokens, predicted_tokens, masses
    )
    return (
        int(sum(row[0].sum() for row in matches)),
        int(true_aa),
        int(predicted_aa),
        int(sum(bool(row[1]) for row in matches)),
        len(truths),
    )


class CountRatio(Metric):
    """Sum counts across batches/ranks before dividing, including a short tail."""

    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state(
            "numerator", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum"
        )
        self.add_state(
            "denominator",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def update(self, numerator, denominator):
        self.numerator += int(numerator)
        self.denominator += int(denominator)

    def compute(self):
        return self.numerator.double() / self.denominator.clamp_min(1)
