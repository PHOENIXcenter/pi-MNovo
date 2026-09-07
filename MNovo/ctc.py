"""CTC target feasibility, independent of the decoder implementation."""


def required_steps(tokens):
    """Repeated adjacent labels require an intervening blank timestep."""
    return len(tokens) + sum(a == b for a, b in zip(tokens, tokens[1:]))


def validate_targets(targets, lengths, time_steps, blank):
    for index, (target, length) in enumerate(zip(targets, lengths)):
        labels = target[: int(length)].tolist()
        required = required_steps(labels)
        if not labels or blank in labels or required > time_steps:
            raise ValueError(
                f"CTC-unalignable target at batch index {index}: "
                f"labels={len(labels)}, required_steps={required}, T={time_steps}. "
                "Audit and explicitly exclude unsupported training targets; "
                "retain them in the final evaluation denominator."
            )
