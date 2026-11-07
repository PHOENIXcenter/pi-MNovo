"""User-adjustable decoding and peak filters; checkpoint architecture stays fixed."""

from math import isfinite


EDITABLE = {"PMC_enable", "mass_control_tol", "n_peaks", "min_mz", "max_mz",
            "min_intensity", "remove_precursor_tol"}


def resolve_settings(base, overrides):
    """Validate before applying overrides, including paired lower/upper m/z limits."""
    if unknown := overrides.keys() - EDITABLE:
        raise ValueError(f"Cannot override frozen inference fields: {sorted(unknown)}")
    for key, value in overrides.items():
        if key == "PMC_enable":
            valid = type(value) is bool
        elif key == "n_peaks":
            valid = type(value) is int and value > 0
        else:
            valid = type(value) in (int, float) and isfinite(value) and value >= 0
            if key == "mass_control_tol":
                valid = valid and value > 0
            if key == "min_intensity":
                valid = valid and value <= 1
        if not valid:
            raise ValueError(f"Invalid {key}: {value!r}")
    config = {**base, **overrides}
    if {"min_mz", "max_mz"} <= config.keys() and config["min_mz"] >= config["max_mz"]:
        raise ValueError("min_mz must be less than max_mz")
    return config
