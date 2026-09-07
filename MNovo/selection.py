"""Validate requested spectrum subsets before starting inference."""

import numpy as np


def validate_selection(indices, total, max_samples=0, task="denovo"):
    if max_samples < 0:
        raise ValueError("--max-samples must be non-negative (0 means all).")
    if task == "eval" and (indices is not None or max_samples):
        raise ValueError(
            "eval requires all input spectra; subset parameters are unsupported."
        )
    if indices is not None:
        indices = np.asarray(indices)
        if indices.ndim != 1 or not indices.size:
            raise ValueError("--indices must be a non-empty one-dimensional array.")
        if indices.dtype.kind not in "iu":
            raise ValueError("--indices must contain integers, without coercion.")
        if np.any(indices < 0) or np.any(indices >= total):
            raise ValueError("--indices contains negative or out-of-range indices.")
        if np.unique(indices).size != indices.size:
            raise ValueError("--indices must be unique.")
        indices = indices.astype(np.int64)
        return indices[:max_samples] if max_samples else indices
    if max_samples:
        return np.arange(min(max_samples, total), dtype=np.int64)
    return None
