"""Compatibility entry point for python -m MNovo.multi_gpu."""

if __name__ == "__main__":
    from runpy import run_module

    run_module("MNovo.inference.multi_gpu", run_name="__main__")
else:
    from MNovo.inference.multi_gpu import (
        batch_shards, merge_features, merge_predictions, run_selected_devices, selected_devices,
    )
