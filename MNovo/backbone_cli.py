"""Internal command-line entry point for backbone training."""

import datetime
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore", category=DeprecationWarning)

import click
import torch
import yaml
from pytorch_lightning.lite import LightningLite

from . import utils
from .denovo import backbone_runner


logger = logging.getLogger("MNovo")


@click.command()
@click.option(
    "--mode",
    required=True,
    default="train",
    help="Internal mode; only backbone training is supported.",
    type=click.Choice(["train"]),
)
@click.option(
    "--model",
    help="The file name of the model weights (.ckpt file).",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--peak_path",
    required=True,
    help="The file path with peak files for predicting peptide sequences or "
    "training MNovo.",
)
@click.option(
    "--peak_path_val",
    help="The file path with peak files to be used as validation data during training.",
)
@click.option(
    "--peak_path_test",
    help="The file path with peak files to be used as testing data during training.",
)
@click.option(
    "--config",
    help="The file name of the configuration file with custom options. If not "
    "specified, a default configuration will be used.",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--output",
    help="Training run directory for logs, metrics, and checkpoints.",
    type=click.Path(file_okay=False),
)
def main(
    mode: str,
    model: Optional[str],
    peak_path: str,
    peak_path_val: Optional[str],
    peak_path_test: Optional[str],
    config: Optional[str],
    output: Optional[str],
):
    if output is None:
        output_dir = Path.cwd() / (
            f"MNovo_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        )
    else:
        output_dir = Path(output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging.
    logging.captureWarnings(True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    log_formatter = logging.Formatter(
        "{asctime} {levelname} [{name}/{processName}] {module}.{funcName} : {message}",
        style="{",
    )
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(log_formatter)
    root.addHandler(console_handler)
    file_handler = logging.FileHandler(output_dir / "train.log")
    file_handler.setFormatter(log_formatter)
    root.addHandler(file_handler)
    # Disable dependency non-critical log messages.
    logging.getLogger("depthcharge").setLevel(logging.INFO)
    logging.getLogger("h5py").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Read parameters from the config file.
    if config is None:
        config = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "config.yaml"
        )
    config_fn = config
    with open(config) as f_in:
        config = yaml.safe_load(f_in)
    # Ensure that the config values have the correct type.
    config_types = dict(
        random_seed=int,
        n_peaks=int,
        min_mz=float,
        max_mz=float,
        min_intensity=float,
        remove_precursor_tol=float,
        max_charge=int,
        precursor_mass_tol=float,
        isotope_error_range=lambda min_max: (int(min_max[0]), int(min_max[1])),
        dim_model=int,
        n_head=int,
        dim_feedforward=int,
        n_layers=int,
        dropout=float,
        dim_intensity=int,
        max_length=int,
        n_log=int,
        warmup_iters=int,
        max_iters=int,
        learning_rate=float,
        weight_decay=float,
        train_batch_size=int,
        train_num_samples=int,
        predict_batch_size=int,
        n_beams=int,
        max_epochs=int,
        num_sanity_val_steps=int,
        n_workers=int,
        limit_train_batches=int,
        limit_val_batches=int,
        limit_test_batches=int,
        train_from_scratch=bool,
        save_model=bool,
        model_save_folder_path=str,
        save_weights_only=bool,
        every_n_train_steps=int,
    )
    for k, t in config_types.items():
        try:
            if k in config and config[k] is not None:
                config[k] = t(config[k])
        except (TypeError, ValueError) as e:
            logger.error("Incorrect type for configuration value %s: %s", k, e)
            raise TypeError(f"Incorrect type for configuration value {k}: {e}")
    config["residues"] = {
        str(aa): float(mass) for aa, mass in config["residues"].items()
    }
    # Add extra configuration options and scale by the number of GPUs.
    n_gpus = torch.cuda.device_count()
    config["n_workers"] = (
        int(config["n_workers"])
        if config.get("n_workers") is not None
        else utils.n_workers()
    )
    if n_gpus > 1:
        config["train_batch_size"] = config["train_batch_size"] // n_gpus

    config["model_save_folder_path"] = str(output_dir / "checkpoints")
    if not config.get("metrics_csv_path"):
        config["metrics_csv_path"] = str(output_dir / "metrics.csv")

    import random

    if config["random_seed"] == -1:
        config["random_seed"] = random.randint(1, 9999)
    LightningLite.seed_everything(seed=config["random_seed"], workers=True)

    # Log the active configuration.
    logger.debug("mode = %s", mode)
    logger.debug("model = %s", model)
    logger.debug("peak_path = %s", peak_path)
    logger.debug("peak_path_val = %s", peak_path_val)
    logger.debug("peak_path_test = %s", peak_path_test)
    logger.debug("config = %s", config_fn)
    logger.debug("output = %s", output_dir)
    for key, value in config.items():
        logger.debug("%s = %s", str(key), str(value))

    logger.info("Train the MNovo backbone.")
    backbone_runner.train(peak_path, peak_path_val, peak_path_test, model, config)


if __name__ == "__main__":
    main()
