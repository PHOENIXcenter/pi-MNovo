"""Training and testing functionality for the de novo peptide sequencing
model."""
import csv
import glob
import hashlib
import json
import logging
import operator
import os
import tempfile
import uuid
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
from depthcharge.data import AnnotatedSpectrumIndex, SpectrumIndex
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.profiler import SimpleProfiler

from .. import utils
from .db_dataloader import DeNovoDataModule
from .model import Spec2Pep



logger = logging.getLogger("MNovo")


def _stable_index_path(
    tmp_dir: str,
    prefix: str,
    filenames: List[str],
    config: Dict[str, Any],
    annotated: bool,
) -> str:
    """Build a deterministic LMDB index path for a fixed input file set."""
    payload = {
        "prefix": prefix,
        "annotated": bool(annotated),
        "ms_level": 2,
        "max_charge": int(config["max_charge"]),
        "files": [
            {
                "path": os.path.abspath(fn),
                "size": os.path.getsize(fn),
                "mtime_ns": os.stat(fn).st_mtime_ns,
            }
            for fn in sorted(filenames)
        ],
    }
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return os.path.join(tmp_dir, f"{prefix}_{digest}.lmdb")


def _maybe_reuse_index(index_path: str, filenames: Optional[List[str]]) -> Optional[List[str]]:
    """Return None for filenames when an existing index should be reused."""
    if filenames is not None and os.path.exists(index_path):
        logger.info("Reusing existing spectrum index: %s", index_path)
        return None
    return filenames


class MetricsCsvCallback(pl.callbacks.Callback):
    """Append validation/test metrics to a local CSV file after each epoch."""

    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path
        self.last_train_loss = float("nan")
        self.last_grad_norm = float("nan")
        self.last_lr = float("nan")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    @staticmethod
    def _metric_value(metrics: Dict[str, Any], key: str) -> float:
        value = metrics.get(key)
        if value is None:
            return float("nan")
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "item"):
            value = value.item()
        return float(value)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        value = outputs
        if isinstance(outputs, dict):
            value = outputs.get("loss")
        if value is not None:
            self.last_train_loss = self._metric_value({"loss": value}, "loss")
        if trainer.optimizers:
            self.last_lr = float(trainer.optimizers[0].param_groups[0].get("lr", float("nan")))

    def on_before_optimizer_step(self, trainer, pl_module, optimizer, optimizer_idx=None):
        total_norm = 0.0
        for param in pl_module.parameters():
            if param.grad is not None:
                param_norm = param.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        self.last_grad_norm = total_norm ** 0.5
        if optimizer.param_groups:
            self.last_lr = float(optimizer.param_groups[0].get("lr", float("nan")))

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metrics = trainer.callback_metrics
        train_loss = self.last_train_loss
        for train_key in ("train/CELoss", "train/CELoss_epoch", "train/loss", "loss"):
            if train_key in metrics:
                train_loss = self._metric_value(metrics, train_key)
                break
        row = {
            "step": int(trainer.global_step),
            "epoch": int(trainer.current_epoch),
            "learning_rate": self.last_lr,
            "train_loss": train_loss,
            "grad_norm": self.last_grad_norm,
            "valid_CELoss": self._metric_value(metrics, "valid/CELoss"),
            "valid_aa_precision": self._metric_value(metrics, "valid/aa_precision"),
            "valid_aa_recall": self._metric_value(metrics, "valid/aa_recall"),
            "valid_peptide_exact": self._metric_value(metrics, "valid/pep_recall"),
            "test_CELoss": self._metric_value(metrics, "test/CELoss"),
            "test_aa_precision": self._metric_value(metrics, "test/aa_precision"),
            "test_aa_recall": self._metric_value(metrics, "test/aa_recall"),
            "test_peptide_exact": self._metric_value(metrics, "test/pep_recall"),
        }
        write_header = not os.path.exists(self.path)
        with open(self.path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)


def predict(
    peak_path: str,
    model_filename: str,
    config: Dict[str, Any],
    out_writer: None,
) -> None:
    """
    Predict peptide sequences with a trained mnovo model.

    Parameters
    ----------
    peak_path : str
        The path with peak files for predicting peptide sequences.
    model_filename : str
        The file name of the model weights (.ckpt file).
    config : Dict[str, Any]
        The configuration options.
    out_writer : ms_io.MztabWriter
        The mzTab writer to export the prediction results.
    """
    _execute_existing(peak_path, model_filename, config, False, out_writer)


def evaluate(peak_path: str, model_filename: str, config: Dict[str,
                                                               Any]) -> None:
    """
    Evaluate peptide sequence predictions from a trained mnovo model.

    Parameters
    ----------
    peak_path : str
        The path with peak files for predicting peptide sequences.
    model_filename : str
        The file name of the model weights (.ckpt file).
    config : Dict[str, Any]
        The configuration options.
    """
    _execute_existing(peak_path, model_filename, config, True)


def _execute_existing(
    peak_path: str,
    model_filename: str,
    config: Dict[str, Any],
    annotated: bool,
    out_writer = None,
) -> None:
    """
    Predict peptide sequences with a trained mnovo model with/without
    evaluation.

    Parameters
    ----------
    peak_path : str
        The path with peak files for predicting peptide sequences.
    model_filename : str
        The file name of the model weights (.ckpt file).
    config : Dict[str, Any]
        The configuration options.
    annotated : bool
        Whether the input peak files are annotated (execute in evaluation mode)
        or not (execute in prediction mode only).
    out_writer : Optional[ms_io.MztabWriter]
        The mzTab writer to export the prediction results.
    """
    # Load the trained model.
    if not os.path.isfile(model_filename):
        logger.error(
            "Could not find the trained model weights at file %s",
            model_filename,
        )
        raise FileNotFoundError("Could not find the trained model weights")
    model = Spec2Pep().load_from_checkpoint(
        model_filename,
        PMC_enable=config["PMC_enable"],
        mass_control_tol=config["mass_control_tol"],

        dim_model=config["dim_model"],
        n_head=config["n_head"],
        dim_feedforward=config["dim_feedforward"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
        dim_intensity=config["dim_intensity"],
        custom_encoder=config["custom_encoder"],
        max_length=config["max_length"],
        residues=config["residues"],
        max_charge=config["max_charge"],
        precursor_mass_tol=config["precursor_mass_tol"],
        isotope_error_range=config["isotope_error_range"],
        n_beams=config["n_beams"],
        n_log=config["n_log"],
        out_writer=out_writer,
    )
    # Read the MS/MS spectra for which to predict peptide sequences.
    if annotated:
        peak_ext = (".mgf", ".h5", ".hdf5")
    else:
        peak_ext = (".mgf", ".mzml", ".mzxml", ".h5", ".hdf5")
    logger.info("Reading spectra from %s", peak_path)
    if len(peak_filenames := _get_peak_filenames(peak_path, peak_ext)) == 0:
        logger.error("Could not find peak files from %s", peak_path)
        raise FileNotFoundError("Could not find peak files")
    peak_is_not_index = any(
        [os.path.splitext(fn)[1] in (".mgf", ".mzxml", ".mzml") for fn in peak_filenames])
    class MyDirectory:
        def __init__(self, sdir=None):
            self.name = sdir
        def cleanup(self):
            pass
    if not config["temp_dir_auto"]: # if we don't use auto-generated temp dir
        tmp_base = config.get("temp_dir_path")
        if not tmp_base:
            raise ValueError("temp_dir_path is required when temp_dir_auto is false")
        os.makedirs(tmp_base, exist_ok=True)
        tmp_dir = MyDirectory(tmp_base)
    else:
        tmp_dir = tempfile.TemporaryDirectory()
    if peak_is_not_index:
        index_path = [_stable_index_path(tmp_dir.name, "eval", peak_filenames, config, annotated)]
        peak_filenames = _maybe_reuse_index(index_path[0], peak_filenames)
    else:
        index_path = peak_filenames
        peak_filenames = None
    logger.debug("Input requires indexing: %s", peak_is_not_index)

    #SpectrumIdx = AnnotatedSpectrumIndex if annotated else SpectrumIndex
    valid_charge = np.arange(1, config["max_charge"] + 1)
    dataloader_params = dict(
        batch_size=config["predict_batch_size"],
        n_peaks=config["n_peaks"],
        min_mz=config["min_mz"],
        max_mz=config["max_mz"],
        min_intensity=config["min_intensity"],
        remove_precursor_tol=config["remove_precursor_tol"],
        n_workers=config["n_workers"],
        pin_memory=config.get("pin_memory", True),
        train_filenames = None,
        val_filenames = None,
        test_filenames = peak_filenames,


        train_index_path = None, #always a list, either a list containing one index path file or a list containing multiple db files
        val_index_path = None,
        test_index_path = index_path,
        annotated = annotated,
        valid_charge = valid_charge ,
        mode = "test"

    )
    # Initialize the data loader.
    dataModule = DeNovoDataModule(**dataloader_params)
    dataModule.prepare_data()
    dataModule.setup(stage="test")
    test_dataloader = dataModule.test_dataloader()

    # Create the Trainer object.
    trainer = pl.Trainer(
        enable_model_summary=True,
        accelerator="auto",
        auto_select_gpus=True,
        devices=_get_devices(),
        **_trainer_precision_kwargs(config),
        logger=config["logger"],
        max_epochs=config["max_epochs"],
        num_sanity_val_steps=config["num_sanity_val_steps"],
        strategy=_get_strategy(),
    )
    # Run the model with/without validation.
    run_trainer = trainer.validate if annotated else trainer.predict
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %d", pytorch_total_params)
    run_trainer(model,test_dataloader )
    # Clean up temporary files.
    tmp_dir.cleanup()


def train(
    peak_path: str,
    peak_path_val: str,
    peak_path_test: str,
    model_filename: str,
    config: Dict[str, Any],
) -> None:
    """
    Train a mnovo model.

    The model can be trained from scratch or by continuing training an existing
    model.

    Parameters
    ----------
    peak_path : str
        The path with peak files to be used as training data.
    peak_path_val : str
        The path with peak files to be used as validation data.
    peak_path_test : str
        The path with peak files to be used as testing data.
    model_filename : str
        The file name of the model weights (.ckpt file).
    config : Dict[str, Any]
        The configuration options.
    """
    # Read the MS/MS spectra to use for training and validation.
    ext = (".mgf", ".h5", ".hdf5")
    logger.info("Training spectra: %s", peak_path)
    #check if input peak files are valid and exist
    if len(train_filenames := _get_peak_filenames(peak_path, ext)) == 0:
        logger.error("Could not find training peak files from %s", peak_path)
        raise FileNotFoundError("Could not find training peak files")
    train_is_not_index = any([
        os.path.splitext(fn)[1] in (".mgf", ".mzxml", ".mzml") for fn in train_filenames
    ])  #check if training peak files are raw files or lmdb_index files

    if (peak_path_val is None
            or len(val_filenames := _get_peak_filenames(peak_path_val, ext))
            == 0):
        logger.error("Could not find validation peak files from %s",
                     peak_path_val)
        raise FileNotFoundError("Could not find validation peak files")
    val_is_not_index = any(
        [os.path.splitext(fn)[1] in (".mgf", ".mzxml", ".mzml") for fn in val_filenames])

    if (peak_path_test is None
            or len(test_filenames := _get_peak_filenames(peak_path_test, ext))
            == 0):
        logger.error("Could not find testing peak files from %s",
                     peak_path_test)
        raise FileNotFoundError("Could not find testing peak files")
    test_is_not_index = any(
        [os.path.splitext(fn)[1] in (".mgf", ".mzxml", ".mzml") for fn in test_filenames])



    class MyDirectory:
        def __init__(self, sdir=None):
            self.name = sdir
        def cleanup(self):
            pass
    if not config["temp_dir_auto"]: # if we don't use auto-generated temp dir
        tmp_base = config.get("temp_dir_path")
        if not tmp_base:
            raise ValueError("temp_dir_path is required when temp_dir_auto is false")
        os.makedirs(tmp_base, exist_ok=True)
        tmp_dir = MyDirectory(tmp_base)
    else:
        tmp_dir = tempfile.TemporaryDirectory()


    if train_is_not_index:
        train_index_path = [_stable_index_path(tmp_dir.name, "Train", train_filenames, config, True)]
        train_filenames = _maybe_reuse_index(train_index_path[0], train_filenames)
    else:
        train_index_path = train_filenames
        train_filenames = None



    if val_is_not_index:
        val_index_path = [_stable_index_path(tmp_dir.name, "valid", val_filenames, config, True)]
        val_filenames = _maybe_reuse_index(val_index_path[0], val_filenames)
    else:
        val_index_path = val_filenames
        val_filenames = None
    if test_is_not_index:
        test_index_path = [_stable_index_path(tmp_dir.name, "test", test_filenames, config, True)]
        test_filenames = _maybe_reuse_index(test_index_path[0], test_filenames)
    else:
        test_index_path = test_filenames
        test_filenames = None

    valid_charge = np.arange(1, config["max_charge"] + 1)
    '''
    train_index = AnnotatedSpectrumIndex(train_idx_fn,
                                        train_filenames,
                                        valid_charge=valid_charge)
    if val_is_index:
        val_idx_fn, val_filenames = val_filenames[0], None
    else:
        val_idx_fn = os.path.join(tmp_dir.name, f"Valid_{uuid.uuid4().hex}.hdf5")
    val_index = AnnotatedSpectrumIndex(val_idx_fn,
                                       val_filenames,
                                       valid_charge=valid_charge)
    if test_is_index:
        test_idx_fn, test_filenames = test_filenames[0], None
    else:
        test_idx_fn = os.path.join(tmp_dir.name, f"Test_{uuid.uuid4().hex}.hdf5")
    test_index = AnnotatedSpectrumIndex(test_idx_fn,
                                       test_filenames,
                                       valid_charge=valid_charge)
    '''
    # Initialize the data loaders.
    dataloader_params = dict(
        batch_size=config["train_batch_size"],
        n_peaks=config["n_peaks"],
        min_mz=config["min_mz"],
        max_mz=config["max_mz"],
        min_intensity=config["min_intensity"],
        remove_precursor_tol=config["remove_precursor_tol"],
        n_workers=config["n_workers"],
        train_filenames = train_filenames,
        val_filenames = val_filenames,
        test_filenames = test_filenames,


        train_index_path = train_index_path, #always a list, either a list containing one index path file or a list containing multiple db files
        val_index_path = val_index_path,
        test_index_path = test_index_path,
        annotated = True,
        valid_charge = valid_charge ,
        mode = "fit",
        train_num_samples = config.get("train_num_samples")

    )
    dataModule = DeNovoDataModule(**dataloader_params)
    dataModule.prepare_data()
    dataModule.setup()
    train_dataloader=dataModule.train_dataloader()
    #train_loader = DeNovoDataModule(train_index=train_index,
                                  #  **dataloader_params)
    #train_loader.setup()
    #train_dataloader=train_loader.train_dataloader()


    #val_loader = DeNovoDataModule(valid_index=val_index, **dataloader_params)
    #val_loader.setup()

    #test_loader = DeNovoDataModule(valid_index=test_index, **dataloader_params)
    #test_loader.setup()

    # Set warmup_iters & max_iters
    # Author: Sheng Xu
    # Date: 20230202
    device_count = max(torch.cuda.device_count(), 1)
    batches_per_device = int(
        len(train_dataloader)
        / (device_count * config["accumulate_grad_batches"])
    )
    config["warmup_iters"] = batches_per_device * config["warm_up_epochs"]
    config["max_iters"] = batches_per_device * int(config["max_epochs"])
    logger.info("Training batches per epoch: %d", len(train_dataloader))
    # Initialize the model.
    ctc_params = dict(model_path=None,  #to change
                                      alpha=0, beta=0,
                                      cutoff_top_n=100,
                                      cutoff_prob= 1.0,
                                      beam_width=config["n_beams"],
                                      num_processes=4,
                                      log_probs_input = False)
    model_params = dict(
        PMC_enable = config["PMC_enable"],
        mass_control_tol = config["mass_control_tol"],
        custom_ctc_loss = config["custom_ctc"],
        dim_model=config["dim_model"],
        n_head=config["n_head"],
        dim_feedforward=config["dim_feedforward"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
        dim_intensity=config["dim_intensity"],
        custom_encoder=config["custom_encoder"],
        max_length=config["max_length"],
        residues=config["residues"],
        max_charge=config["max_charge"],
        precursor_mass_tol=config["precursor_mass_tol"],
        isotope_error_range=config["isotope_error_range"],
        n_beams=config["n_beams"],
        n_log=config["n_log"],
        tb_summarywriter=config["tb_summarywriter"],
        warmup_iters=config["warmup_iters"],
        max_iters=config["max_iters"],
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        finetune_strategy=config.get("finetune_strategy", "full"),
        backbone_lr_multiplier=config.get("backbone_lr_multiplier", 0.1),
        output_module_prefixes=config.get("output_module_prefixes", []),
        freeze_module_prefixes=config.get("freeze_module_prefixes", []),
        ctc_dic = ctc_params
    )

    if config["train_from_scratch"]:
        model = Spec2Pep(**model_params)
    else:
        logger.info("Training from checkpoint...")
        model_filename = model_filename or config.get("load_file_name")
        if not model_filename:
            raise ValueError(
                "Provide --checkpoint unless train_from_scratch is true."
            )
        if not os.path.isfile(model_filename):
            logger.error(
                "Could not find the model weights at file %s to continue "
                "training",
                model_filename,
            )
            raise FileNotFoundError(
                "Could not find the model weights to continue training")
        model = Spec2Pep().load_from_checkpoint(model_filename, **model_params)
    # Create the Trainer object and (optionally) a checkpoint callback to
    # periodically save the model.
    if config["save_model"]:
        callbacks = [
            pl.callbacks.ModelCheckpoint(
                dirpath=config["model_save_folder_path"],
                save_top_k=-1,
                save_weights_only=False,
                every_n_train_steps=config["every_n_train_steps"],
            )
        ]
    else:
        callbacks = []

    if config["SWA"]:
        callbacks.append(pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2))
    if config.get("metrics_csv_path"):
        callbacks.append(MetricsCsvCallback(config["metrics_csv_path"]))

    logger.info("Available CUDA devices: %d", torch.cuda.device_count())
    trainer_limits = {
        key: config[key]
        for key in ("limit_train_batches", "limit_val_batches", "limit_test_batches", "max_steps")
        if key in config and config[key] is not None
    }
    if config["val_interval"] == 1:
        trainer = pl.Trainer(


            reload_dataloaders_every_n_epochs=1,
            enable_model_summary= True,
            accelerator="auto",
            auto_select_gpus=True,
            callbacks=callbacks,
            enable_checkpointing=config["save_model"],
            devices=_get_devices(),
            **_trainer_precision_kwargs(config),
            num_nodes=config["n_nodes"],

            logger=None,
            max_epochs=config["max_epochs"],
            num_sanity_val_steps=config["num_sanity_val_steps"],
            strategy= _get_strategy(),
            gradient_clip_val=config["gradient_clip_val"],
            gradient_clip_algorithm=config["gradient_clip_algorithm"],
            accumulate_grad_batches=config["accumulate_grad_batches"],
            sync_batchnorm=config["sync_batchnorm"],
            **trainer_limits,
        )
    else:
        trainer = pl.Trainer(
            val_check_interval=config["val_interval"],

            reload_dataloaders_every_n_epochs=1,
            enable_model_summary= True,
            accelerator="auto",
            auto_select_gpus=True,
            callbacks=callbacks,
            enable_checkpointing=config["save_model"],
            devices=_get_devices(),
            **_trainer_precision_kwargs(config),
            num_nodes=config["n_nodes"],

            logger=None,
            max_epochs=config["max_epochs"],
            num_sanity_val_steps=config["num_sanity_val_steps"],
            strategy= _get_strategy(),
            gradient_clip_val=config["gradient_clip_val"],
            gradient_clip_algorithm=config["gradient_clip_algorithm"],
            accumulate_grad_batches=config["accumulate_grad_batches"],
            sync_batchnorm=config["sync_batchnorm"],
            **trainer_limits,
        )

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %d", pytorch_total_params)
    # Train the model.
    if config["train_from_resume"] == True and config["train_from_scratch"] == False:
        trainer.fit(model, datamodule=dataModule, ckpt_path=model_filename)
    else:
        trainer.fit(model,
                datamodule=dataModule)
    # Clean up temporary files.
    tmp_dir.cleanup()


def _get_peak_filenames(
    path: str, supported_ext: Iterable[str] = (".mgf", )) -> List[str]:
    """
    Get all matching peak file names from the path pattern.

    Performs cross-platform path expansion akin to the Unix shell (glob, expand
    user, expand vars).

    Parameters
    ----------
    path : str
        The path pattern.
    supported_ext : Iterable[str]
        Extensions of supported peak file formats. Default: MGF.

    Returns
    -------
    List[str]
        The peak file names matching the path pattern.
    """
    if path is None:
        return []

    supported_ext = tuple(ext.lower() for ext in supported_ext)
    files = []
    for one_path in str(path).split("&"):
        one_path = os.path.expanduser(one_path.strip())
        one_path = os.path.expandvars(one_path)
        if not one_path:
            continue
        if os.path.isdir(one_path):
            one_path = os.path.join(one_path, "**", "*")
        files.extend(glob.glob(one_path, recursive=True))

    return sorted({
        fn
        for fn in files
        if os.path.isfile(fn)
        and os.path.splitext(fn.lower())[1] in supported_ext
    })


def _get_strategy() -> Optional[DDPStrategy]:
    """
    Get the strategy for the Trainer.

    The DDP strategy works best when multiple GPUs are used. It can work for
    CPU-only, but definitely fails using MPS (the Apple Silicon chip) due to
    Gloo.

    Returns
    -------
    Optional[DDPStrategy]
        The strategy parameter for the Trainer.
    """
    if torch.cuda.device_count() > 1:
        return DDPStrategy(find_unused_parameters=False, static_graph=True)

    return None


def _trainer_precision_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    precision = config.get("trainer_precision")
    if precision is None or precision == "":
        return {}
    if precision == "bf16-mixed":
        precision = "bf16"
    return {"precision": precision}


def _get_devices() -> Union[int, str]:
    """
    Get the number of GPUs/CPUs for the Trainer to use.

    Returns
    -------
    Union[int, str]
        The number of GPUs/CPUs to use, or "auto" to let PyTorch Lightning
        determine the appropriate number of devices.
    """

    if any(
            operator.attrgetter(device + ".is_available")(torch)()
            for device in ["cuda", "backends.mps"]):
        return -1
    elif not (n_workers := utils.n_workers()):
        return "auto"
    else:
        return n_workers
