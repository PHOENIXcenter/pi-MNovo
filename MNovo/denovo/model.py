# The backbone returns per-position predictions; training hooks also consume reference labels.
# 骨干输出位置级预测；训练钩子还使用参考标注。
"""A de novo peptide sequencing model."""

from concurrent.futures import ThreadPoolExecutor
import logging
import re
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import depthcharge.masses
import einops
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.tensorboard import SummaryWriter
from . import mass_con
from ..components import ModelMixin, PeptideDecoder, SpectrumEncoder
from .ctc_beam_search import CTCBeamSearchDecoder
from .metric_counts import CountRatio, match_counts, metric_components
from MNovo.training_ctc import masked_ctc_loss
from MNovo.distributed_training import distributed_ctc, check_finite_gradients

AA_MASSES = {
    "G": 57.021464,
    "A": 71.037114,
    "S": 87.032028,
    "P": 97.052764,
    "V": 99.068414,
    "T": 101.04767,
    "C+57.021": 160.030649,
    "L": 113.084064,
    "I": 113.084064,
    "N": 114.042927,
    "D": 115.026943,
    "Q": 128.058578,
    "K": 128.094963,
    "E": 129.042593,
    "M": 131.040485,
    "H": 137.058912,
    "F": 147.068414,
    "R": 156.101111,
    "Y": 163.063329,
    "W": 186.079313,
    "M+15.995": 147.0354,
    "N+0.984": 115.026943,
    "Q+0.984": 129.042594,
    "+42.011": 42.010565,
    "+43.006": 43.005814,
    "-17.027": -17.026549,
    "+43.006-17.027": 25.980265,
    "_": 0,
}

logger = logging.getLogger(__name__)


def calculate_residue_mass(sequence):
    sequence = sequence.replace("I", "L")
    sequence = re.split(r"(?<=.)(?=[A-Z])", sequence)
    total = 0
    for each in sequence:
        try:
            total += AA_MASSES[each]
        except KeyError:
            h1 = each.count("+42.011")
            h2 = each.count("+43.006")
            h3 = each.count("-17.027")
            total += h1 * 42.010565 + h2 * 43.005814 + h3 * -17.026549
            each = each.replace("+42.011", "")
            each = each.replace("+43.006", "")
            each = each.replace("-17.027", "")
            if each:
                total += AA_MASSES[each]

    return total, sequence


def collapse_repeated_tokens(index_list: List[int]) -> List[int]:
    """
    Eliminate repeated index in list. e.g., [1, 1, 2, 2, 3] --> [1, 2, 3]
    """
    return [
        a for a, b in zip(index_list, index_list[1:] + [not index_list[-1]]) if a != b
    ]


def ctc_post_processing(sentence_index: List[int]) -> List[int]:
    """
    Merge repetitive tokens, then eliminate <blank> tokens and <pad> tokens.
    The input sentence_index is expected to be a 1-D index list
    """
    sentence_index = collapse_repeated_tokens(sentence_index)
    temp = []
    for each in sentence_index:
        if each != 27:
            temp.append(each)
    return temp


class Spec2Pep(pl.LightningModule, ModelMixin):
    """
    A Transformer model for de novo peptide sequencing.

    Use this model in conjunction with a pytorch-lightning Trainer.

    Parameters
    ----------
    dim_model : int
        The latent dimensionality used by the transformer model.
    n_head : int
        The number of attention heads in each layer. ``dim_model`` must be
        divisible by ``n_head``.
    dim_feedforward : int
        The dimensionality of the fully connected layers in the transformer
        model.
    n_layers : int
        The number of transformer layers.
    dropout : float
        The dropout probability for all layers.
    dim_intensity : Optional[int]
        The number of features to use for encoding peak intensity. The remaining
        (``dim_model - dim_intensity``) are reserved for encoding the m/z value.
        If ``None``, the intensity will be projected up to ``dim_model`` using a
        linear layer, then summed with the m/z encoding for each peak.
    custom_encoder : Optional[Union[SpectrumEncoder, PairedSpectrumEncoder]]
        A pretrained encoder to use. The ``dim_model`` of the encoder must be
        the same as that specified by the ``dim_model`` parameter here.
    max_length : int
        The maximum peptide length to decode.
    residues: Union[Dict[str, float], str]
        The amino acid dictionary and their masses. By default ("canonical) this
        is only the 20 canonical amino acids, with cysteine carbamidomethylated.
        If "massivekb", this dictionary will include the modifications found in
        MassIVE-KB. Additionally, a dictionary can be used to specify a custom
        collection of amino acids and masses.
    max_charge : int
        The maximum precursor charge to consider.
    precursor_mass_tol : float, optional
        The maximum allowable precursor mass tolerance (in ppm) for correct
        predictions.
    isotope_error_range : Tuple[int, int]
        Take into account the error introduced by choosing a non-monoisotopic
        peak for fragmentation by not penalizing predicted precursor m/z's that
        fit the specified isotope error:
        `abs(calc_mz - (precursor_mz - isotope * 1.00335 / precursor_charge))
        < precursor_mass_tol`
    n_beams: int
        Number of beams used during beam search decoding.
    n_log : int
        The number of epochs to wait between logging messages.
    tb_summarywriter: Optional[str]
        Folder path to record performance metrics during training. If ``None``,
        don't use a ``SummaryWriter``.
    warmup_iters: int
        The number of warm up iterations for the learning rate scheduler.
    max_iters: int
        The total number of iterations for the learning rate scheduler.
    out_writer: Optional[str]
        The output writer for the prediction results.
    **kwargs : Dict
        Additional keyword arguments passed to the Adam optimizer.
    """

    def __init__(
        self,
        custom_ctc_loss=True,
        dim_model: int = 512,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 9,
        dropout: float = 0.0,
        dim_intensity: Optional[int] = None,
        custom_encoder: Optional[SpectrumEncoder] = None,
        max_length: int = 100,
        residues: Union[Dict[str, float], str] = "canonical",
        max_charge: int = 5,
        precursor_mass_tol: float = 50,
        isotope_error_range: Tuple[int, int] = (0, 1),
        n_beams: int = 5,
        n_log: int = 10,
        mass_control_tol: float = 0.1,
        tb_summarywriter: Optional[torch.utils.tensorboard.SummaryWriter] = None,
        warmup_iters: int = 100_000,
        max_iters: int = 600_000,
        out_writer=None,
        ctc_dic: Optional[dict] = None,
        PMC_enable=True,
        finetune_strategy: str = "full",
        backbone_lr_multiplier: float = 0.1,
        output_module_prefixes=None,
        freeze_module_prefixes=None,
        ctc_max_invalid_fraction: float = 0.05,
        ctc_guard_min_samples: int = 1000,
        ctc_max_empty_batches: int = 10,
        lr_schedule: str = "cosine",
        **kwargs: Dict,
    ):
        super().__init__()
        self.mass_control_tol = mass_control_tol
        self.save_hyperparameters()
        self.ctc_dic = dict(ctc_dic or {})
        self.PMC_enable = PMC_enable

        self.ctc_dic["beam"] = n_beams

        # Build the model.
        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            self.encoder = SpectrumEncoder(
                dim_model=dim_model,
                n_head=n_head,
                dim_feedforward=dim_feedforward,
                n_layers=n_layers,
                dropout=dropout,
                dim_intensity=dim_intensity,
            )
        self.decoder = PeptideDecoder(
            dim_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
            residues=residues,
            max_charge=max_charge,
            max_pep_len=max_length,
        )
        if self.PMC_enable:
            from MNovo.pmc_schema import validate_pmc_schema

            validate_pmc_schema(self.decoder, max_length)
        self.n_layers = n_layers
        self.ctc_decoder = CTCBeamSearchDecoder(self.decoder, self.ctc_dic)
        self.ctcloss = torch.nn.CTCLoss(
            blank=self.decoder.get_blank_idx(), reduction="none", zero_infinity=False
        )
        self.sequence_metrics = torch.nn.ModuleDict(
            {
                f"{stage}_{name}": CountRatio()
                for stage in ("train", "valid")
                for name in ("aa_precision", "aa_recall", "pep_recall")
            }
        )
        # Optimizer settings.
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        self.opt_kwargs = kwargs
        self.finetune_strategy = finetune_strategy
        self.backbone_lr_multiplier = backbone_lr_multiplier
        self.output_module_prefixes = tuple(output_module_prefixes or [])
        self.freeze_module_prefixes = tuple(freeze_module_prefixes or [])
        self.custom_ctc_loss = custom_ctc_loss
        if lr_schedule not in ("cosine", "warmup_constant"):
            raise ValueError("Unknown learning-rate schedule")
        self.lr_schedule = lr_schedule
        if not 0 <= ctc_max_invalid_fraction <= 1:
            raise ValueError("ctc_max_invalid_fraction must be in [0, 1]")
        if ctc_guard_min_samples < 1 or ctc_max_empty_batches < 1:
            raise ValueError("CTC guard counts must be positive")
        self.ctc_max_invalid_fraction = ctc_max_invalid_fraction
        self.ctc_guard_min_samples = ctc_guard_min_samples
        self.ctc_max_empty_batches = ctc_max_empty_batches
        self._ctc_counts = {}
        self._ctc_empty_batches = 0
        self._ctc_window_has_data = False
        self._ctc_last_update = False
        self._ctc_valid_count = 0

        # Data properties.
        self.max_length = max_length
        self.residues = residues
        self.precursor_mass_tol = precursor_mass_tol
        self.isotope_error_range = isotope_error_range
        self.n_beams = n_beams
        self.peptide_mass_calculator = depthcharge.masses.PeptideMass(self.residues)

        # Logging.
        self.n_log = n_log
        if tb_summarywriter is not None:
            self.tb_summarywriter = SummaryWriter(tb_summarywriter)
        else:
            self.tb_summarywriter = tb_summarywriter

        # Output writer during predicting.
        self.out_writer = out_writer

    def forward(
        self,
        spectra: torch.Tensor,
        precursors: torch.Tensor,
        _true_peps,
    ) -> Tuple[List[List[str]], torch.Tensor]:
        """Return decoded peptides and exp(-beam score) for each spectrum.

        PMC can replace a beam sequence; the accompanying scores still come from beam decoding.
        """
        output_logits, _, _ = self.decoder(None, precursors, *self.encoder(spectra))
        top_tokens, beamscores = self.ctc_decoder.decode(F.softmax(output_logits, -1))
        batchscores = 1 / torch.exp(beamscores)
        top_tokens_beam = top_tokens.tolist()
        batch_size = output_logits.shape[0]
        top_tokens = [[] for i in range(batch_size)]

        def worker(logits, mass, i):
            mass = mass.clone().detach()
            if not self.PMC_enable:
                top_tokens[i] = top_tokens_beam[i]
            else:
                mass_true = mass[0].item() - 18.01
                sequence = list(
                    filter((self.decoder.get_pad_idx()).__ne__, top_tokens_beam[i])
                )
                token_true = [self.decoder._idx2aa[each] for each in sequence]
                pred_mass, _sequence = calculate_residue_mass("".join(token_true))
                assert self.mass_control_tol > 0
                if abs(mass_true - pred_mass) < self.mass_control_tol:
                    top_tokens[i] = top_tokens_beam[i]
                else:
                    temp = mass_con.knapDecode(
                        logits,
                        mass,
                        self.mass_control_tol,
                    )
                    temp = ctc_post_processing(temp)
                    top_tokens[i] = temp if temp else top_tokens_beam[i]

        log_prob = F.log_softmax(output_logits, -1)
        with ThreadPoolExecutor(max_workers=min(batch_size, 16)) as executor:
            futures = [
                executor.submit(worker, log_prob[[i]], precursors[[i], 0], i)
                for i in range(batch_size)
            ]
            for future in futures:
                future.result()  # Propagate PMC exceptions to the caller.

        return [self.decoder.detokenize_truth(t, True) for t in top_tokens], batchscores

    def _forward_step(
        self,
        spectra: torch.Tensor,
        precursors: torch.Tensor,
        sequences: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return final decoder logits, tokenized targets and the logits from each layer."""
        return self.decoder(sequences, precursors, *self.encoder(spectra))

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, List[str]],
        *args,
        mode: str = "train",
    ) -> torch.Tensor:
        """
        A single training step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor, List[str]]
            A batch of (i) MS/MS spectra, (ii) precursor information, (iii)
            peptide sequences as torch Tensors.
        mode : str
            Logging key to describe the current stage.

        Returns
        -------
        torch.Tensor
            The loss of the training step.

        """
        pred, truth, output_list = self._forward_step(*batch[:3])
        assert len(output_list) == self.n_layers

        # Compute AA recall, AA precision and Pep recall in training step
        # Author: Sheng Xu
        # Date: 20230220
        tokens = torch.argmax(pred, axis=2)
        peptides_pred = []
        peptides_true = []
        for idx in range(tokens.size()[0]):
            tokens_true = truth[idx, :]
            tokens_true = self.decoder.detokenize_truth(tokens_true)
            peptides_true.append("".join(tokens_true))

            tokens_pred = tokens[idx, :]
            tokens_pred = self.decoder.detokenize(tokens_pred)
            peptides_pred.append(tokens_pred)

        if mode == "train" or self.n_beams == 0:
            self._record_matches(mode, peptides_true, peptides_pred)

        if self.custom_ctc_loss:
            raise NotImplementedError("custom_ctc_loss is not supported.")
        prediction = output_list[-1].permute(1, 0, 2)
        target_lengths = (truth != self.decoder.get_pad_idx()).sum(axis=1)
        result = masked_ctc_loss(prediction, truth, target_lengths,
                                 self.decoder.get_blank_idx())
        loss, reported, global_count, global_total = distributed_ctc(result, prediction)
        self._ctc_valid_count = global_count
        self._ctc_global_total = global_total
        self._ctc_reported_loss = reported
        self._record_ctc_batch(mode, result, batch)

        if mode == "train":
            self._ctc_window_has_data |= global_count > 0
            if global_count:
                self.log(
                "train/CELoss",
                reported,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                add_dataloader_idx=False,
                batch_size=self._ctc_valid_count,
            )

        return loss

    def validation_step(
        self, batch, batch_idx=None, dataloader_idx=None
    ) -> torch.Tensor:
        """
        A single validation step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor, List[str]]
            A batch of (i) MS/MS spectra, (ii) precursor information, (iii)
            peptide sequences.

        Returns
        -------
        torch.Tensor
            The loss of the validation step.
        """
        if dataloader_idx not in (None, 0):
            raise ValueError(
                "fit accepts only validation; evaluate test after freezing."
            )
        loss = self.training_step(batch, mode="valid")
        if self._ctc_valid_count:
            self.log(
                "valid/CELoss", self._ctc_reported_loss, on_step=False, on_epoch=True,
                sync_dist=True, add_dataloader_idx=False,
                batch_size=self._ctc_valid_count,
            )

        # Calculate and log amino acid and peptide match evaluation metrics from
        # the predicted peptides.

        if self.n_beams > 0:
            peptides_pred_raw, inferscores = self.forward(batch[0], batch[1], batch[2])
            self._record_matches("valid", batch[2], peptides_pred_raw)
        return loss

    def _record_ctc_batch(self, mode, result, batch):
        total = self._ctc_global_total
        invalid = total - self._ctc_valid_count
        counts = self._ctc_counts.setdefault(mode, {"seen": 0, "excluded": 0})
        counts["seen"] += total
        counts["excluded"] += invalid
        for index, reason in enumerate(result.reasons):
            if reason:
                identity = batch[3][index] if len(batch) > 3 else {"batch_index": index}
                logger.warning("CTC excluded stage=%s sample=%s reason=%s sequence=%s",
                               mode, identity, reason, batch[2][index])
        for name, value in (("seen", total), ("excluded", invalid),
                            ("effective", self._ctc_valid_count)):
            self.log(f"{mode}/ctc_{name}", float(value), on_step=False,
                     on_epoch=True, reduce_fx="sum", batch_size=total)
        if mode != "train":
            return
        self._ctc_empty_batches = self._ctc_empty_batches + 1 if invalid == total else 0
        if self._ctc_empty_batches >= self.ctc_max_empty_batches:
            raise RuntimeError("CTC safety stop: consecutive empty training batches")
        if (counts["seen"] >= self.ctc_guard_min_samples and
                counts["excluded"] / counts["seen"] > self.ctc_max_invalid_fraction):
            raise RuntimeError("CTC safety stop: excessive excluded training samples")

    def on_train_start(self):
        logger.info("Sample-masked CTC: global valid-count weighting, world_size=%s",
                    self.trainer.world_size)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                       optimizer_closure, **kwargs):
        def closure():
            output = optimizer_closure()
            if not self._ctc_window_has_data:
                # None gradients also prevent AdamW momentum and weight decay.
                for group in optimizer.param_groups:
                    for parameter in group["params"]:
                        parameter.grad = None
            return output
        optimizer.step(closure=closure)
        self._ctc_last_update = self._ctc_window_has_data
        self._ctc_window_has_data = False

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        if self._ctc_last_update:
            scheduler.step() if metric is None else scheduler.step(metric)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ctc_accounting"] = {
            "counts": self._ctc_counts, "empty_batches": self._ctc_empty_batches,
        }

    def on_load_checkpoint(self, checkpoint):
        state = checkpoint.get("ctc_accounting", {})
        self._ctc_counts = state.get("counts", {})
        self._ctc_empty_batches = state.get("empty_batches", 0)

    def _record_matches(self, stage, truths, predictions):
        counts = match_counts(truths, predictions, self.decoder._peptide_mass.masses)
        for name, numerator, denominator in metric_components(counts):
            metric = self.sequence_metrics[f"{stage}_{name}"]
            metric.update(numerator, denominator)
            self.log(
                f"{stage}/{name}",
                metric,
                on_step=False,
                on_epoch=True,
                add_dataloader_idx=False,
            )

    @staticmethod
    def _matches_any_prefix(name, prefixes):
        return any(
            name == prefix or name.startswith(prefix + ".") for prefix in prefixes
        )

    def _apply_freezing(self):
        if self.finetune_strategy != "partial":
            return
        for name, param in self.named_parameters():
            param.requires_grad = not self._matches_any_prefix(
                name, self.freeze_module_prefixes
            )

    def configure_optimizers(
        self,
    ) -> Tuple[torch.optim.Optimizer, Dict[str, Any]]:
        """
        Initialize the optimizer.

        This is used by pytorch-lightning when preparing the model for training.

        Returns
        -------
        Tuple[torch.optim.Optimizer, Dict[str, Any]]
            The initialized Adam optimizer and its learning rate scheduler.
        """
        self._apply_freezing()
        if self.finetune_strategy == "layerwise":
            base_lr = self.opt_kwargs.get("lr", 1e-3)
            backbone_kwargs = dict(self.opt_kwargs)
            output_kwargs = dict(self.opt_kwargs)
            backbone_kwargs["lr"] = base_lr * self.backbone_lr_multiplier
            output_kwargs["lr"] = base_lr
            output_params = []
            backbone_params = []
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                if self._matches_any_prefix(name, self.output_module_prefixes):
                    output_params.append(param)
                else:
                    backbone_params.append(param)
            param_groups = []
            if backbone_params:
                param_groups.append(dict(params=backbone_params, **backbone_kwargs))
            if output_params:
                param_groups.append(dict(params=output_params, **output_kwargs))
            optimizer = torch.optim.AdamW(param_groups)
        else:
            optimizer = torch.optim.AdamW(
                [p for p in self.parameters() if p.requires_grad], **self.opt_kwargs
            )
        # optimizer = Lion(self.parameters(), **self.opt_kwargs)
        # Apply learning rate scheduler per step.
        if self.lr_schedule == "warmup_constant":
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda step: min(1.0, (step + 1) / max(1, self.warmup_iters))
            )
        else:
            lr_scheduler = CosineWarmupScheduler(
                optimizer, warmup=self.warmup_iters, max_iters=self.max_iters
            )
        return [optimizer], {"scheduler": lr_scheduler, "interval": "step"}

    def on_after_backward(self) -> None:
        check_finite_gradients(self.parameters())


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with linear warm up followed by cosine shaped decay.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer object.
    warmup : int
        The number of warm up iterations.
    max_iters : int
        The total number of iterations.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, warmup: int, max_iters: int):
        self.warmup, self.max_iters = warmup, max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        # Linear warmup followed by cosine decay.
        # Author: Sheng Xu
        # Date: 20230214

        decay = self.warmup / self.max_iters
        if epoch <= self.warmup and self.warmup > 0:
            # lr_factor = 1

            lr_factor = 1 * (epoch / self.warmup)
        else:
            lr_factor = 0.5 * (
                1
                + np.cos(
                    np.pi
                    * (
                        (epoch - (decay * self.max_iters))
                        / ((1 - decay) * self.max_iters)
                    )
                )
            )
            # lr_factor = 0.05

        return lr_factor
