"""Fused, cache-free inference runtime for the MNovo release model."""

from __future__ import annotations

import math
import os
import site
import ctypes
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader


def _configure_cuda_runtime() -> None:
    candidates = []
    for package_root in site.getsitepackages():
        candidates.append(Path(package_root) / "nvidia" / "cuda_nvrtc" / "lib")
    existing = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    additions = [str(path) for path in candidates if path.is_dir()]
    merged = additions + [path for path in existing if path]
    if merged:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(dict.fromkeys(merged))
    for directory in candidates:
        library = directory / "libnvrtc.so.12"
        if library.is_file():
            ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
            break


_configure_cuda_runtime()

from MNovo.denovo.ctc_beam_search import CTCBeamSearchDecoder
from MNovo.denovo.data import prepare_batch
from MNovo.denovo.spectrum_dataset import SpectrumDataset
from MNovo.denovo.spectrum_index import LmdbSpectrumIndex
from MNovo.denovo.model import Spec2Pep
from MNovo.ranker.features import (
    calibrated_confidence,
    candidate_core_features,
    isotope_aware_mass_features,
)
from MNovo.ranker.fragment import fragment_features
from MNovo.ranker.length import SpectrumLengthPredictor, length_candidate_features
from MNovo.ranker.model import ResidualCandidateRanker
from MNovo.ranker.observable_router import (
    ObservablePathRouter,
    apply_frozen_threshold,
)
from MNovo.candidate_generation import (
    ctc_nll_for_batch,
    merge_pmc_candidate,
    model_parameters,
    pmc_candidates,
)
from MNovo.router_schema import ROUTER_FEATURE_NAMES


@dataclass(frozen=True)
class RuntimeOptions:
    mode: str = "fast"
    batch_size: int = 96
    n_workers: int = 16
    ctc_processes: int = 16
    initial_beam: int = 5
    cutoff_top_n: int = 30
    precision: str = "bf16"
    length_alpha: float = 0.7
    fragment_peak_count: int = 120


@dataclass
class Prediction:
    dataset_index: int
    peptide: str
    confidence: float
    route: str


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: SpectrumDataset, indices: np.ndarray) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, index: int):
        dataset_index = int(self.indices[index])
        return (*self.dataset[dataset_index], dataset_index)


def collate_indexed(batch):
    spectra, precursors, peptides = prepare_batch([row[:4] for row in batch])
    indices = torch.tensor([row[4] for row in batch], dtype=torch.long)
    return spectra, precursors, peptides, indices


def _selected(
    scores: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    masked = scores.masked_fill(~mask, -1e9)
    top = masked.topk(k=min(2, masked.size(1)), dim=1)
    index = top.indices[:, 0]
    margin = (
        top.values[:, 0] - top.values[:, 1]
        if top.values.size(1) > 1
        else torch.full_like(top.values[:, 0], float("inf"))
    )
    return index, margin


def _gather(matrix: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return matrix.gather(1, index.unsqueeze(1)).squeeze(1)


class MNovoRuntime:
    """Load every frozen component once and run fused in-memory inference."""

    def __init__(
        self,
        model_dir: str | Path,
        options: RuntimeOptions | None = None,
        device: str = "auto",
    ) -> None:
        self.root = Path(model_dir)
        self.components = self.root / "components"
        self.options = options or RuntimeOptions()
        if self.options.mode != "fast":
            raise ValueError("The release runtime supports only mode='fast'.")
        if self.options.initial_beam != 5:
            raise ValueError("The release runtime is frozen to beam width 5.")
        self.device = torch.device(
            "cuda"
            if device == "auto" and torch.cuda.is_available()
            else ("cpu" if device == "auto" else device)
        )
        self.config = yaml.safe_load(
            (self.root / "config" / "inference.yaml").read_text(encoding="utf-8")
        )
        args = SimpleNamespace(
            beam_size=self.options.initial_beam,
            cutoff_top_n=self.options.cutoff_top_n,
            ctc_processes=self.options.ctc_processes,
        )
        self.backbone = Spec2Pep.load_from_checkpoint(
            self.components / "backbone.ckpt",
            map_location="cpu",
            **model_parameters(self.config, args),
        ).to(self.device)
        self.backbone.eval()
        self.backbone.requires_grad_(False)
        self.decoder5 = CTCBeamSearchDecoder(
            self.backbone.decoder,
            {
                "beam_width": self.options.initial_beam,
                "cutoff_top_n": self.options.cutoff_top_n,
                "cutoff_prob": 1.0,
                "num_processes": self.options.ctc_processes,
            },
        )
        self.r1, self.r1_checkpoint = self._load_ranker("r1_ranker.pt")
        self.r2, self.r2_checkpoint = self._load_ranker("r2_long_expert.pt")
        self.r3, self.r3_checkpoint = self._load_ranker("r3_fragment_expert.pt")
        self.length_predictor, self.length_checkpoint = self._load_length()
        self.fast_router, self.fast_router_checkpoint = self._load_fast_router()
        if self.fast_router_checkpoint.get("feature_names") != ROUTER_FEATURE_NAMES:
            raise RuntimeError(
                "This model uses an obsolete router schema. The release "
                "runtime accepts only the frozen 36-feature observable router."
            )

    def _checkpoint(self, name: str) -> dict:
        return torch.load(
            self.components / name,
            map_location="cpu",
            weights_only=True,
        )

    def _load_ranker(self, name: str):
        checkpoint = self._checkpoint(name)
        model = ResidualCandidateRanker(**checkpoint["model_args"]).to(self.device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        model.requires_grad_(False)
        return model, checkpoint

    def _load_length(self):
        checkpoint = self._checkpoint("length_predictor.pt")
        model = SpectrumLengthPredictor(**checkpoint["model_args"]).to(self.device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        model.requires_grad_(False)
        return model, checkpoint

    def _load_fast_router(self):
        name = (
            "observable_router.pt"
            if (self.components / "observable_router.pt").is_file()
            else "fast_router.pt"
        )
        checkpoint = self._checkpoint(name)
        model = ObservablePathRouter(**checkpoint["model_args"]).to(self.device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        model.requires_grad_(False)
        return model, checkpoint

    def _clean_tokens(self, row: Iterable[int]) -> list[int]:
        pad = int(self.backbone.decoder.get_pad_idx())
        blank = int(self.backbone.decoder.get_blank_idx())
        result = []
        for value in row:
            token = int(value)
            if token in {pad, blank}:
                continue
            result.append(token)
        return result

    def _raw_symbols(self, tokens: list[int]) -> list[str]:
        symbols = [str(self.backbone.decoder._idx2aa[int(token)]) for token in tokens]
        if self.backbone.decoder.reverse:
            symbols = list(reversed(symbols))
        return symbols

    def _symbols(self, tokens: list[int]) -> list[str]:
        symbols = self._raw_symbols(tokens)
        if symbols and symbols[0] == "$":
            symbols = symbols[1:]
        if not symbols or "$" in symbols:
            return []
        return symbols

    def _peptide_text(self, tokens: list[int]) -> str:
        return "".join(self._raw_symbols(tokens))

    def _token_mass(self, token: int) -> float:
        symbol = str(self.backbone.decoder._idx2aa[int(token)])
        return float(self.config["residues"].get(symbol, 0.0))

    def _candidate_mass(self, tokens: list[int]) -> float:
        return sum(self._token_mass(token) for token in tokens)

    def _pmc_candidates(
        self,
        log_probs: torch.Tensor,
        precursors: torch.Tensor,
        top_tokens: list[list[int]],
    ) -> tuple[list[list[int]], list[float]]:
        candidates = pmc_candidates(
            self.backbone,
            log_probs,
            precursors,
            top_tokens,
            True,
        )
        nll = ctc_nll_for_batch(self.backbone, log_probs, candidates)
        return candidates, nll

    def _rows_from_decode(
        self,
        beam_tokens: torch.Tensor,
        beam_nlls: torch.Tensor,
        log_probs: torch.Tensor,
        precursors: torch.Tensor,
        pmc_bundle: tuple[list[list[int]], list[float]] | None = None,
        return_pmc: bool = False,
    ) -> (
        list[dict]
        | tuple[
            list[dict],
            tuple[list[list[int]], list[float]],
        ]
    ):
        beam_tokens = beam_tokens.detach().cpu()
        beam_nlls = beam_nlls.detach().float().cpu()
        decoded = [
            [self._clean_tokens(tokens.tolist()) for tokens in rows]
            for rows in beam_tokens
        ]
        if pmc_bundle is None:
            pmc, pmc_nll = self._pmc_candidates(
                log_probs,
                precursors,
                [rows[0] for rows in decoded],
            )
        else:
            pmc, pmc_nll = pmc_bundle
            if len(pmc) != len(decoded) or len(pmc_nll) != len(decoded):
                raise RuntimeError("Cached PMC rows do not match decoded rows.")
        output = []
        for row_index, candidates in enumerate(decoded):
            nlls = [float(value) for value in beam_nlls[row_index].tolist()]
            candidates, nlls, sources, _injected = merge_pmc_candidate(
                candidates,
                nlls,
                pmc[row_index],
                pmc_nll[row_index],
                len(candidates),
                "union",
            )
            unique_tokens, unique_nll, unique_sources = [], [], []
            seen = set()
            for tokens, nll, source in zip(candidates, nlls, sources):
                key = tuple(tokens)
                if not tokens or key in seen or not math.isfinite(nll):
                    continue
                seen.add(key)
                unique_tokens.append(tokens)
                unique_nll.append(float(nll))
                unique_sources.append(source)
            texts = [self._peptide_text(tokens) for tokens in unique_tokens]
            lengths = [len(self._symbols(tokens)) for tokens in unique_tokens]
            output.append(
                {
                    "tokens": unique_tokens,
                    "texts": texts,
                    "nll": unique_nll,
                    "sources": unique_sources,
                    "length": lengths,
                    "mass": [self._candidate_mass(tokens) for tokens in unique_tokens],
                }
            )
        if return_pmc:
            return output, (pmc, pmc_nll)
        return output

    def _tensorize(
        self,
        rows: list[dict],
        precursors: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> dict:
        batch = len(rows)
        width = max(max(len(row["texts"]), 1) for row in rows)
        core = torch.zeros((batch, width, 3), dtype=torch.float32)
        mask = torch.zeros((batch, width), dtype=torch.bool)
        length = torch.ones((batch, width), dtype=torch.float32)
        mass = torch.zeros((batch, width), dtype=torch.float32)
        is_pmc = torch.zeros((batch, width), dtype=torch.bool)
        for row_index, row in enumerate(rows):
            count = len(row["texts"])
            if count == 0:
                continue
            core[row_index, :count] = candidate_core_features(
                row["nll"],
                row["length"],
                self.options.length_alpha,
            )
            mask[row_index, :count] = True
            length[row_index, :count] = torch.tensor(row["length"])
            mass[row_index, :count] = torch.tensor(row["mass"])
            is_pmc[row_index, :count] = torch.tensor(
                [source == "pmc_union" for source in row["sources"]]
            )
        data = {
            "core": core.to(self.device),
            "mask": mask.to(self.device),
            "length": length.to(self.device),
            "mass": mass.to(self.device),
            "is_pmc": is_pmc.to(self.device),
            "precursor": precursors.to(self.device),
            "embedding": embeddings.to(self.device),
            "rows": rows,
        }
        data["mass0"] = isotope_aware_mass_features(
            data["mass"],
            data["precursor"][:, 0],
            (0,),
        )[..., :1]
        return data

    def _rank(
        self,
        model: ResidualCandidateRanker,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores, _delta = model(features)
        index, margin = _selected(scores, mask)
        return scores, index, margin

    def _r1_confidence(self, score: torch.Tensor) -> torch.Tensor:
        calibration = self.r1_checkpoint["ranker_calibrator"]
        return calibrated_confidence(
            score,
            float(calibration["scale"]),
            float(calibration["bias"]),
        )

    def _observable_select(
        self,
        data: dict,
        spectra: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        """Reproduce the frozen 36-feature Benchmark10 selection online."""
        r1_input = torch.cat([data["core"], data["mass0"]], dim=-1)
        r1_scores, r1_index, r1_margin = self._rank(self.r1, r1_input, data["mask"])
        length_logits = self.length_predictor(data["embedding"], data["precursor"])
        length_features = length_candidate_features(length_logits, data["length"])
        long_extra = torch.stack(
            [
                data["length"] / 40.0,
                (data["length"] - data["length"][:, :1]) / 10.0,
            ],
            dim=-1,
        )
        r2_input = torch.cat(
            [
                r1_scores.unsqueeze(-1),
                data["core"],
                data["mass0"],
                length_features,
                long_extra,
            ],
            dim=-1,
        )
        r2_scores, r2_index, r2_margin = self._rank(self.r2, r2_input, data["mask"])
        fragment = self._fragment_tensor(data, spectra)
        r3_input = torch.cat(
            [r1_scores.unsqueeze(-1), data["core"], data["mass0"], fragment],
            dim=-1,
        )
        r3_scores, r3_index, r3_margin = self._rank(self.r3, r3_input, data["mask"])

        indices = [r1_index, r2_index, r3_index]
        scores = [r1_scores, r2_scores, r3_scores]
        margins = [r1_margin, r2_margin, r3_margin]
        selected_scores = [
            _gather(value, index) for value, index in zip(scores, indices)
        ]
        fragment_composite = (
            fragment[..., [0, 1, 2, 5, 6, 7, 8, 9, 11]]
            * torch.tensor(
                [0.12, 0.18, 0.14, 0.14, 0.08, 0.08, 0.10, 0.10, 0.06],
                device=self.device,
            )
        ).sum(dim=-1)
        selected_mass = [_gather(data["mass0"][..., 0], index) for index in indices]
        selected_length = [_gather(data["length"] / 40.0, index) for index in indices]
        selected_length_logp = [
            _gather(length_features[..., 0], index) for index in indices
        ]
        selected_length_dev = [
            _gather(length_features[..., 1], index) for index in indices
        ]
        selected_fragment = [_gather(fragment_composite, index) for index in indices]
        selected_pmc = [_gather(data["is_pmc"].float(), index) for index in indices]
        features = torch.stack(
            [
                selected_scores[0],
                selected_scores[1],
                selected_scores[2],
                selected_scores[1] - selected_scores[0],
                selected_scores[2] - selected_scores[0],
                margins[0],
                margins[1],
                margins[2],
                indices[0].float() / 20.0,
                indices[1].float() / 20.0,
                indices[2].float() / 20.0,
                (indices[0] == indices[1]).float(),
                (indices[0] == indices[2]).float(),
                (indices[1] == indices[2]).float(),
                torch.log1p(data["precursor"][:, 0].clamp_min(0.0)) / 10.0,
                data["precursor"][:, 1] / 10.0,
                selected_mass[0],
                selected_mass[1],
                selected_mass[2],
                selected_length[0],
                selected_length[1],
                selected_length[2],
                selected_length_logp[0],
                selected_length_logp[1],
                selected_length_logp[2],
                selected_length_dev[0],
                selected_length_dev[1],
                selected_length_dev[2],
                selected_fragment[0],
                selected_fragment[1],
                selected_fragment[2],
                selected_fragment[1] - selected_fragment[0],
                selected_fragment[2] - selected_fragment[0],
                selected_pmc[0],
                selected_pmc[1],
                selected_pmc[2],
            ],
            dim=1,
        ).float()
        if self.fast_router_checkpoint.get("feature_names") != ROUTER_FEATURE_NAMES:
            raise RuntimeError("Observable-router feature schema changed.")
        normalized = (
            features - self.fast_router_checkpoint["feature_mean"].to(self.device)
        ) / self.fast_router_checkpoint["feature_std"].to(self.device)
        logits = self.fast_router(normalized)
        route, _probability, _advantage = apply_frozen_threshold(
            logits, float(self.fast_router_checkpoint["frozen_threshold"])
        )
        path_index = torch.stack(indices, dim=1)
        selected = path_index[torch.arange(route.numel(), device=self.device), route]
        route_names = ["r1", "r2_long", "r3_fragment"]
        names = [route_names[int(value)] for value in route.cpu().tolist()]
        # R2/R3 have no independent calibrator. Score every routed peptide on
        # the common R1 scale so exported confidence remains comparable.
        confidence = self._r1_confidence(_gather(r1_scores, selected))
        return selected, confidence, names

    def _fragment_tensor(
        self,
        data: dict,
        spectra: torch.Tensor,
        selected_rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, width = data["mask"].shape
        features = torch.zeros((batch, width, 12), dtype=torch.float32)
        selected = (
            set(range(batch))
            if selected_rows is None
            else set(int(value) for value in selected_rows.cpu().tolist())
        )
        spectra_np = spectra.detach().cpu().numpy()
        charges = data["precursor"][:, 1].detach().cpu().long().tolist()
        for row_index in selected:
            spectrum = spectra_np[row_index]
            for candidate_index, peptide in enumerate(data["rows"][row_index]["texts"]):
                features[row_index, candidate_index] = torch.from_numpy(
                    fragment_features(
                        peptide,
                        spectrum[:, 0],
                        spectrum[:, 1],
                        int(charges[row_index]),
                        self.config["residues"],
                        0.5,
                        2,
                    )
                )
        return features.to(self.device)

    def predict_batch(
        self,
        spectra: torch.Tensor,
        precursors: torch.Tensor,
        dataset_indices: torch.Tensor,
    ) -> list[Prediction]:
        spectra_device = spectra.to(self.device, non_blocking=True)
        precursors_device = precursors.to(self.device, non_blocking=True)
        autocast = self.options.precision == "bf16" and self.device.type == "cuda"
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=autocast,
            ),
        ):
            memory, memory_mask = self.backbone.encoder(spectra_device)
            logits, _, _ = self.backbone.decoder(
                None,
                precursors_device,
                memory,
                memory_mask,
            )
            probabilities = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            beam_tokens, beam_nll = self.decoder5.decode_all(probabilities)
            rows = self._rows_from_decode(
                beam_tokens,
                beam_nll,
                log_probs,
                precursors_device,
            )
            data = self._tensorize(rows, precursors_device, memory[:, 0])
            selected, confidence, routes = self._observable_select(data, spectra)

        output = []
        for row_index, candidate_index in enumerate(selected.cpu().tolist()):
            texts = rows[row_index]["texts"]
            peptide = texts[candidate_index] if texts else ""
            output.append(
                Prediction(
                    dataset_index=int(dataset_indices[row_index]),
                    peptide=peptide,
                    confidence=float(confidence[row_index]),
                    route=routes[row_index],
                )
            )
        return output

    def predict_lmdb(
        self,
        lmdb: str | Path,
        indices: np.ndarray | None = None,
    ):
        valid_charge = np.arange(1, int(self.config["max_charge"]) + 1)
        spectrum_index = LmdbSpectrumIndex(
            str(lmdb), None, 2, valid_charge, True, lock=False
        )
        dataset = SpectrumDataset(
            [spectrum_index],
            n_peaks=int(self.config["n_peaks"]),
            min_mz=float(self.config["min_mz"]),
            max_mz=float(self.config["max_mz"]),
            min_intensity=float(self.config["min_intensity"]),
            remove_precursor_tol=float(self.config["remove_precursor_tol"]),
            random_state=3407,
        )
        if indices is None:
            indices = np.arange(len(dataset), dtype=np.int64)
        loader = DataLoader(
            IndexedDataset(dataset, np.asarray(indices, dtype=np.int64)),
            batch_size=self.options.batch_size,
            shuffle=False,
            num_workers=self.options.n_workers,
            pin_memory=True,
            persistent_workers=self.options.n_workers > 0,
            prefetch_factor=4 if self.options.n_workers > 0 else None,
            collate_fn=collate_indexed,
        )
        for spectra, precursors, _truth, dataset_indices in loader:
            yield from self.predict_batch(spectra, precursors, dataset_indices)
