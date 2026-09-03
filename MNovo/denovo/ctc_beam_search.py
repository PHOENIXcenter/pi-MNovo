# Copyright (c) Puyuan Liu
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from ctcdecode import CTCBeamDecoder
from torch import TensorType
from .ctc_decoder_base import CTCDecoderBase
from typing import Dict, List

LOGGER = logging.getLogger(__name__)


class CTCBeamSearchDecoder(CTCDecoderBase):
    """
    CTC Beam Search Decoder
    """

    def __init__(self, decoder, decoder_parameters: Dict) -> None:
        super().__init__(decoder)
        self.attn_decoder = decoder
        beam_width = int(
            decoder_parameters.get("beam_width", decoder_parameters.get("beam", 10))
        )
        cutoff_top_n = int(decoder_parameters.get("cutoff_top_n", 30))
        cutoff_prob = float(decoder_parameters.get("cutoff_prob", 1.0))
        num_processes = int(decoder_parameters.get("num_processes", 4))
        self.decoder = CTCBeamDecoder(
            decoder.get_symbols(),
            model_path=None,
            alpha=0,
            beta=0,
            cutoff_top_n=cutoff_top_n,
            cutoff_prob=cutoff_prob,
            beam_width=beam_width,
            num_processes=num_processes,
            blank_id=decoder.get_blank_idx(),
            log_probs_input=False,
        )
        LOGGER.info(
            "ctcdecode beam_width=%d cutoff_top_n=%d num_processes=%d",
            self.decoder._beam_width,
            cutoff_top_n,
            num_processes,
        )

    def _decode_raw(self, probabilities: TensorType):
        beam_results, beam_scores, timesteps, out_lens = self.decoder.decode(
            probabilities
        )
        max_len = beam_results.size(-1)
        mask = (
            torch.arange(0, max_len)
            .type_as(out_lens)
            .view(1, 1, -1)
            .lt(out_lens.unsqueeze(-1))
        )
        beam_results = beam_results.clone()
        beam_results[~mask] = self.attn_decoder.get_pad_idx()
        return beam_results, beam_scores, timesteps, out_lens

    def decode_all(self, probabilities: TensorType):
        """Return every beam and its CTC negative log score."""
        beam_results, beam_scores, _timesteps, _out_lens = self._decode_raw(
            probabilities
        )
        return beam_results, beam_scores

    def decode(self, log_prob: TensorType, **kwargs) -> List[List[int]]:
        """
        Decoding function for the CTC beam search decoder.
        """

        """
        if log_prob.dtype != torch.float16:
            log_prob = log_prob.cpu()
        """
        beam_results, beam_scores, _timesteps, _out_lens = self._decode_raw(log_prob)
        return beam_results[:, 0, :], beam_scores[:, 0]
