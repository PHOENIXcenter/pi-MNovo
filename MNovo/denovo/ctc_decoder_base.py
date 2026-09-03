# Copyright (c) Puyuan Liu
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Tuple

import torch
from torch import TensorType


class CTCDecoderBase:
    """Base class for CTC decoders."""

    def __init__(self, decoder):
        self.attn_decoder = decoder

    @staticmethod
    def unravel_indices(
        indices: torch.LongTensor, shape: Tuple[int, ...]
    ) -> torch.LongTensor:
        """Unravel flat tensor indexes for the provided shape."""
        coord = []
        for dim in reversed(shape):
            coord.append(indices % dim)
            indices = torch.div(indices, dim, rounding_mode="floor")

        coord = torch.stack(coord[::-1], dim=-1)

        return coord

    def decode(
        self, log_prob: TensorType, source_length: TensorType
    ) -> List[List[int]]:
        """Decode log probabilities into token indexes."""
        raise NotImplementedError("Subclasses must implement decode().")

    def ctc_post_processing(self, sentence_index: List[int]) -> List[int]:
        """
        Merge repetitive tokens, then eliminate <blank> tokens and <pad> tokens.
        The input sentence_index is expected to be a 1-D index list
        """
        sentence_index = self.collapse_repeated_tokens(sentence_index)
        sentence_index = list(
            filter((self.attn_decoder.get_blank_idx()).__ne__, sentence_index)
        )

        return sentence_index

    @staticmethod
    def collapse_repeated_tokens(index_list: List[int]) -> List[int]:
        """Collapse consecutive duplicate token indexes."""
        return [
            a
            for a, b in zip(index_list, index_list[1:] + [not index_list[-1]])
            if a != b
        ]
