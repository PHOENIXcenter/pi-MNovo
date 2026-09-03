import pytest

from MNovo.denovo.ctc_decoder_base import CTCDecoderBase


class _Decoder:
    @staticmethod
    def get_blank_idx() -> int:
        return 0


def test_base_decoder_raises_clear_not_implemented_error() -> None:
    decoder = CTCDecoderBase(_Decoder())

    with pytest.raises(NotImplementedError, match="implement decode"):
        decoder.decode(None, None)


def test_ctc_post_processing_collapses_repeats_and_blanks() -> None:
    decoder = CTCDecoderBase(_Decoder())

    assert decoder.ctc_post_processing([1, 1, 0, 0, 2, 2]) == [1, 2]
