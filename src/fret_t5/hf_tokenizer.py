"""Utilities for wrapping SynthTab tokenizers as Hugging Face tokenizers."""

from __future__ import annotations

from typing import Dict

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

__all__ = ["build_hf_tokenizer"]


def build_hf_tokenizer(
    vocab: Dict[str, int],
    pad: str = "<pad>",
    eos: str = "<eos>",
    unk: str = "<unk>",
) -> PreTrainedTokenizerFast:
    """Create a :class:`PreTrainedTokenizerFast` backed by the SynthTab vocab."""

    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token=unk))
    hf = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token=pad,
        eos_token=eos,
        unk_token=unk,
        bos_token=None,
    )
    hf.pad_token_id = vocab[pad]
    hf.eos_token_id = vocab[eos]
    hf.unk_token_id = vocab[unk]
    return hf
