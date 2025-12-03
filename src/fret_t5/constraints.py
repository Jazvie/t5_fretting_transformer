"""Adapters that expose SynthTab constraints as Hugging Face processors."""

from __future__ import annotations

from typing import Callable

import torch
from transformers import LogitsProcessor

from .constrained_generation import V3ConstrainedProcessor
from .tokenization import MidiTabTokenizerV3

__all__ = ["TabConstraintProcessor", "build_v3_constraint_processor"]

ConstraintFn = Callable[[torch.LongTensor], torch.Tensor]


class TabConstraintProcessor(LogitsProcessor):
    """Apply boolean masks produced by a constraint function to logits."""

    def __init__(self, constraint_fn: ConstraintFn) -> None:
        self.constraint_fn = constraint_fn

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        mask = self.constraint_fn(input_ids)
        if mask.dim() == 1:
            scores[:, ~mask] = float("-inf")
        else:
            for batch_idx in range(scores.size(0)):
                current_mask = mask[batch_idx]
                scores[batch_idx, ~current_mask] = float("-inf")
        return scores


def build_v3_constraint_processor(tokenizer: MidiTabTokenizerV3) -> TabConstraintProcessor:
    """Wrap the v3 constraint logic so it can be used with ``model.generate``."""

    processor = V3ConstrainedProcessor(tokenizer)
    vocab_size = len(tokenizer.shared_token_to_id)

    def constraint_fn(input_ids: torch.LongTensor) -> torch.BoolTensor:
        batch_size, _ = input_ids.shape
        mask = torch.zeros((batch_size, vocab_size), dtype=torch.bool, device=input_ids.device)
        chord_states = processor._track_chord_state(input_ids)
        for batch_idx in range(batch_size):
            sequence = input_ids[batch_idx]
            non_pad = sequence != processor.pad_id
            if torch.any(non_pad):
                last_index = torch.nonzero(non_pad, as_tuple=False)[-1].item()
                last_token = int(sequence[last_index].item())
            else:
                last_token = processor.pad_id
            allowed = processor._get_allowed_tokens(last_token, chord_states[batch_idx])
            if allowed:
                mask[batch_idx, allowed] = True
            else:
                mask[batch_idx, :] = True
        return mask

    return TabConstraintProcessor(constraint_fn)
