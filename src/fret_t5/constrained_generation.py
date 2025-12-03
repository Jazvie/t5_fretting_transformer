"""Constrained generation utilities for Fretting-Transformer v3.

This module provides LogitsProcessor classes to enforce the v3 encoding
pattern during generation: TAB<s,f> → TIME_SHIFT<d> → TAB<s,f> → ...

The constraints ensure musical validity and prevent degenerate sequences
while maintaining the paper's intended token alternation.
"""

from typing import Set, Dict, List
import torch
from transformers import LogitsProcessor

from .tokenization import MidiTabTokenizerV3

__all__ = [
    "V3ConstrainedProcessor",
    "create_v3_processor",
]


class V3ConstrainedProcessor(LogitsProcessor):
    """Enforces v3 TAB ↔ TIME_SHIFT alternation with chord awareness.

    Rules:
    - After TAB<s,f>: must emit TIME_SHIFT<d> (0 or non-zero, no EOS)
    - After TIME_SHIFT<0>: must emit TAB<s,f> (continue chord)
    - After TIME_SHIFT>0: can emit TAB<s,f> (new onset) or EOS
    - Within chord: no duplicate strings, max 6 strings total

    This preserves the v3 pairing structure and properly handles chords.
    """

    def __init__(self, tokenizer: MidiTabTokenizerV3, max_chord_size: int = 6):
        self.tokenizer = tokenizer
        self.max_chord_size = max_chord_size

        # Precompute token sets for efficiency
        self.tab_ids: Set[int] = set(tokenizer.get_tab_token_ids())
        self.time_shift_ids: Set[int] = set(tokenizer.get_time_shift_token_ids())
        self.eos_id: int = tokenizer.shared_token_to_id["<eos>"]
        self.pad_id: int = tokenizer.shared_token_to_id["<pad>"]

        # Special handling for zero time shift
        self.zero_time_shift_id: int = tokenizer.shared_token_to_id.get("TIME_SHIFT<0>", -1)
        if self.zero_time_shift_id == -1:
            print("Warning: No TIME_SHIFT<0> token found. Chords may not be representable.")

        # Create string-to-tab mapping for chord validation
        self.string_to_tab_ids: Dict[int, Set[int]] = {}
        for token, token_id in tokenizer.shared_token_to_id.items():
            if token.startswith("TAB<"):
                # Extract string number from TAB<string,fret>
                parts = token[4:-1].split(",")
                if len(parts) == 2:
                    string_num = int(parts[0])
                    if string_num not in self.string_to_tab_ids:
                        self.string_to_tab_ids[string_num] = set()
                    self.string_to_tab_ids[string_num].add(token_id)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Apply v3 constraints to generation logits.

        Args:
            input_ids: Shape [batch_size, seq_len]
            scores: Shape [batch_size, vocab_size]

        Returns:
            Constrained scores with invalid tokens set to -inf
        """
        batch_size = input_ids.shape[0]

        # Find last non-pad token for each sequence
        last_tokens = []
        for b in range(batch_size):
            non_pad_mask = input_ids[b] != self.pad_id
            if non_pad_mask.any():
                last_pos = non_pad_mask.nonzero()[-1].item()
                last_tokens.append(input_ids[b, last_pos].item())
            else:
                last_tokens.append(self.pad_id)  # Empty sequence

        # Track chord state for each sequence
        chord_states = self._track_chord_state(input_ids)

        # Apply constraints per batch item
        constrained_scores = scores.clone()
        constrained_scores.fill_(float("-inf"))  # Mask everything initially

        for b, last_token in enumerate(last_tokens):
            allowed_ids = self._get_allowed_tokens(last_token, chord_states[b])
            constrained_scores[b, allowed_ids] = scores[b, allowed_ids]

        return constrained_scores

    def _get_allowed_tokens(self, last_token_id: int, chord_state: Dict = None) -> list[int]:
        """Get allowed next tokens based on v3 constraints and chord state."""
        if chord_state is None:
            chord_state = {"strings_used": set(), "in_chord": False}

        if last_token_id in self.tab_ids:
            # After TAB: must emit TIME_SHIFT (0 or non-zero, no EOS)
            allowed = list(self.time_shift_ids)

            # If we've reached max chord size, force non-zero TIME_SHIFT
            if len(chord_state["strings_used"]) >= self.max_chord_size:
                if self.zero_time_shift_id in allowed:
                    allowed.remove(self.zero_time_shift_id)

            return allowed

        elif last_token_id == self.zero_time_shift_id:
            # After TIME_SHIFT<0>: must emit TAB (continue chord, no EOS)
            allowed_tabs = list(self.tab_ids)

            # Remove TABs for strings already used in this chord
            if chord_state["strings_used"]:
                filtered_tabs = []
                for tab_id in allowed_tabs:
                    token = self.tokenizer.shared_id_to_token.get(tab_id, "")
                    if token.startswith("TAB<"):
                        parts = token[4:-1].split(",")
                        if len(parts) == 2:
                            string_num = int(parts[0])
                            if string_num not in chord_state["strings_used"]:
                                filtered_tabs.append(tab_id)
                allowed_tabs = filtered_tabs

            return allowed_tabs

        elif last_token_id in self.time_shift_ids:
            # After TIME_SHIFT>0: can emit TAB (new onset) or EOS
            return list(self.tab_ids) + [self.eos_id]
        else:
            # Start of sequence or special token: allow TAB only
            return list(self.tab_ids)

    def _extract_string_from_tab_token(self, token_id: int) -> int:
        """Extract string number from TAB token ID."""
        token = self.tokenizer.shared_id_to_token.get(token_id, "")
        if token.startswith("TAB<"):
            parts = token[4:-1].split(",")
            if len(parts) == 2:
                return int(parts[0])
        return -1

    def _track_chord_state(self, input_ids: torch.Tensor) -> List[Dict]:
        """Track chord state for each sequence in the batch."""
        batch_size = input_ids.shape[0]
        chord_states = []

        for b in range(batch_size):
            strings_used = set()
            in_chord = False

            # Scan backwards to find current chord state
            for i in range(input_ids.shape[1] - 1, -1, -1):
                token_id = input_ids[b, i].item()

                if token_id == self.pad_id:
                    continue

                # If we hit a non-zero TIME_SHIFT, we're not in a chord
                if token_id in self.time_shift_ids and token_id != self.zero_time_shift_id:
                    break

                # If we hit TIME_SHIFT<0>, we're in a chord
                if token_id == self.zero_time_shift_id:
                    in_chord = True

                # If we hit a TAB token, add its string to the set
                if token_id in self.tab_ids:
                    string_num = self._extract_string_from_tab_token(token_id)
                    if string_num != -1:
                        strings_used.add(string_num)

            chord_states.append({
                "strings_used": strings_used,
                "in_chord": in_chord
            })

        return chord_states


def create_v3_processor(tokenizer: MidiTabTokenizerV3) -> V3ConstrainedProcessor:
    """Factory function to create a v3 constrained processor.

    Args:
        tokenizer: Tokenizer with v3 encoding support

    Returns:
        Configured LogitsProcessor for v3 constraints
    """
    return V3ConstrainedProcessor(tokenizer)


def validate_v3_sequence(tokens: list[str]) -> tuple[bool, str]:
    """Validate that a token sequence follows v3 constraints.

    Args:
        tokens: List of token strings to validate

    Returns:
        (is_valid, error_message)
    """
    if not tokens:
        return True, ""

    # Check alternating pattern
    expect_tab = True
    for i, token in enumerate(tokens):
        if token in ["<pad>", "<eos>", "<unk>"]:
            continue

        is_tab = token.startswith("TAB<")
        is_time_shift = token.startswith("TIME_SHIFT<")

        if not (is_tab or is_time_shift):
            continue  # Skip special tokens

        if expect_tab and not is_tab:
            return False, f"Expected TAB at position {i}, got {token}"
        elif not expect_tab and not is_time_shift:
            return False, f"Expected TIME_SHIFT at position {i}, got {token}"

        expect_tab = not expect_tab

    # Check that sequence ends properly (after TIME_SHIFT)
    last_musical_token = None
    for token in reversed(tokens):
        if token.startswith("TAB<") or token.startswith("TIME_SHIFT<"):
            last_musical_token = token
            break

    if last_musical_token and last_musical_token.startswith("TAB<"):
        return False, "Sequence ends with TAB instead of TIME_SHIFT"

    return True, ""