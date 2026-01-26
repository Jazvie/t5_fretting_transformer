"""Constrained generation utilities for Fretting-Transformer v3.

This module provides LogitsProcessor classes to enforce the v3 encoding
pattern during generation: TAB<s,f> → TIME_SHIFT<d> → TAB<s,f> → ...

The constraints ensure musical validity and prevent degenerate sequences
while maintaining the paper's intended token alternation.

This module also provides HuggingFace-compatible adapters for using
these constraints with model.generate().
"""

from __future__ import annotations

from typing import Set, Dict, List, Callable
import torch
from transformers import LogitsProcessor

from .tokenization import MidiTabTokenizerV3

__all__ = [
    "V3ConstrainedProcessor",
    "ForcedTokenLogitsProcessor",
    "create_v3_processor",
    "TabConstraintProcessor",
    "build_v3_constraint_processor",
]

# Type alias for constraint functions
ConstraintFn = Callable[[torch.LongTensor], torch.Tensor]


class V3ConstrainedProcessor(LogitsProcessor):
    """Enforces v3 TAB ↔ TIME_SHIFT alternation with chord awareness.

    Rules:
    - After TAB<s,f>: must emit TIME_SHIFT<d> (0 or non-zero, no EOS)
    - After TIME_SHIFT<0>: must emit TAB<s,f> (continue chord)
    - After TIME_SHIFT>0: can emit TAB<s,f> (new onset) or EOS
    - Within chord: no duplicate strings, max 6 strings total
    - Within chord: fret span (excluding open strings) must not exceed max_fret_span

    This preserves the v3 pairing structure and properly handles chords.
    """

    def __init__(self, tokenizer: MidiTabTokenizerV3, max_chord_size: int = 6, max_fret_span: int = 5):
        self.tokenizer = tokenizer
        self.max_chord_size = max_chord_size
        self.max_fret_span = max_fret_span

        # Precompute token sets for efficiency
        self.tab_ids: Set[int] = set(tokenizer.get_tab_token_ids())
        self.time_shift_ids: Set[int] = set(tokenizer.get_time_shift_token_ids())
        self.eos_id: int = tokenizer.shared_token_to_id["<eos>"]
        self.pad_id: int = tokenizer.shared_token_to_id["<pad>"]

        # Special handling for zero time shift
        self.zero_time_shift_id: int = tokenizer.shared_token_to_id.get("TIME_SHIFT<0>", -1)
        if self.zero_time_shift_id == -1:
            print("Warning: No TIME_SHIFT<0> token found. Chords may not be representable.")

        # Precompute string AND fret mappings for chord validation
        self.string_to_tab_ids: Dict[int, Set[int]] = {}
        self.tab_token_info: Dict[int, tuple] = {}  # token_id -> (string, fret)
        for token, token_id in tokenizer.shared_token_to_id.items():
            if token.startswith("TAB<"):
                # Extract string and fret from TAB<string,fret>
                parts = token[4:-1].split(",")
                if len(parts) == 2:
                    string_num, fret_num = int(parts[0]), int(parts[1])
                    self.tab_token_info[token_id] = (string_num, fret_num)
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
            chord_state = {"strings_used": set(), "frets_used": [], "in_chord": False}

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
            # Filter TABs to:
            # 1. Exclude strings already used in this chord
            # 2. Exclude frets that would make the chord span exceed max_fret_span

            current_frets = chord_state.get("frets_used", [])
            min_fret = min(current_frets) if current_frets else None
            max_fret = max(current_frets) if current_frets else None

            allowed_tabs = []
            for tab_id in self.tab_ids:
                if tab_id not in self.tab_token_info:
                    continue

                string_num, fret_num = self.tab_token_info[tab_id]

                # Skip if string already used
                if string_num in chord_state["strings_used"]:
                    continue

                # Fret span check - only for non-open strings (fret > 0)
                # Open strings don't require finger placement
                if fret_num > 0 and min_fret is not None:
                    new_min = min(min_fret, fret_num)
                    new_max = max(max_fret, fret_num)
                    if (new_max - new_min) > self.max_fret_span:
                        continue

                allowed_tabs.append(tab_id)

            return allowed_tabs

        elif last_token_id in self.time_shift_ids:
            # After TIME_SHIFT>0: can emit TAB (new onset) or EOS
            # All TABs are allowed since hand has time to move
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
        """Track chord state for each sequence in the batch.

        Returns a list of dicts with:
        - strings_used: Set of string numbers used in current chord
        - frets_used: List of fret numbers (excluding open strings) in current chord
        - in_chord: Whether we are in the middle of a chord (after TIME_SHIFT<0>)
        """
        batch_size = input_ids.shape[0]
        chord_states = []

        for b in range(batch_size):
            strings_used = set()
            frets_used: List[int] = []
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

                # If we hit a TAB token, add its string and fret to the tracking
                if token_id in self.tab_token_info:
                    string_num, fret_num = self.tab_token_info[token_id]
                    strings_used.add(string_num)
                    # Only track non-open strings for fret span calculation
                    if fret_num > 0:
                        frets_used.append(fret_num)

            chord_states.append({
                "strings_used": strings_used,
                "frets_used": frets_used,
                "in_chord": in_chord
            })

        return chord_states


class ForcedTokenLogitsProcessor(LogitsProcessor):
    """Forces specific tokens at specific generation steps."""

    def __init__(self, forced_schedule: Dict[int, int]):
        """Initialize forced token processor.
        
        Parameters
        ----------
        forced_schedule : Dict[int, int]
            Mapping from generation step to token ID to force
        """
        self.forced_schedule = forced_schedule
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Apply forced token constraints to logits.
        
        Parameters
        ----------
        input_ids : torch.LongTensor
            Input token IDs
        scores : torch.FloatTensor
            Generation logits
            
        Returns
        -------
        torch.FloatTensor
            Modified logits with forced tokens applied
        """
        current_step = input_ids.shape[1] - 1
        
        if current_step in self.forced_schedule:
            forced_token_id = self.forced_schedule[current_step]
            scores.fill_(float("-inf"))
            scores[:, forced_token_id] = 0.0
            
        return scores


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


# =============================================================================
# HuggingFace-compatible adapters (formerly in constraints.py)
# =============================================================================

class TabConstraintProcessor(LogitsProcessor):
    """Apply boolean masks produced by a constraint function to logits.
    
    This adapter wraps a constraint function to work with HuggingFace's
    LogitsProcessor interface for use with model.generate().
    """

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
    """Wrap the v3 constraint logic so it can be used with ``model.generate``.
    
    This creates a TabConstraintProcessor that enforces the v3 TAB ↔ TIME_SHIFT
    alternation pattern during generation.

    Args:
        tokenizer: The v3 tokenizer instance

    Returns:
        A TabConstraintProcessor ready for use with model.generate()
    """
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