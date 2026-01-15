"""Tokenization utilities for the Fretting-Transformer SynthTab acoustic pipeline.

This module implements the v3 tokenization strategy described in the
Fretting-Transformer paper. The encoder consumes MIDI events represented as
``NOTE_ON<pitch> TIME_SHIFT<duration> NOTE_OFF<pitch>`` triplets while the
decoder consumes tablature events represented as ``TAB<string,fret>
TIME_SHIFT<duration>`` pairs. The tokenizer is responsible for building the
vocabulary from the SynthTab acoustic subset, converting PrettyMIDI objects and
pre-computed tablature annotations into token sequences, and serialising the
vocabulary for later use.

The implementation purposely stays close to the textual representation used in
the paper so that we can easily inspect datasets and debug issues. Tokens are
stored as strings during vocabulary construction and converted to integer ids
through the ``token_to_id``/``id_to_token`` mappings for downstream training.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import pretty_midi

__all__ = [
    "TokenizerConfig",
    "TokenizedTrack",
    "MidiTabTokenizerV3",
    "build_tokenizer_from_manifests",
    "STANDARD_TUNING",
    "DEFAULT_CONDITIONING_TUNINGS",
]


# ---------------------------------------------------------------------------
# Tuning constants used across the conditioning pipeline
# ---------------------------------------------------------------------------

# Standard acoustic guitar tuning (string 1 == high E)
STANDARD_TUNING: Tuple[int, ...] = (64, 59, 55, 50, 45, 40)
HALF_STEP_DOWN_TUNING: Tuple[int, ...] = tuple(pitch - 1 for pitch in STANDARD_TUNING)
FULL_STEP_DOWN_TUNING: Tuple[int, ...] = tuple(pitch - 2 for pitch in STANDARD_TUNING)
DROP_D_TUNING: Tuple[int, ...] = (64, 59, 55, 50, 45, 38)

# Default collection of tunings used during augmentation/conditioning
DEFAULT_CONDITIONING_TUNINGS: Tuple[Tuple[int, ...], ...] = (
    STANDARD_TUNING,
    HALF_STEP_DOWN_TUNING,
    FULL_STEP_DOWN_TUNING,
    DROP_D_TUNING,
)


@dataclass
class TokenizerConfig:
    """Configuration controlling tokenizer behaviour.

    Attributes
    ----------
    time_shift_quantum_ms:
        Milliseconds used to quantise durations when generating ``TIME_SHIFT``
        tokens. For improved vocabulary efficiency, consider 50-100ms steps.
    max_duration_ms:
        Maximum duration to encode explicitly. Longer durations get capped
        to this value to reduce vocabulary size.
    quantization_strategy:
        Strategy for TIME_SHIFT binning:
        - 'linear': Fixed step size (time_shift_quantum_ms)
        - 'hybrid': Linear for short durations, logarithmic for long ones
        - 'capped': Linear with maximum cap
    max_fret:
        Highest fret value to encode per string. The Fretting-Transformer paper
        reports using up to the 24th fret for acoustic guitars.
    min_string:
        Lowest guitar string index (1 == high E).
    max_string:
        Highest guitar string index (6 == low E).
    """

    time_shift_quantum_ms: int = 100
    max_duration_ms: int = 5000       # Cap very long durations
    quantization_strategy: str = 'capped'  # Default to capped linear
    force_zero_time_shift: bool = True # Ensure TIME_SHIFT<0> for chord support
    max_fret: int = 24
    min_string: int = 1
    max_string: int = 6

    def quantise_duration(self, duration_ms: float, same_onset: bool = False) -> int:
        """Quantise ``duration_ms`` into a ``TIME_SHIFT`` bucket.

        Uses the configured quantization strategy to balance vocabulary size
        with temporal resolution. Caps durations at max_duration_ms.

        Args:
            duration_ms: Duration in milliseconds to quantize
            same_onset: True if this represents notes at the same onset (chord)

        Returns:
            Quantized duration in milliseconds
        """
        if duration_ms < 0:
            raise ValueError("Durations must be non-negative")

        # Special case: zero for same-onset chords
        if self.force_zero_time_shift and same_onset:
            return 0

        duration_ms = min(duration_ms, self.max_duration_ms)

        if self.quantization_strategy == 'linear':
            quantized = int(round(duration_ms / self.time_shift_quantum_ms)) * self.time_shift_quantum_ms
        elif self.quantization_strategy == 'capped':
            # Linear quantization with capping (recommended)
            quantized = int(round(duration_ms / self.time_shift_quantum_ms)) * self.time_shift_quantum_ms
        elif self.quantization_strategy == 'hybrid':
            # High resolution for short durations, coarser for long ones
            if duration_ms < 500:
                # Fine resolution for fast passages (25ms steps)
                quantized = int(round(duration_ms / 25)) * 25
            elif duration_ms < 2000:
                # Medium resolution for normal durations (100ms steps)
                quantized = int(round(duration_ms / 100)) * 100
            else:
                # Coarse resolution for long rests (500ms steps)
                quantized = int(round(duration_ms / 500)) * 500
        else:
            raise ValueError(f"Unknown quantization strategy: {self.quantization_strategy}")

        # Never return 0 unless same_onset=True (avoid collapsing short gaps)
        if quantized == 0 and not same_onset:
            quantized = self.time_shift_quantum_ms

        return quantized

    def estimate_time_shift_vocab_size(self) -> int:
        """Estimate the number of unique TIME_SHIFT tokens with current config."""
        if self.quantization_strategy == 'linear' or self.quantization_strategy == 'capped':
            return (self.max_duration_ms // self.time_shift_quantum_ms) + 1

        elif self.quantization_strategy == 'hybrid':
            # Count bins for each range
            short_bins = 500 // 25  # 0-500ms in 25ms steps = 20 bins
            medium_bins = (2000 - 500) // 100  # 500-2000ms in 100ms steps = 15 bins
            long_bins = (self.max_duration_ms - 2000) // 500  # 2000-5000ms in 500ms steps = 6 bins
            return short_bins + medium_bins + long_bins + 1

        return -1  # Unknown


@dataclass
class TokenizedTrack:
    """Container holding the tokenised representation of a track.

    Attributes
    ----------
    encoder_tokens:
        List of encoder-side tokens represented as strings.
    decoder_tokens:
        List of decoder-side tokens represented as strings.
    encoder_group_lengths:
        Length of each encoder event group (``NOTE_ON``, ``TIME_SHIFT``,
        ``NOTE_OFF``) in number of tokens. Used to chunk tracks without breaking
        events.
    decoder_group_lengths:
        Length of each decoder event group (``TAB``, ``TIME_SHIFT``).
    """

    encoder_tokens: List[str]
    decoder_tokens: List[str]
    encoder_group_lengths: List[int]
    decoder_group_lengths: List[int]
    note_metadata: List["NoteMetadata"]


@dataclass
class NoteMetadata:
    """Metadata describing a single tablature event."""

    string: int
    fret: int


@dataclass
class _Vocabulary:
    token_to_id: Dict[str, int]
    id_to_token: Dict[int, str]

    @classmethod
    def from_tokens(cls, tokens: Iterable[str], special_tokens: Sequence[str]) -> "_Vocabulary":
        token_to_id: Dict[str, int] = {}
        id_to_token: Dict[int, str] = {}

        # Insert special tokens first to maintain stable ids.
        for idx, tok in enumerate(special_tokens):
            token_to_id[tok] = idx
            id_to_token[idx] = tok

        for tok in tokens:
            if tok in token_to_id:
                continue
            idx = len(token_to_id)
            token_to_id[tok] = idx
            id_to_token[idx] = tok
        return cls(token_to_id=token_to_id, id_to_token=id_to_token)

    def to_json(self) -> Dict[str, Dict[str, int]]:
        return {
            "token_to_id": self.token_to_id,
        }

    @classmethod
    def from_json(cls, data: Dict[str, Dict[str, int]]) -> "_Vocabulary":
        token_to_id = data["token_to_id"]
        id_to_token = {int(idx): tok for tok, idx in token_to_id.items()}
        return cls(token_to_id=token_to_id, id_to_token=id_to_token)


class MidiTabTokenizerV3:
    """Tokenizer implementing the v3 SynthTab representation."""

    SPECIAL_TOKENS: Tuple[str, ...] = ("<pad>", "<eos>", "<unk>") + tuple(
        f"<extra_id_{i}>" for i in range(100)
    )

    def __init__(
        self,
        config: TokenizerConfig,
        encoder_vocab: _Vocabulary,
        decoder_vocab: _Vocabulary,
    ) -> None:
        self.config = config
        self.encoder_vocab = encoder_vocab
        self.decoder_vocab = decoder_vocab
        self._build_shared_vocabulary()

    # ------------------------------------------------------------------
    # Vocabulary helpers
    # ------------------------------------------------------------------
    @property
    def encoder_token_to_id(self) -> Dict[str, int]:
        return self.encoder_vocab.token_to_id

    @property
    def decoder_token_to_id(self) -> Dict[str, int]:
        return self.decoder_vocab.token_to_id

    @property
    def shared_token_to_id(self) -> Dict[str, int]:
        return self._shared_token_to_id

    @property
    def shared_id_to_token(self) -> Dict[int, str]:
        return self._shared_id_to_token

    def encode_encoder_tokens_shared(self, tokens: Sequence[str]) -> List[int]:
        return [self._encoder_to_shared.get(tok, self.shared_token_to_id["<unk>"]) for tok in tokens]

    def encode_decoder_tokens_shared(self, tokens: Sequence[str]) -> List[int]:
        return [self._decoder_to_shared.get(tok, self.shared_token_to_id["<unk>"]) for tok in tokens]

    def shared_to_decoder_tokens(self, ids: Sequence[int]) -> List[str]:
        vocab = self.decoder_vocab.token_to_id
        id_to_token = self.decoder_vocab.id_to_token
        tokens: List[str] = []
        for idx in ids:
            token = self._shared_id_to_token.get(int(idx))
            if token in vocab:
                tokens.append(token)
            else:
                tokens.append("<unk>")
        return tokens

    def shared_to_encoder_tokens(self, ids: Sequence[int]) -> List[str]:
        vocab = self.encoder_vocab.token_to_id
        id_to_token = self.encoder_vocab.id_to_token
        tokens: List[str] = []
        for idx in ids:
            token = self._shared_id_to_token.get(int(idx))
            if token in vocab:
                tokens.append(token)
            else:
                tokens.append("<unk>")
        return tokens

    def is_tab_token(self, token_id: int) -> bool:
        token = self.decoder_vocab.id_to_token.get(token_id)
        return token is not None and token.startswith("TAB<")

    def is_time_shift_token(self, token_id: int) -> bool:
        token = self.decoder_vocab.id_to_token.get(token_id)
        return token is not None and token.startswith("TIME_SHIFT<")

    def get_tab_token_ids(self) -> List[int]:
        """Get all TAB token IDs for constrained decoding."""
        return [
            self.shared_token_to_id[token]
            for token in self.decoder_vocab.token_to_id.keys()
            if token.startswith("TAB<")
        ]

    def get_time_shift_token_ids(self) -> List[int]:
        """Get all TIME_SHIFT token IDs for constrained decoding."""
        return [
            self.shared_token_to_id[token]
            for token in self.decoder_vocab.token_to_id.keys()
            if token.startswith("TIME_SHIFT<")
        ]

    def get_constrained_next_tokens(self, last_token_id: int) -> List[int]:
        """Get valid next tokens for constrained v3 decoding.

        v3 pattern: TAB<s,f> → TIME_SHIFT<d> → TAB<s,f> → TIME_SHIFT<d> ...
        EOS is only allowed after TIME_SHIFT to preserve TAB+TIME_SHIFT pairs.

        Args:
            last_token_id: The previous decoder token ID

        Returns:
            List of valid next token IDs
        """
        if self.is_tab_token(last_token_id):
            # After TAB, must emit TIME_SHIFT (no EOS allowed here)
            return self.get_time_shift_token_ids()
        elif self.is_time_shift_token(last_token_id):
            # After TIME_SHIFT, can emit TAB or end sequence
            return self.get_tab_token_ids() + [self.shared_token_to_id["<eos>"]]
        else:
            # Start of sequence or special token - allow TAB only
            return self.get_tab_token_ids()

    # ------------------------------------------------------------------
    # Serialisation utilities
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        payload = {
            "config": self.config.__dict__,
            "encoder_vocab": self.encoder_vocab.to_json(),
            "decoder_vocab": self.decoder_vocab.to_json(),
        }
        with open(os.path.join(path, "tokenizer.json"), "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "MidiTabTokenizerV3":
        with open(os.path.join(path, "tokenizer.json"), "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        config = TokenizerConfig(**payload["config"])
        encoder_vocab = _Vocabulary.from_json(payload["encoder_vocab"])
        decoder_vocab = _Vocabulary.from_json(payload["decoder_vocab"])
        return cls(config=config, encoder_vocab=encoder_vocab, decoder_vocab=decoder_vocab)

    # ------------------------------------------------------------------
    # Tokenisation API
    # ------------------------------------------------------------------
    def tokenize_track_from_jams(
        self,
        jams_events: Sequence[Dict[str, float]],
        *,
        capo: int = 0,
        tuning: Optional[Sequence[int]] = None,
    ) -> TokenizedTrack:
        """Tokenise a JAMS tablature track with optional tuning/capo conditioning."""

        if tuning is None:
            tuning = STANDARD_TUNING
        if len(tuning) != 6:
            raise ValueError("Tuning sequences must contain exactly six MIDI pitches")

        encoder_tokens: List[str] = []
        decoder_tokens: List[str] = []
        encoder_groups: List[int] = []
        decoder_groups: List[int] = []
        metadata: List[NoteMetadata] = []

        # Map strings (1 == high E) to MIDI pitches for open strings.
        open_strings = {string_idx + 1: int(pitch) for string_idx, pitch in enumerate(tuning)}

        # Sort events by time, then by string for consistent ordering within chords
        sorted_events = sorted(jams_events, key=lambda e: (e.get("time_ticks", 0), e.get("string", 0)))

        # Group events by onset time to detect chords
        onset_groups = []
        current_onset = None
        current_group = []

        for event in sorted_events:
            event_time = event.get("time_ticks", 0)
            if current_onset is None or abs(event_time - current_onset) < 1e-6:  # Same onset (within tolerance)
                current_group.append(event)
                current_onset = event_time
            else:
                if current_group:
                    onset_groups.append(current_group)
                current_group = [event]
                current_onset = event_time

        if current_group:
            onset_groups.append(current_group)

        # Process each onset group
        for group_idx, onset_group in enumerate(onset_groups):
            for event_idx, event in enumerate(onset_group):
                string = int(event["string"])
                fret = int(event["fret"])
                duration_ms = float(event["duration_ms"])

                # Validate ranges
                if not (self.config.min_string <= string <= self.config.max_string):
                    raise ValueError(f"Tab string {string} is outside configured range")
                if not (0 <= fret <= self.config.max_fret):
                    raise ValueError(f"Tab fret {fret} is outside configured range")

                # Calculate MIDI pitch from string and fret
                base_pitch = open_strings[string] + fret + int(capo)
                if base_pitch < 0 or base_pitch > 127:
                    raise ValueError(
                        f"Computed MIDI pitch {base_pitch} for string {string}, fret {fret}, capo {capo} is outside the 0-127 range"
                    )

                # Determine if this is within a chord
                is_chord_note = len(onset_group) > 1
                is_last_in_chord = event_idx == len(onset_group) - 1
                is_same_onset = is_chord_note and not is_last_in_chord

                # Generate duration token
                duration_token = self._time_shift_token(
                    duration_ms if is_last_in_chord else 0.0,  # Use 0 for chord internal transitions
                    same_onset=is_same_onset
                )

                # Generate encoder tokens (MIDI representation)
                encoder_tokens.extend([
                    self._note_on_token(base_pitch),
                    duration_token,
                    self._note_off_token(base_pitch),
                ])
                encoder_groups.append(3)

                metadata.append(NoteMetadata(string=string, fret=fret))

                # Generate decoder tokens (tablature representation)
                decoder_tokens.extend([
                    self._tab_token(string, fret),
                    duration_token,
                ])
                decoder_groups.append(2)

        return TokenizedTrack(
            encoder_tokens=encoder_tokens,
            decoder_tokens=decoder_tokens,
            encoder_group_lengths=encoder_groups,
            decoder_group_lengths=decoder_groups,
            note_metadata=metadata,
        )

    def encode_encoder_tokens(self, tokens: Sequence[str]) -> List[int]:
        return [self.encoder_vocab.token_to_id.get(tok, self.encoder_vocab.token_to_id["<unk>"]) for tok in tokens]

    def encode_decoder_tokens(self, tokens: Sequence[str]) -> List[int]:
        return [self.decoder_vocab.token_to_id.get(tok, self.decoder_vocab.token_to_id["<unk>"]) for tok in tokens]

    def decode_decoder_tokens(self, ids: Sequence[int]) -> List[str]:
        vocab = self.decoder_vocab.id_to_token
        return [vocab.get(int(idx), "<unk>") for idx in ids]

    # ------------------------------------------------------------------
    # Internal token constructors
    # ------------------------------------------------------------------
    @staticmethod
    def _note_on_token(pitch: int) -> str:
        return f"NOTE_ON<{pitch}>"

    @staticmethod
    def _note_off_token(pitch: int) -> str:
        return f"NOTE_OFF<{pitch}>"

    @staticmethod
    def _tab_token(string: int, fret: int) -> str:
        return f"TAB<{string},{fret}>"

    def _time_shift_token(self, duration_ms: float, same_onset: bool = False) -> str:
        quantised = self.config.quantise_duration(duration_ms, same_onset=same_onset)
        return f"TIME_SHIFT<{quantised}>"

    @staticmethod
    def _capo_token(capo: int) -> str:
        return f"CAPO<{int(capo)}>"

    @staticmethod
    def _tuning_token(tuning: Sequence[int]) -> str:
        values = ",".join(str(int(pitch)) for pitch in tuning)
        return f"TUNING<{values}>"

    def build_conditioning_prefix(self, capo: int, tuning: Sequence[int]) -> List[str]:
        """Create conditioning tokens describing capo position and tuning."""

        return [self._capo_token(capo), self._tuning_token(tuning)]

    def ensure_conditioning_tokens(
        self,
        capo_values: Sequence[int],
        tuning_options: Sequence[Sequence[int]],
    ) -> None:
        """Add conditioning tokens to the encoder vocabulary when required."""

        added = False

        for capo in capo_values:
            token = self._capo_token(capo)
            if token not in self.encoder_vocab.token_to_id:
                idx = len(self.encoder_vocab.token_to_id)
                self.encoder_vocab.token_to_id[token] = idx
                self.encoder_vocab.id_to_token[idx] = token
                added = True

        for tuning in tuning_options:
            token = self._tuning_token(tuning)
            if token not in self.encoder_vocab.token_to_id:
                idx = len(self.encoder_vocab.token_to_id)
                self.encoder_vocab.token_to_id[token] = idx
                self.encoder_vocab.id_to_token[idx] = token
                added = True

        if added:
            self._build_shared_vocabulary()

    def _build_shared_vocabulary(self) -> None:
        """Merge encoder/decoder vocabularies into a shared dictionary.

        Hugging Face's T5 implementation assumes a single embedding table. To
        maintain compatibility we build a shared vocabulary that contains all
        encoder and decoder tokens. Tokens keep their semantics via the
        ``_encoder_to_shared`` and ``_decoder_to_shared`` lookup tables.
        """

        shared_tokens: List[str] = []
        seen: Dict[str, int] = {}

        for tok in self.SPECIAL_TOKENS:
            shared_tokens.append(tok)
            seen[tok] = len(seen)

        for vocab in (self.encoder_vocab, self.decoder_vocab):
            for tok in vocab.token_to_id:
                if tok not in seen:
                    seen[tok] = len(seen)
                    shared_tokens.append(tok)

        self._shared_token_to_id = {tok: idx for idx, tok in enumerate(shared_tokens)}
        self._shared_id_to_token = {idx: tok for tok, idx in self._shared_token_to_id.items()}

        self._encoder_to_shared = {
            tok: self._shared_token_to_id.get(tok, self._shared_token_to_id["<unk>"])
            for tok in self.encoder_vocab.token_to_id
        }
        self._decoder_to_shared = {
            tok: self._shared_token_to_id.get(tok, self._shared_token_to_id["<unk>"])
            for tok in self.decoder_vocab.token_to_id
        }

