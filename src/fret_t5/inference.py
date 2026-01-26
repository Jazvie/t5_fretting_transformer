"""Inference utilities for Fretting-Transformer v3."""

from __future__ import annotations

import os
import torch
from transformers import LogitsProcessorList, T5Config, T5ForConditionalGeneration
from typing import Dict, List, Optional, Tuple

from .tokenization import MidiTabTokenizerV3, STANDARD_TUNING, DEFAULT_CONDITIONING_TUNINGS
from .training import ModelConfig
from .constrained_generation import V3ConstrainedProcessor, ForcedTokenLogitsProcessor
from .postprocess import (
    TimingContext,
    TabEvent,
    midi_notes_to_encoder_tokens_with_timing,
    postprocess_with_timing,
    tab_events_to_dict_list,
)

# Constants for chunking (matching training configuration)
MAX_ENCODER_LENGTH = 512
CONDITIONING_TOKENS = 2  # CAPO + TUNING
TOKENS_PER_NOTE = 3  # NOTE_ON, TIME_SHIFT, NOTE_OFF
MAX_NOTES_PER_CHUNK = (MAX_ENCODER_LENGTH - CONDITIONING_TOKENS) // TOKENS_PER_NOTE  # ~170
OVERLAP_NOTES = 4  # Match training overlap


class FretT5Inference:
    """Inference pipeline for Fretting-Transformer v3."""

    def __init__(self, checkpoint_path: str, tokenizer_path: str = "universal_tokenizer", device: Optional[str] = None, max_fret_span: int = 5):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_fret_span = max_fret_span

        if not os.path.exists(tokenizer_path):
             raise ValueError(f"Tokenizer not found at {tokenizer_path}")
        self.tokenizer = MidiTabTokenizerV3.load(tokenizer_path)

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Get vocab size from checkpoint to detect if conditioning was used
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        checkpoint_vocab_size = state_dict['encoder.embed_tokens.weight'].shape[0]

        # Always ensure conditioning tokens are available - the model may have been
        # trained with them even if 'conditioning_enabled' flag isn't in checkpoint
        # (vocab size 572 = base 560 + 12 conditioning tokens)
        self.tokenizer.ensure_conditioning_tokens(
            capo_values=tuple(range(8)),
            tuning_options=DEFAULT_CONDITIONING_TUNINGS
        )

        if "model_config" in checkpoint:
            self.config = checkpoint["model_config"]
        else:
            self.config = ModelConfig(use_pretrained=False, d_model=128, num_layers=3)

        # Create model with correct vocab size from checkpoint
        hf_config = T5Config(
            vocab_size=checkpoint_vocab_size,
            d_model=self.config.tiny_dims.get("d_model", 128),
            d_ff=self.config.tiny_dims.get("d_ff", 512),
            num_layers=self.config.tiny_dims.get("num_layers", 4),
            num_heads=self.config.tiny_dims.get("num_heads", 4),
            dropout_rate=self.config.tiny_dims.get("dropout_rate", 0.1),
            is_encoder_decoder=True,
            decoder_start_token_id=self.tokenizer.shared_token_to_id.get("<sos>", 0),
            eos_token_id=self.tokenizer.shared_token_to_id["<eos>"],
            pad_token_id=self.tokenizer.shared_token_to_id["<pad>"],
        )
        self.model = T5ForConditionalGeneration(hf_config)

        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, 
                midi_notes: List[Dict], 
                capo: int = 0, 
                tuning: tuple = STANDARD_TUNING, 
                forced_tokens: Optional[Dict[int, int]] = None) -> List[str]:
        """Generate tablature from MIDI notes.
        
        Parameters
        ----------
        midi_notes : List[Dict]
            List of dicts with 'pitch' and 'duration' keys
        capo : int, optional
            Capo position for conditioning
        tuning : tuple, optional
            Tuning tuple for conditioning
        forced_tokens : Optional[Dict[int, int]], optional
            Dict of {step: token_id} to force specific outputs
            
        Returns
        -------
        List[str]
            List of decoded tablature tokens
        """
        encoder_tokens = self._notes_to_tokens(midi_notes)
        prefix = self.tokenizer.build_conditioning_prefix(capo, tuning)
        full_tokens = prefix + encoder_tokens
        
        input_ids = self.tokenizer.encode_encoder_tokens_shared(full_tokens)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        logits_processors: List = [V3ConstrainedProcessor(self.tokenizer, max_fret_span=self.max_fret_span)]
        if forced_tokens:
            logits_processors.append(ForcedTokenLogitsProcessor(forced_tokens))
            
        with torch.no_grad():
            outputs = self.model.generate(
                input_tensor,
                max_length=512,
                num_beams=1,
                do_sample=False,
                eos_token_id=self.tokenizer.shared_token_to_id["<eos>"],
                pad_token_id=self.tokenizer.shared_token_to_id["<pad>"],
                logits_processor=LogitsProcessorList(logits_processors)
            )

        # Use shared_to_decoder_tokens since model outputs shared vocab IDs
        return self.tokenizer.shared_to_decoder_tokens(outputs[0].cpu().tolist())

    def _notes_to_tokens(self, notes: List[Dict]) -> List[str]:
        """Convert note list to encoder tokens handling chords correctly."""
        sorted_notes = sorted(notes, key=lambda x: x.get('start', 0))
        
        tokens = []
        for i, n in enumerate(sorted_notes):
            dur_ms = int(round(n['duration'] * 1000 / 100)) * 100
            
            is_chord = False
            if i < len(sorted_notes) - 1:
                current_start = n.get('start', 0)
                next_start = sorted_notes[i+1].get('start', 0)
                if abs(next_start - current_start) < 0.01:
                    is_chord = True
            
            token_dur = 0 if is_chord else dur_ms
            
            if token_dur == 0 and not is_chord:
                token_dur = 100

            tokens.extend([
                f"NOTE_ON<{n['pitch']}>",
                f"TIME_SHIFT<{token_dur}>",
                f"NOTE_OFF<{n['pitch']}>"
            ])
        return tokens

    def _predict_single_chunk(
        self,
        midi_notes: List[Dict],
        capo: int,
        tuning: tuple,
        pitch_window: int,
        alignment_window: int,
        max_fret_span: int = 5,
        enforce_playability: bool = True,
    ) -> List[TabEvent]:
        """Run inference on a single chunk of notes (internal method)."""
        # Create encoder tokens and timing context
        encoder_tokens, timing_context = midi_notes_to_encoder_tokens_with_timing(
            midi_notes,
            time_shift_quantum_ms=self.tokenizer.config.time_shift_quantum_ms,
            max_duration_ms=self.tokenizer.config.max_duration_ms,
        )

        # Add conditioning prefix
        prefix = self.tokenizer.build_conditioning_prefix(capo, tuning)
        full_tokens = prefix + encoder_tokens

        # Encode and run model
        input_ids = self.tokenizer.encode_encoder_tokens_shared(full_tokens)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)

        logits_processors: List = [V3ConstrainedProcessor(self.tokenizer, max_fret_span=self.max_fret_span)]

        with torch.no_grad():
            outputs = self.model.generate(
                input_tensor,
                max_length=512,
                num_beams=1,
                do_sample=False,
                eos_token_id=self.tokenizer.shared_token_to_id["<eos>"],
                pad_token_id=self.tokenizer.shared_token_to_id["<pad>"],
                logits_processor=LogitsProcessorList(logits_processors)
            )

        # Decode tokens (use shared vocab since model outputs shared IDs)
        decoder_tokens = self.tokenizer.shared_to_decoder_tokens(outputs[0].cpu().tolist())

        # Postprocess with timing reconstruction
        tab_events = postprocess_with_timing(
            encoder_tokens=full_tokens,
            decoder_tokens=decoder_tokens,
            timing_context=timing_context,
            capo=capo,
            tuning=tuning,
            pitch_window=pitch_window,
            alignment_window=alignment_window,
            max_fret_span=max_fret_span,
            enforce_playability=enforce_playability,
        )

        return tab_events

    def predict_with_timing(
        self,
        midi_notes: List[Dict],
        capo: int = 0,
        tuning: tuple = STANDARD_TUNING,
        pitch_window: int = 5,
        alignment_window: int = 5,
        forced_tokens: Optional[Dict[int, int]] = None,
        return_dict: bool = False,
        max_fret_span: int = 5,
        enforce_playability: bool = True,
    ) -> List[TabEvent] | List[Dict]:
        """Generate tablature from MIDI notes with original timing preserved.

        This is the recommended method for audio-to-tab pipelines where you need
        the output tablature to have continuous timestamps matching the original
        MIDI input (not quantized to 100ms steps).

        Automatically handles long sequences by chunking (matching training config):
        - Max ~170 notes per chunk
        - 4-note overlap between chunks for continuity

        Parameters
        ----------
        midi_notes : List[Dict]
            List of dicts with keys:
            - 'pitch': MIDI pitch (int, 0-127)
            - 'start': onset time in seconds (float)
            - 'duration': duration in seconds (float)
        capo : int, optional
            Capo position for conditioning (default: 0)
        tuning : tuple, optional
            Tuning tuple for conditioning (default: standard tuning)
        pitch_window : int, optional
            Maximum pitch difference for correction in semitones (default: 5)
        alignment_window : int, optional
            Window size for aligning input/output sequences (default: 5)
        forced_tokens : Optional[Dict[int, int]], optional
            Dict of {step: token_id} to force specific outputs (only used for single chunk)
        return_dict : bool, optional
            If True, return list of dicts instead of TabEvent objects (default: False)
        max_fret_span : int, optional
            Maximum allowed fret span for playable chords (default: 5)
        enforce_playability : bool, optional
            If True, apply fret span constraint to chords (default: True)

        Returns
        -------
        List[TabEvent] or List[Dict]
            List of tab events with continuous timing. Each event has:
            - string: Guitar string (1-6)
            - fret: Fret number (0-24)
            - onset_sec: Original onset time from MIDI
            - duration_sec: Original duration from MIDI
            - midi_pitch: MIDI pitch produced by this position

        Example
        -------
        >>> inference = FretT5Inference("checkpoint.pt")
        >>> midi_notes = [
        ...     {'pitch': 60, 'start': 0.0, 'duration': 0.5},
        ...     {'pitch': 64, 'start': 0.55, 'duration': 0.3},
        ... ]
        >>> tab_events = inference.predict_with_timing(midi_notes)
        >>> for event in tab_events:
        ...     print(f"String {event.string}, Fret {event.fret} at {event.onset_sec:.3f}s")
        """
        # Sort notes by start time
        sorted_notes = sorted(midi_notes, key=lambda x: (x.get('start', x.get('onset', 0)), x['pitch']))

        # If sequence fits in single chunk, use simple path
        if len(sorted_notes) <= MAX_NOTES_PER_CHUNK:
            tab_events = self._predict_single_chunk(
                sorted_notes, capo, tuning, pitch_window, alignment_window,
                max_fret_span, enforce_playability
            )
            if return_dict:
                return tab_events_to_dict_list(tab_events)
            return tab_events

        # Chunk the sequence with overlap
        all_tab_events: List[TabEvent] = []
        chunk_start = 0
        chunk_idx = 0

        while chunk_start < len(sorted_notes):
            chunk_end = min(chunk_start + MAX_NOTES_PER_CHUNK, len(sorted_notes))
            chunk_notes = sorted_notes[chunk_start:chunk_end]

            # Run inference on this chunk
            chunk_events = self._predict_single_chunk(
                chunk_notes, capo, tuning, pitch_window, alignment_window,
                max_fret_span, enforce_playability
            )

            if chunk_idx == 0:
                # First chunk: take all events
                all_tab_events.extend(chunk_events)
            else:
                # Subsequent chunks: skip overlap region (first OVERLAP_NOTES events)
                # to avoid duplicates from the previous chunk
                events_to_add = chunk_events[OVERLAP_NOTES:] if len(chunk_events) > OVERLAP_NOTES else []
                all_tab_events.extend(events_to_add)

            # Move to next chunk with overlap
            chunk_start += MAX_NOTES_PER_CHUNK - OVERLAP_NOTES
            chunk_idx += 1

        if return_dict:
            return tab_events_to_dict_list(all_tab_events)
        return all_tab_events
    
    def predict_raw(
        self,
        midi_notes: List[Dict],
        capo: int = 0,
        tuning: tuple = STANDARD_TUNING,
        forced_tokens: Optional[Dict[int, int]] = None,
    ) -> Tuple[List[str], List[str], TimingContext]:
        """Generate tablature and return raw tokens plus timing context.
        
        Useful for debugging or custom postprocessing pipelines.
        
        Parameters
        ----------
        midi_notes : List[Dict]
            List of dicts with 'pitch', 'start', 'duration' keys
        capo : int, optional
            Capo position for conditioning
        tuning : tuple, optional
            Tuning tuple for conditioning
        forced_tokens : Optional[Dict[int, int]], optional
            Dict of {step: token_id} to force specific outputs
            
        Returns
        -------
        Tuple[List[str], List[str], TimingContext]
            - encoder_tokens: Full encoder token sequence (with conditioning prefix)
            - decoder_tokens: Raw model output tokens
            - timing_context: TimingContext for timing reconstruction
        """
        # Create encoder tokens and timing context
        encoder_tokens, timing_context = midi_notes_to_encoder_tokens_with_timing(
            midi_notes,
            time_shift_quantum_ms=self.tokenizer.config.time_shift_quantum_ms,
            max_duration_ms=self.tokenizer.config.max_duration_ms,
        )
        
        # Add conditioning prefix
        prefix = self.tokenizer.build_conditioning_prefix(capo, tuning)
        full_tokens = prefix + encoder_tokens
        
        # Encode and run model
        input_ids = self.tokenizer.encode_encoder_tokens_shared(full_tokens)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        logits_processors: List = [V3ConstrainedProcessor(self.tokenizer, max_fret_span=self.max_fret_span)]
        if forced_tokens:
            logits_processors.append(ForcedTokenLogitsProcessor(forced_tokens))

        with torch.no_grad():
            outputs = self.model.generate(
                input_tensor,
                max_length=512,
                num_beams=1,
                do_sample=False,
                eos_token_id=self.tokenizer.shared_token_to_id["<eos>"],
                pad_token_id=self.tokenizer.shared_token_to_id["<pad>"],
                logits_processor=LogitsProcessorList(logits_processors)
            )

        # Use shared_to_decoder_tokens since model outputs shared vocab IDs
        decoder_tokens = self.tokenizer.shared_to_decoder_tokens(outputs[0].cpu().tolist())

        return full_tokens, decoder_tokens, timing_context