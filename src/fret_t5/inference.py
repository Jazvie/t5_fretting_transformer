"""Inference utilities for Fretting-Transformer v3."""

from __future__ import annotations

import os
import torch
from transformers import LogitsProcessorList
from typing import Dict, List, Optional, Tuple

from .tokenization import MidiTabTokenizerV3, STANDARD_TUNING
from .training import create_model, ModelConfig
from .constrained_generation import V3ConstrainedProcessor, ForcedTokenLogitsProcessor
from .postprocess import (
    TimingContext,
    TabEvent,
    midi_notes_to_encoder_tokens_with_timing,
    postprocess_with_timing,
    tab_events_to_dict_list,
)


class FretT5Inference:
    """Inference pipeline for Fretting-Transformer v3."""

    def __init__(self, checkpoint_path: str, tokenizer_path: str = "universal_tokenizer", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        if not os.path.exists(tokenizer_path):
             raise ValueError(f"Tokenizer not found at {tokenizer_path}")
        self.tokenizer = MidiTabTokenizerV3.load(tokenizer_path)
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        if "model_config" in checkpoint:
            self.config = checkpoint["model_config"]
        else:
            self.config = ModelConfig(use_pretrained=False, d_model=128, num_layers=3)

        self.model = create_model(self.tokenizer, self.config)
        
        state_dict = checkpoint.get("model_state_dict", checkpoint)
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
        
        logits_processors: List = [V3ConstrainedProcessor(self.tokenizer)]
        if forced_tokens:
            logits_processors.append(ForcedTokenLogitsProcessor(forced_tokens))
            
        with torch.no_grad():
            outputs = self.model.generate(
                input_tensor,
                max_length=512,
                logits_processor=LogitsProcessorList(logits_processors)
            )
            
        return self.tokenizer.decode_decoder_tokens(outputs[0].cpu().tolist())

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

    def predict_with_timing(
        self,
        midi_notes: List[Dict],
        capo: int = 0,
        tuning: tuple = STANDARD_TUNING,
        pitch_window: int = 5,
        alignment_window: int = 5,
        forced_tokens: Optional[Dict[int, int]] = None,
        return_dict: bool = False,
    ) -> List[TabEvent] | List[Dict]:
        """Generate tablature from MIDI notes with original timing preserved.
        
        This is the recommended method for audio-to-tab pipelines where you need
        the output tablature to have continuous timestamps matching the original
        MIDI input (not quantized to 100ms steps).
        
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
            Dict of {step: token_id} to force specific outputs
        return_dict : bool, optional
            If True, return list of dicts instead of TabEvent objects (default: False)
            
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
        
        logits_processors: List = [V3ConstrainedProcessor(self.tokenizer)]
        if forced_tokens:
            logits_processors.append(ForcedTokenLogitsProcessor(forced_tokens))
            
        with torch.no_grad():
            outputs = self.model.generate(
                input_tensor,
                max_length=512,
                logits_processor=LogitsProcessorList(logits_processors)
            )
        
        # Decode tokens
        decoder_tokens = self.tokenizer.decode_decoder_tokens(outputs[0].cpu().tolist())
        
        # Postprocess with timing reconstruction
        tab_events = postprocess_with_timing(
            encoder_tokens=full_tokens,
            decoder_tokens=decoder_tokens,
            timing_context=timing_context,
            capo=capo,
            tuning=tuning,
            pitch_window=pitch_window,
            alignment_window=alignment_window,
        )
        
        if return_dict:
            return tab_events_to_dict_list(tab_events)
        return tab_events
    
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
        
        logits_processors: List = [V3ConstrainedProcessor(self.tokenizer)]
        if forced_tokens:
            logits_processors.append(ForcedTokenLogitsProcessor(forced_tokens))
            
        with torch.no_grad():
            outputs = self.model.generate(
                input_tensor,
                max_length=512,
                logits_processor=LogitsProcessorList(logits_processors)
            )
        
        decoder_tokens = self.tokenizer.decode_decoder_tokens(outputs[0].cpu().tolist())
        
        return full_tokens, decoder_tokens, timing_context