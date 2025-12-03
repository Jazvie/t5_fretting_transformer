"""Inference utilities for Fretting-Transformer v3."""

from __future__ import annotations

import os
import torch
from transformers import LogitsProcessorList
from typing import Dict, List, Optional

from .tokenization import MidiTabTokenizerV3, STANDARD_TUNING
from .training import create_model, ModelConfig
from .constrained_generation import V3ConstrainedProcessor, ForcedTokenLogitsProcessor


class FretT5Inference:
    """Inference pipeline for Fretting-Transformer v3."""

    def __init__(self, checkpoint_path: str, tokenizer_path: str = "universal_tokenizer", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        if not os.path.exists(tokenizer_path):
             raise ValueError(f"Tokenizer not found at {tokenizer_path}")
        self.tokenizer = MidiTabTokenizerV3.load(tokenizer_path)
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
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