"""Fretting-Transformer SynthTab acoustic training package."""

from .tokenization import (
    MidiTabTokenizerV3,
    TokenizerConfig,
    build_tokenizer_from_manifests,
    STANDARD_TUNING,
    DEFAULT_CONDITIONING_TUNINGS,
)
from .data import DataConfig, SynthTabTokenDataset, chunk_tokenized_track, load_manifest, create_song_level_splits
from .metrics import compute_tab_accuracy
from .training import ModelConfig, TrainingConfig, TabSeq2SeqTrainer, train_model, create_model
from .hf_tokenizer import build_hf_tokenizer
from .collators import TabSeq2SeqCollator
from .postprocess import (
    TimingContext,
    NoteTimingInfo,
    TabEvent,
    midi_notes_to_encoder_tokens_with_timing,
    postprocess_with_timing,
    postprocess_to_timed_tabs,
    tab_events_to_dict_list,
    postprocess_decoder_tokens,
)
from .inference import FretT5Inference

__all__ = [
    "MidiTabTokenizerV3",
    "TokenizerConfig",
    "build_tokenizer_from_manifests",
    "STANDARD_TUNING",
    "DEFAULT_CONDITIONING_TUNINGS",
    "DataConfig",
    "SynthTabTokenDataset",
    "chunk_tokenized_track",
    "load_manifest",
    "create_song_level_splits",
    "compute_tab_accuracy",
    "build_hf_tokenizer",
    "TabSeq2SeqCollator",
    "ModelConfig",
    "TrainingConfig",
    "TabSeq2SeqTrainer",
    "create_model",
    "train_model",
    # Timing-aware inference and postprocessing
    "FretT5Inference",
    "TimingContext",
    "NoteTimingInfo", 
    "TabEvent",
    "midi_notes_to_encoder_tokens_with_timing",
    "postprocess_with_timing",
    "postprocess_to_timed_tabs",
    "tab_events_to_dict_list",
    "postprocess_decoder_tokens",
]
