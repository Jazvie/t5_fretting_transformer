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
]
