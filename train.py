#!/usr/bin/env python3
"""
Unified Training Script for Fretting-Transformer

This script consolidates training for all supported datasets:
- SynthTab (synthetic guitar tablature)
- GuitarSet (real guitar recordings with annotations)
- DadaGP (Guitar Pro extracted data)
- Custom datasets (user-provided JSONL manifests)

Usage Examples:
    # Train on SynthTab with conditioning (default)
    python train.py --dataset synthtab

    # Train on SynthTab without conditioning
    python train.py --dataset synthtab --no-conditioning

    # Train on GuitarSet with conditioning
    python train.py --dataset guitarset --conditioning

    # Train on DadaGP
    python train.py --dataset dadagp

    # Finetune from a checkpoint
    python train.py --dataset guitarset --pretrained-checkpoint checkpoints/best_model.pt

    # Use LoRA for efficient finetuning
    python train.py --dataset guitarset --pretrained-checkpoint checkpoints/best_model.pt --use-lora

    # Custom dataset
    python train.py --dataset custom --train-manifest data/my_train.jsonl --val-manifest data/my_val.jsonl

    # Use HuggingFace pretrained T5
    python train.py --dataset synthtab --use-t5-pretrained --model-name t5-small
"""

import argparse
import json
import random
import sys
import torch
from dataclasses import replace
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Sequence

sys.path.insert(0, str(Path(__file__).parent / "src"))

from fret_t5 import (
    MidiTabTokenizerV3,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    train_model,
    create_model,
    DEFAULT_CONDITIONING_TUNINGS,
    STANDARD_TUNING,
    SynthTabTokenDataset,
    chunk_tokenized_track,
)
from fret_t5.data import load_manifest
from fret_t5.tokenization import NoteMetadata
from fret_t5.hf_tokenizer import build_hf_tokenizer


# =============================================================================
# Dataset Configuration Presets
# =============================================================================

DATASET_PRESETS = {
    "synthtab": {
        "train_manifest": "data/synthtab_acoustic_train.jsonl",
        "val_manifest": "data/synthtab_acoustic_val.jsonl",
        "description": "SynthTab synthetic guitar tablature",
        "default_conditioning": True,
        "default_epochs": 120,
        "default_batch_size": 16,
        "default_eval_batch_size": 8,
    },
    "guitarset": {
        "description": "GuitarSet real guitar recordings (requires --guitarset-dir)",
        "default_conditioning": True,
        "default_epochs": 150,
        "default_batch_size": 16,
        "default_eval_batch_size": 8,
    },
    "dadagp": {
        "train_manifest": "data/dadagp_acoustic_train.jsonl",
        "val_manifest": "data/dadagp_acoustic_val.jsonl",
        "description": "DadaGP Guitar Pro extracted data",
        "default_conditioning": True,
        "default_epochs": 120,
        "default_batch_size": 16,
        "default_eval_batch_size": 8,
    },
    "custom": {
        "description": "Custom dataset from user-provided manifests",
        "default_conditioning": False,
        "default_epochs": 100,
        "default_batch_size": 16,
        "default_eval_batch_size": 8,
    },
}


# =============================================================================
# GuitarSet Dataset (specialized loader)
# =============================================================================

class GuitarSetDataset(torch.utils.data.Dataset):
    """Dataset for GuitarSet with optional conditioning support."""

    def __init__(
        self,
        tokenizer: MidiTabTokenizerV3,
        file_paths: List[Path],
        data_config: DataConfig,
        split_name: str = "train",
        inverted_strings: bool = False,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.data_config = data_config
        self.split_name = split_name
        self.examples: List[Dict[str, torch.Tensor]] = []
        self._rng = random.Random(data_config.augmentation_seed)
        self.inverted_strings = inverted_strings

        self._preload_examples(file_paths)


    def _preload_examples(self, file_paths: List[Path]) -> None:
        """Load and tokenize all GuitarSet files."""
        # Import here to avoid circular imports
        from scripts.guitarset_loader import (
            load_guitarset_jams,
            extract_tablature_from_guitarset_jams,
        )

        skipped = 0
        is_training = self.split_name.lower() in {"train", "all"}

        # Setup conditioning parameters
        if self.data_config.enable_conditioning:
            capo_values = (
                self.data_config.conditioning_capo_values_train
                if is_training
                else self.data_config.conditioning_capo_values_eval
            )
            tuning_options = (
                self.data_config.conditioning_tunings_train
                if is_training
                else self.data_config.conditioning_tunings_eval
            )
            if not capo_values:
                capo_values = (0,)
            if not tuning_options:
                tuning_options = (STANDARD_TUNING,)

            self.tokenizer.ensure_conditioning_tokens(capo_values, tuning_options)

            sample_prefix = self.tokenizer.build_conditioning_prefix(capo_values[0], tuning_options[0])
            chunk_config = replace(
                self.data_config,
                max_encoder_length=self.data_config.max_encoder_length - len(sample_prefix)
            )
        else:
            capo_values = (0,)
            tuning_options = (STANDARD_TUNING,)
            chunk_config = self.data_config

        for file_path in file_paths:
            try:
                jams_data = load_guitarset_jams(file_path)
                tab_events = extract_tablature_from_guitarset_jams(
                    jams_data,
                    auto_detect_tuning=False,
                    inverted_string_convention=self.inverted_strings,
                )

                if len(tab_events) < 10:
                    skipped += 1
                    continue

                # Convert to JAMS format for tokenizer
                jams_events = []
                for event in tab_events:
                    jams_events.append({
                        "string": float(event["string"]),
                        "fret": float(event["fret"]),
                        "duration_ms": float(event.get("duration", 0.5) * 1000),
                        "time_ticks": float(event.get("time", 0) * 1000)
                    })

                # Tokenize
                tokenized = self.tokenizer.tokenize_track_from_jams(jams_events)
                chunks = list(chunk_tokenized_track(tokenized, chunk_config))

                if not self.data_config.enable_conditioning:
                    for enc_tokens, dec_tokens, _ in chunks:
                        self._append_example(list(enc_tokens), list(dec_tokens))
                else:
                    for enc_tokens, dec_tokens, note_metadata in chunks:
                        for capo in capo_values:
                            if is_training and self.data_config.randomize_tuning_per_sequence:
                                tuning_choices = [self._rng.choice(tuning_options)]
                            else:
                                tuning_choices = list(tuning_options)

                            for tuning in tuning_choices:
                                conditioned_encoder = self._apply_conditioning(
                                    list(enc_tokens),
                                    list(note_metadata),
                                    capo,
                                    tuning,
                                )
                                prefix_tokens = self.tokenizer.build_conditioning_prefix(capo, tuning)
                                final_encoder = prefix_tokens + conditioned_encoder
                                self._append_example(final_encoder, list(dec_tokens))

            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"   Skipped {file_path.name}: {e}")
                skipped += 1
                continue

        print(f"   Created {len(self.examples)} examples from {len(file_paths) - skipped} files ({skipped} skipped)")

        if self.data_config.enable_conditioning and is_training:
            print(f"   Conditioning: {len(capo_values)} capos x {len(tuning_options)} tunings")

    @staticmethod
    def _apply_conditioning(
        encoder_tokens: List[str],
        note_metadata: List[NoteMetadata],
        capo: int,
        tuning: Tuple[int, ...],
    ) -> List[str]:
        """Apply pitch transformation for conditioning."""
        conditioned_tokens = list(encoder_tokens)
        if not note_metadata:
            return conditioned_tokens

        tuning_values = [int(v) for v in tuning]
        pointer = 0

        for note in note_metadata:
            while pointer < len(conditioned_tokens) and not conditioned_tokens[pointer].startswith("NOTE_ON<"):
                pointer += 1

            if pointer >= len(conditioned_tokens):
                break

            base_pitch = tuning_values[note.string - 1] + note.fret + int(capo)
            base_pitch = max(0, min(127, base_pitch))
            conditioned_tokens[pointer] = f"NOTE_ON<{base_pitch}>"

            off_idx = pointer + 1
            while off_idx < len(conditioned_tokens) and not conditioned_tokens[off_idx].startswith("NOTE_OFF<"):
                off_idx += 1

            if off_idx < len(conditioned_tokens):
                conditioned_tokens[off_idx] = f"NOTE_OFF<{base_pitch}>"
                pointer = off_idx + 1
            else:
                pointer += 1

        return conditioned_tokens

    def _append_example(self, enc_tokens: List[str], dec_tokens: List[str]) -> None:
        """Convert tokens to tensor example."""
        input_ids = self.tokenizer.encode_encoder_tokens_shared(enc_tokens)
        labels = self.tokenizer.encode_decoder_tokens_shared(dec_tokens)

        input_ids = input_ids[:self.data_config.max_encoder_length]
        labels = labels[:self.data_config.max_decoder_length]

        attention_mask = [1] * len(input_ids)

        pad_id = self.tokenizer.shared_token_to_id.get("<pad>", 0)
        while len(input_ids) < self.data_config.max_encoder_length:
            input_ids.append(pad_id)
            attention_mask.append(0)

        if self.data_config.train_on_time_shift:
            loss_values = [
                self.data_config.tab_loss_weight if token.startswith("TAB<")
                else 1.0 if token.startswith("TIME_SHIFT<")
                else 1.0 if token == "<eos>"
                else 0.0
                for token in dec_tokens
            ]
        else:
            loss_values = [
                1.0 if token.startswith("TAB<") or token == "<eos>" else 0.0
                for token in dec_tokens
            ]

        while len(labels) < self.data_config.max_decoder_length:
            labels.append(-100)
            loss_values.append(0.0)

        self.examples.append({
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'loss_mask': torch.tensor(loss_values, dtype=torch.float),
        })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


# =============================================================================
# GuitarSet Splits Helper
# =============================================================================

def create_guitarset_splits(
    guitarset_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Path], List[Path], List[Path], Dict]:
    """Create song-level train/val/test splits for GuitarSet."""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

    all_files = sorted(guitarset_dir.glob("*.jams"))
    print(f"\n   Found {len(all_files)} GuitarSet files")

    # Group by song
    songs_to_files = defaultdict(list)
    for file_path in all_files:
        parts = file_path.stem.split('_')
        if len(parts) >= 2:
            song_id = '_'.join(parts[1:-1])
            songs_to_files[song_id].append(file_path)

    print(f"   Found {len(songs_to_files)} unique songs")

    # Split songs
    song_ids = list(songs_to_files.keys())
    random.seed(seed)
    random.shuffle(song_ids)

    n_songs = len(song_ids)
    train_end = int(n_songs * train_ratio)
    val_end = int(n_songs * (train_ratio + val_ratio))

    train_songs = song_ids[:train_end]
    val_songs = song_ids[train_end:val_end]
    test_songs = song_ids[val_end:]

    train_files = [f for s in train_songs for f in songs_to_files[s]]
    val_files = [f for s in val_songs for f in songs_to_files[s]]
    test_files = [f for s in test_songs for f in songs_to_files[s]]

    print(f"\n   Split summary:")
    print(f"     Train: {len(train_songs)} songs, {len(train_files)} recordings")
    print(f"     Val:   {len(val_songs)} songs, {len(val_files)} recordings")
    print(f"     Test:  {len(test_songs)} songs, {len(test_files)} recordings")

    split_info = {
        'seed': seed,
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'guitarset_dir': str(guitarset_dir),
        'train_songs': train_songs,
        'val_songs': val_songs,
        'test_songs': test_songs,
        'train_files': [str(f) for f in train_files],
        'val_files': [str(f) for f in val_files],
        'test_files': [str(f) for f in test_files],
    }

    return train_files, val_files, test_files, split_info


# =============================================================================
# Argument Parser
# =============================================================================

DEFAULT_WARMUP_EPOCHS = 10
DEFAULT_EARLY_STOPPING = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified training script for Fretting-Transformer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Dataset selection
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["synthtab", "guitarset", "dadagp", "custom"],
        default="synthtab",
        help="Dataset to train on (default: synthtab)"
    )
    parser.add_argument(
        "--train-manifest",
        type=str,
        default=None,
        help="Training manifest file (required for custom dataset, optional override for others)"
    )
    parser.add_argument(
        "--val-manifest",
        type=str,
        default=None,
        help="Validation manifest file (required for custom dataset, optional override for others)"
    )
    parser.add_argument(
        "--guitarset-dir",
        type=str,
        default=None,
        help="GuitarSet annotation directory (overrides preset)"
    )
    parser.add_argument(
        "--guitarset-inverted-strings",
        action="store_true",
        help="Use inverted string numbering for GuitarSet (String 1 = Low E)"
    )

    # Conditioning
    cond_group = parser.add_mutually_exclusive_group()
    cond_group.add_argument(
        "--conditioning",
        action="store_true",
        dest="enable_conditioning",
        help="Enable capo/tuning conditioning augmentation"
    )
    cond_group.add_argument(
        "--no-conditioning",
        action="store_false",
        dest="enable_conditioning",
        help="Disable capo/tuning conditioning (no augmentation)"
    )
    parser.set_defaults(enable_conditioning=None)  # Will use dataset default

    parser.add_argument(
        "--capo-range",
        type=int,
        default=8,
        help="Number of capo positions for augmentation (0 to N-1, default: 8)"
    )

    # Model source
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        default=None,
        help="Path to .pt checkpoint to finetune from"
    )
    parser.add_argument(
        "--use-t5-pretrained",
        action="store_true",
        help="Initialize from HuggingFace T5 checkpoint"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="t5-small",
        help="HuggingFace model ID when using --use-t5-pretrained (default: t5-small)"
    )

    # LoRA
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Use LoRA for parameter-efficient finetuning"
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (default: 16)"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha scaling factor (default: 32)"
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.1,
        help="LoRA dropout (default: 0.1)"
    )

    # Training hyperparameters
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save checkpoints (default: checkpoints_{dataset})"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4, use 5e-5 for finetuning)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (default: dataset-specific)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Per-device training batch size (default: dataset-specific)"
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Per-device evaluation batch size (default: dataset-specific)"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps (default: 4)"
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=DEFAULT_WARMUP_EPOCHS,
        help="Number of warmup epochs (default: 10)"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Override warmup steps directly (takes precedence over --warmup-epochs)"
    )
    parser.add_argument(
        "--eval-delay",
        type=int,
        default=5,
        help="Epochs to wait before first evaluation (default: 5)"
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=DEFAULT_EARLY_STOPPING,
        help="Early stopping patience, 0 to disable (default: 15)"
    )

    # Resume training
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to HuggingFace checkpoint directory to resume training"
    )

    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    return parser.parse_args()


# =============================================================================
# Main Training Function
# =============================================================================

def main(args: argparse.Namespace) -> None:
    """Run unified training."""

    # Get dataset preset
    preset = DATASET_PRESETS[args.dataset]

    # Resolve conditioning setting
    if args.enable_conditioning is None:
        enable_conditioning = preset["default_conditioning"]
    else:
        enable_conditioning = args.enable_conditioning

    # Resolve other defaults from preset
    epochs = args.epochs or preset["default_epochs"]
    batch_size = args.batch_size or preset["default_batch_size"]
    eval_batch_size = args.eval_batch_size or preset["default_eval_batch_size"]
    output_dir = args.output_dir or f"checkpoints_{args.dataset}"

    # Print header
    cond_str = "WITH" if enable_conditioning else "WITHOUT"
    source_str = "finetuning" if args.pretrained_checkpoint else "scratch"
    if args.use_t5_pretrained:
        source_str = f"pretrained ({args.model_name})"

    print("=" * 70)
    print(f"Fretting-Transformer Training")
    print(f"Dataset: {args.dataset} - {preset['description']}")
    print(f"Mode: {cond_str} conditioning, from {source_str}")
    if args.use_lora:
        print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    print("=" * 70)

    # Load tokenizer
    print("\nLoading universal tokenizer...")
    tokenizer = MidiTabTokenizerV3.load("universal_tokenizer")
    print(f"   Loaded tokenizer: {len(tokenizer.shared_token_to_id)} vocab size")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Build DataConfig
    if enable_conditioning:
        data_config = DataConfig(
            max_encoder_length=512,
            max_decoder_length=512,
            train_on_time_shift=True,
            tab_loss_weight=1.2,
            enable_conditioning=True,
            conditioning_capo_values_train=tuple(range(args.capo_range)),
            conditioning_capo_values_eval=(0,),
            conditioning_tunings_train=DEFAULT_CONDITIONING_TUNINGS,
            conditioning_tunings_eval=(STANDARD_TUNING,),
            randomize_tuning_per_sequence=True,
            augmentation_seed=args.seed,
        )
    else:
        data_config = DataConfig(
            max_encoder_length=512,
            max_decoder_length=512,
            train_on_time_shift=True,
            tab_loss_weight=1.2,
            enable_conditioning=False,
        )

    print(f"\nData configuration:")
    print(f"   enable_conditioning: {data_config.enable_conditioning}")
    if data_config.enable_conditioning:
        print(f"   capo_values_train: {data_config.conditioning_capo_values_train}")
        print(f"   tunings_train: {len(data_config.conditioning_tunings_train)} variants")

    # Load datasets based on type
    print("\nLoading datasets...")

    if args.dataset == "guitarset":
        # GuitarSet-specific loading
        if not args.guitarset_dir:
            raise ValueError(
                "GuitarSet dataset requires --guitarset-dir argument. "
                "Please provide the path to your GuitarSet annotation directory."
            )
        guitarset_dir = Path(args.guitarset_dir)
        if not guitarset_dir.exists():
            raise ValueError(f"GuitarSet directory not found: {guitarset_dir}")

        print(f"   Creating song-level splits from {guitarset_dir}...")
        train_files, val_files, test_files, split_info = create_guitarset_splits(
            guitarset_dir,
            seed=args.seed,
        )

        # Inverted string convention support (String 1 = Low E)
        inverted_strings = bool(args.guitarset_inverted_strings)
        if inverted_strings:
            print("   Using inverted string convention for GuitarSet (String 1 = Low E)")

        # Save split info
        split_file = output_path / "split_info.json"
        with open(split_file, 'w') as f:
            json.dump(split_info, f, indent=2)
        print(f"   Saved splits to {split_file}")

        train_dataset = GuitarSetDataset(
            tokenizer=tokenizer,
            file_paths=train_files,
            data_config=data_config,
            split_name="train",
            inverted_strings=inverted_strings,
        )

        val_dataset = GuitarSetDataset(
            tokenizer=tokenizer,
            file_paths=val_files,
            data_config=data_config,
            split_name="val",
            inverted_strings=inverted_strings,
        )

    else:
        # Manifest-based loading (SynthTab, DadaGP, Custom)
        train_manifest = args.train_manifest or preset.get("train_manifest")
        val_manifest = args.val_manifest or preset.get("val_manifest")

        if not train_manifest or not val_manifest:
            raise ValueError(
                f"--train-manifest and --val-manifest are required for {args.dataset} dataset"
            )

        train_manifest = Path(train_manifest)
        val_manifest = Path(val_manifest)

        if not train_manifest.exists():
            raise ValueError(f"Training manifest not found: {train_manifest}")
        if not val_manifest.exists():
            raise ValueError(f"Validation manifest not found: {val_manifest}")

        print(f"   Loading training data from: {train_manifest}")
        train_dataset = SynthTabTokenDataset(
            tokenizer=tokenizer,
            manifests=[train_manifest],
            data_config=data_config,
            split="train",
            preload=True,
        )

        print(f"   Loading validation data from: {val_manifest}")
        val_dataset = SynthTabTokenDataset(
            tokenizer=tokenizer,
            manifests=[val_manifest],
            data_config=data_config,
            split="val",
            preload=True,
        )

    print(f"\n   Train: {len(train_dataset)} examples")
    print(f"   Val:   {len(val_dataset)} examples")

    # Calculate warmup steps
    steps_per_epoch = max(1, len(train_dataset) // (batch_size * args.gradient_accumulation_steps))
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else args.warmup_epochs * steps_per_epoch

    # Model configuration
    if args.use_t5_pretrained:
        model_config = ModelConfig(
            use_pretrained=True,
            model_name_or_path=args.model_name,
        )
    else:
        model_config = ModelConfig(
            use_pretrained=False,
            d_model=128,
            d_ff=1024,
            num_layers=3,
            num_heads=4,
            dropout_rate=0.1,
            layer_norm_epsilon=1e-6,
            relative_attention_num_buckets=32,
        )

    print(f"\nModel configuration:")
    if args.use_t5_pretrained:
        print(f"   Base: {args.model_name}")
    else:
        print(f"   Architecture: tiny (d_model=128, layers=3)")
    if args.pretrained_checkpoint:
        print(f"   Finetuning from: {args.pretrained_checkpoint}")
    if args.use_lora:
        print(f"   LoRA: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")

    # Training configuration
    training_config = TrainingConfig(
        output_dir=output_dir,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        num_train_epochs=epochs,
        logging_steps=100,
        save_total_limit=5,
        label_smoothing_factor=0.1,
        gradient_clip=1.0,
        bf16=False,
        fp16=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        generation_max_length=512,
        generation_num_beams=1,
        predict_with_generate=True,
        gradient_checkpointing=True,
        lr_scheduler_type="linear",
        use_constrained_generation=True,
        eval_with_constraints=True,
        early_stopping_patience=args.early_stopping_patience,
        eval_delay=args.eval_delay,
    )

    print(f"\nTraining configuration:")
    print(f"   learning_rate: {training_config.learning_rate}")
    print(f"   batch_size: {training_config.batch_size}")
    print(f"   effective_batch_size: {batch_size * args.gradient_accumulation_steps}")
    print(f"   epochs: {training_config.num_train_epochs}")
    print(f"   warmup_steps: {training_config.warmup_steps}")
    print(f"   early_stopping: {training_config.early_stopping_patience}")

    # Load or create model
    model = None
    if args.pretrained_checkpoint or args.use_lora:
        hf_tokenizer = build_hf_tokenizer(tokenizer.shared_token_to_id)
        model = create_model(tokenizer, model_config, hf_tokenizer=hf_tokenizer)

        if args.pretrained_checkpoint:
            print(f"\nLoading pretrained checkpoint: {args.pretrained_checkpoint}")
            checkpoint = torch.load(args.pretrained_checkpoint, map_location='cpu', weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']

                checkpoint_vocab_size = state_dict['shared.weight'].shape[0]
                current_vocab_size = model.get_input_embeddings().weight.shape[0]
                if checkpoint_vocab_size != current_vocab_size:
                    print(f"   Vocab size mismatch: checkpoint={checkpoint_vocab_size}, current={current_vocab_size}")
                    print(f"   Resizing model embeddings to {checkpoint_vocab_size}")
                    model.resize_token_embeddings(checkpoint_vocab_size)

                model.load_state_dict(state_dict, strict=False)
                print("   Loaded pretrained weights")
                model.train()
                for param in model.parameters():
                    param.requires_grad = True
            else:
                print("   Warning: Could not load pretrained weights")

        # Apply LoRA
        if args.use_lora:
            print(f"\nApplying LoRA adapters...")
            try:
                from peft import LoraConfig, TaskType, get_peft_model

                if training_config.gradient_checkpointing:
                    model.gradient_checkpointing_enable()
                    model.config.use_cache = False

                peft_config = LoraConfig(
                    task_type=TaskType.SEQ_2_SEQ_LM,
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    target_modules=["q", "k", "v", "o", "wi", "wo"],
                )
                model = get_peft_model(model, peft_config)
                model.enable_input_require_grads()

                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in model.parameters())
                print(f"   Trainable: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
            except ImportError:
                print("   Error: peft library not installed. Run: pip install peft")
                raise

    # Train!
    print("\nStarting training...")
    print("   Evaluation uses autoregressive generation (slower but accurate)")

    # Save tokenizer to output directory
    if args.use_lora:
        hf_tokenizer = build_hf_tokenizer(tokenizer.shared_token_to_id)
        hf_tokenizer.save_pretrained(output_path)

    trainer = train_model(
        tokenizer=tokenizer,
        model_config=model_config,
        training_config=training_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        resume_from_checkpoint=args.resume_from_checkpoint,
        model=model,
    )

    # Save final model
    final_model_path = output_path / "best_model.pt"

    save_dict = {
        'tokenizer_config': 'universal_tokenizer',
        'model_config': model_config,
        'training_config': training_config,
        'data_config': {
            'enable_conditioning': enable_conditioning,
            'capo_range': args.capo_range if enable_conditioning else 0,
        },
        'dataset': args.dataset,
        'finetuned_from': args.pretrained_checkpoint,
    }

    if args.use_lora:
        # Save LoRA adapters
        adapter_path = output_path / "lora_adapters"
        trainer.model.save_pretrained(adapter_path)
        print(f"\n   LoRA adapters saved to: {adapter_path}")

        # Try to merge and save full model
        try:
            merged_model = trainer.model.merge_and_unload()
            save_dict['model_state_dict'] = merged_model.state_dict()
            save_dict['lora_config'] = {
                'r': args.lora_r,
                'alpha': args.lora_alpha,
                'dropout': args.lora_dropout,
            }
        except Exception as e:
            print(f"   Warning: Could not merge LoRA weights: {e}")
            save_dict['model_state_dict'] = trainer.model.state_dict()
            save_dict['is_peft_model'] = True
    else:
        save_dict['model_state_dict'] = trainer.model.state_dict()

    torch.save(save_dict, final_model_path)

    print(f"\nTraining complete!")
    print(f"   Final model: {final_model_path}")
    print(f"   Checkpoints: {output_dir}")


if __name__ == "__main__":
    main(parse_args())
