#!/usr/bin/env python3
"""
Unified GuitarSet Training Script

Supports:
- Training from scratch or finetuning from pretrained checkpoint
- With or without capo/tuning conditioning
- LoRA for parameter-efficient finetuning
- Proper song-level splits to prevent data leakage
- Autoregressive evaluation with constrained decoding

Usage examples:
    # Train from scratch WITHOUT conditioning (baseline)
    python train_guitarset.py --output-dir checkpoints_guitarset_baseline

    # Train from scratch WITH conditioning (augmentation)
    python train_guitarset.py --enable-conditioning --output-dir checkpoints_guitarset_cond

    # Finetune from SynthTab model WITH conditioning (full finetuning)
    python train_guitarset.py --enable-conditioning \\
        --pretrained-checkpoint checkpoints_conditioning_scratch_retrain/best_model.pt \\
        --learning-rate 5e-5 --epochs 50 \\
        --output-dir checkpoints_guitarset_finetuned

    # Finetune with LoRA (recommended - preserves SynthTab knowledge)
    python train_guitarset.py --enable-conditioning \\
        --pretrained-checkpoint checkpoints_conditioning_scratch_retrain/best_model.pt \\
        --use-lora --lora-r 16 --lora-alpha 32 \\
        --learning-rate 1e-4 --epochs 50 \\
        --output-dir checkpoints_guitarset_lora

Key configuration:
- Song-level 70/15/15 train/val/test splits
- predict_with_generate=True (autoregressive evaluation)
- eval_with_constraints=True (constrained decoding)

LoRA benefits for GuitarSet:
- GuitarSet is small (~100 training files) - LoRA prevents overfitting
- Preserves SynthTab knowledge while adapting to real recordings
- Only trains ~1-2% of parameters
- Faster training, lower memory usage
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
)
from fret_t5.data import chunk_tokenized_track
from fret_t5.tokenization import NoteMetadata
from fret_t5.hf_tokenizer import build_hf_tokenizer

from scripts.guitarset_loader import (
    load_guitarset_jams,
    extract_tablature_from_guitarset_jams,
)


# Default split ratios
DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train on GuitarSet (with or without conditioning)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Mode selection
    parser.add_argument(
        "--enable-conditioning",
        action="store_true",
        help="Enable capo/tuning conditioning augmentation (like SynthTab training)",
    )

    # Model source
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        default=None,
        help="Path to .pt checkpoint to finetune from (e.g., best_model.pt from SynthTab)",
    )
    parser.add_argument(
        "--use-t5-pretrained",
        action="store_true",
        help="Initialize from HuggingFace T5 checkpoint instead of scratch",
    )
    parser.add_argument(
        "--model-name-or-path",
        default="t5-small",
        help="HuggingFace model ID when using --use-t5-pretrained",
    )

    # LoRA options
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Use LoRA for parameter-efficient finetuning (recommended with --pretrained-checkpoint)",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (default: 16)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha scaling factor (default: 32)",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.1,
        help="LoRA dropout (default: 0.1)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        default="checkpoints_guitarset",
        help="Directory to save checkpoints and final model",
    )

    # Training hyperparameters
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (use 1e-4 for scratch, 5e-5 for finetuning)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Maximum number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Per-device training batch size",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=8,
        help="Per-device evaluation batch size",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=10,
        help="Number of warmup epochs",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=15,
        help="Early stopping patience (0 = disabled)",
    )
    parser.add_argument(
        "--eval-delay",
        type=int,
        default=5,
        help="Number of epochs to wait before first evaluation",
    )

    # Conditioning options
    parser.add_argument(
        "--capo-range",
        type=int,
        default=8,
        help="Number of capo positions for training augmentation (0 to N-1)",
    )

    # Data options
    parser.add_argument(
        "--guitarset-dir",
        type=str,
        default="/data/akshaj/MusicAI/GuitarSet/annotation",
        help="Path to GuitarSet annotation directory",
    )
    parser.add_argument(
        "--split-file",
        type=str,
        default=None,
        help="Path to existing split file (skips creating new splits)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducible splits",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help="Train split ratio",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_TEST_RATIO,
        help="Test split ratio",
    )

    # Resume training
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to HuggingFace checkpoint directory to resume training",
    )

    return parser.parse_args()


def create_guitarset_splits(
    guitarset_dir: Path,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = DEFAULT_SEED,
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

    # Split songs (not files)
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


def load_splits_from_file(split_file: Path) -> Tuple[List[Path], List[Path], List[Path], Dict]:
    """Load existing splits from a JSON file."""
    with open(split_file, 'r') as f:
        split_info = json.load(f)

    train_files = [Path(p) for p in split_info['train_files']]
    val_files = [Path(p) for p in split_info['val_files']]
    test_files = [Path(p) for p in split_info['test_files']]

    print(f"\n   Loaded splits from {split_file}")
    print(f"     Train: {len(split_info['train_songs'])} songs, {len(train_files)} files")
    print(f"     Val:   {len(split_info['val_songs'])} songs, {len(val_files)} files")
    print(f"     Test:  {len(split_info['test_songs'])} songs, {len(test_files)} files")

    return train_files, val_files, test_files, split_info


class GuitarSetDataset(torch.utils.data.Dataset):
    """Dataset for GuitarSet with optional conditioning support.

    This dataset properly handles:
    - Note-group-aligned chunking (not raw token chunking)
    - Conditioning augmentation when enabled
    - Correct pitch transformation for different capo/tuning
    """

    def __init__(
        self,
        tokenizer: MidiTabTokenizerV3,
        file_paths: List[Path],
        data_config: DataConfig,
        split_name: str = "train",
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.data_config = data_config
        self.split_name = split_name
        self.examples: List[Dict[str, torch.Tensor]] = []
        self._rng = random.Random(data_config.augmentation_seed)

        self._preload_examples(file_paths)

    def _preload_examples(self, file_paths: List[Path]) -> None:
        """Load and tokenize all GuitarSet files."""
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

            # Register conditioning tokens
            self.tokenizer.ensure_conditioning_tokens(capo_values, tuning_options)

            # Reserve space for conditioning prefix
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
                    auto_detect_tuning=False  # GuitarSet is standard tuning
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

                # Tokenize (creates aligned encoder/decoder/metadata)
                tokenized = self.tokenizer.tokenize_track_from_jams(jams_events)

                # Use proper note-group-aligned chunking
                chunks = list(chunk_tokenized_track(tokenized, chunk_config))

                if not self.data_config.enable_conditioning:
                    # No conditioning - add chunks directly
                    for enc_tokens, dec_tokens, _ in chunks:
                        self._append_example(list(enc_tokens), list(dec_tokens))
                else:
                    # With conditioning - create augmented examples
                    for enc_tokens, dec_tokens, note_metadata in chunks:
                        for capo in capo_values:
                            if is_training and self.data_config.randomize_tuning_per_sequence:
                                tuning_choices = [self._rng.choice(tuning_options)]
                            else:
                                tuning_choices = list(tuning_options)

                            for tuning in tuning_choices:
                                # Apply conditioning transformation
                                conditioned_encoder = self._apply_conditioning(
                                    list(enc_tokens),
                                    list(note_metadata),
                                    capo,
                                    tuning,
                                )

                                # Add conditioning prefix
                                prefix_tokens = self.tokenizer.build_conditioning_prefix(capo, tuning)
                                final_encoder = prefix_tokens + conditioned_encoder

                                self._append_example(final_encoder, list(dec_tokens))

            except Exception:
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
        """Apply pitch transformation for conditioning.

        Modifies NOTE_ON/NOTE_OFF tokens to reflect the pitch that would
        result from playing with the given capo and tuning.

        Formula: pitch = tuning[string-1] + fret + capo
        """
        conditioned_tokens = list(encoder_tokens)
        if not note_metadata:
            return conditioned_tokens

        tuning_values = [int(v) for v in tuning]
        pointer = 0

        for note in note_metadata:
            # Find next NOTE_ON token
            while pointer < len(conditioned_tokens) and not conditioned_tokens[pointer].startswith("NOTE_ON<"):
                pointer += 1

            if pointer >= len(conditioned_tokens):
                break

            # Calculate pitch: open_string + fret + capo
            base_pitch = tuning_values[note.string - 1] + note.fret + int(capo)
            base_pitch = max(0, min(127, base_pitch))
            conditioned_tokens[pointer] = f"NOTE_ON<{base_pitch}>"

            # Find and update NOTE_OFF
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
        try:
            input_ids = self.tokenizer.encode_encoder_tokens_shared(enc_tokens)
            labels = self.tokenizer.encode_decoder_tokens_shared(dec_tokens)

            input_ids = input_ids[:self.data_config.max_encoder_length]
            labels = labels[:self.data_config.max_decoder_length]

            attention_mask = [1] * len(input_ids)

            pad_id = self.tokenizer.shared_token_to_id.get("<pad>", 0)
            while len(input_ids) < self.data_config.max_encoder_length:
                input_ids.append(pad_id)
                attention_mask.append(0)

            # Loss mask
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
        except Exception:
            pass

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


def main(args: argparse.Namespace) -> None:
    """Train GuitarSet model."""

    mode_str = "WITH" if args.enable_conditioning else "WITHOUT"
    source_str = "finetuning" if args.pretrained_checkpoint else "scratch"

    print("=" * 70)
    print(f"GuitarSet Training {mode_str} Conditioning ({source_str})")
    print("=" * 70)

    # Load tokenizer
    print("\nLoading universal tokenizer...")
    tokenizer = MidiTabTokenizerV3.load("universal_tokenizer")
    print(f"   Loaded tokenizer: {len(tokenizer.shared_token_to_id)} vocab size")

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create or load splits
    guitarset_dir = Path(args.guitarset_dir)
    if not guitarset_dir.exists():
        raise ValueError(f"GuitarSet directory not found: {guitarset_dir}")

    if args.split_file:
        train_files, val_files, test_files, split_info = load_splits_from_file(Path(args.split_file))
    else:
        print("\nCreating song-level splits...")
        train_files, val_files, test_files, split_info = create_guitarset_splits(
            guitarset_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        # Save split info
        split_file = output_path / "split_info.json"
        with open(split_file, 'w') as f:
            json.dump(split_info, f, indent=2)
        print(f"   Saved splits to {split_file}")

    # Data configuration
    if args.enable_conditioning:
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
            augmentation_seed=1337,
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

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = GuitarSetDataset(
        tokenizer=tokenizer,
        file_paths=train_files,
        data_config=data_config,
        split_name="train"
    )

    val_dataset = GuitarSetDataset(
        tokenizer=tokenizer,
        file_paths=val_files,
        data_config=data_config,
        split_name="val"
    )

    print(f"\n   Train: {len(train_dataset)} examples")
    print(f"   Val:   {len(val_dataset)} examples")

    # Calculate warmup steps
    steps_per_epoch = max(1, len(train_dataset) // (args.batch_size * args.gradient_accumulation_steps))
    warmup_steps = args.warmup_epochs * steps_per_epoch

    # Model configuration
    # Note: LoRA is applied AFTER loading pretrained weights, so we don't set use_lora here
    if args.use_t5_pretrained:
        model_config = ModelConfig(
            use_pretrained=True,
            model_name_or_path=args.model_name_or_path,
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
        print(f"   Base: {args.model_name_or_path}")
    else:
        print(f"   Architecture: tiny (d_model=128, layers=3)")
    if args.pretrained_checkpoint:
        print(f"   Finetuning from: {args.pretrained_checkpoint}")
    if args.use_lora:
        print(f"   LoRA: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")

    # Training configuration
    training_config = TrainingConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        num_train_epochs=args.epochs,
        logging_steps=50,
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
    print(f"   effective_batch_size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"   epochs: {training_config.num_train_epochs}")
    print(f"   warmup_steps: {training_config.warmup_steps}")
    print(f"   early_stopping: {training_config.early_stopping_patience}")

    # Load or create model
    model = None
    if args.pretrained_checkpoint or args.use_lora:
        # For finetuning or LoRA, we create the model manually
        hf_tokenizer = build_hf_tokenizer(tokenizer.shared_token_to_id)
        model = create_model(tokenizer, model_config, hf_tokenizer=hf_tokenizer)

        # Load pretrained weights if provided
        if args.pretrained_checkpoint:
            print(f"\nLoading pretrained checkpoint: {args.pretrained_checkpoint}")
            checkpoint = torch.load(args.pretrained_checkpoint, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                model.load_state_dict(state_dict, strict=False)
                print("   Loaded pretrained weights")
            else:
                print("   Warning: Could not load pretrained weights")

        # Apply LoRA AFTER loading pretrained weights
        if args.use_lora:
            print(f"\nApplying LoRA adapters...")
            try:
                from peft import LoraConfig, TaskType, get_peft_model

                peft_config = LoraConfig(
                    task_type=TaskType.SEQ_2_SEQ_LM,
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    target_modules=["q", "k", "v", "o", "wi", "wo"],
                )
                model = get_peft_model(model, peft_config)

                # Print trainable parameters
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in model.parameters())
                print(f"   Trainable: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
            except ImportError:
                print("   Error: peft library not installed. Run: pip install peft")
                raise

    # Train!
    print("\nStarting training...")
    print("   Evaluation uses autoregressive generation (slower but accurate)")

    trainer = train_model(
        tokenizer=tokenizer,
        model_config=model_config,
        training_config=training_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        resume_from_checkpoint=args.resume_from_checkpoint,
        model=model,  # Pass pre-loaded model if finetuning
    )

    # Save final model
    final_model_path = output_path / "best_model.pt"

    if args.use_lora:
        # For LoRA, save both adapter-only and merged versions
        print("\nSaving LoRA model...")

        # Save adapter weights only (small file)
        adapter_path = output_path / "lora_adapters"
        trainer.model.save_pretrained(adapter_path)
        print(f"   LoRA adapters: {adapter_path}")

        # Merge and save full model (for inference without PEFT)
        try:
            merged_model = trainer.model.merge_and_unload()
            torch.save({
                'model_state_dict': merged_model.state_dict(),
                'tokenizer_config': 'universal_tokenizer',
                'model_config': model_config,
                'training_config': training_config,
                'data_config': data_config,
                'split_info': split_info,
                'conditioning_enabled': args.enable_conditioning,
                'finetuned_from': args.pretrained_checkpoint,
                'lora_config': {
                    'r': args.lora_r,
                    'alpha': args.lora_alpha,
                    'dropout': args.lora_dropout,
                },
            }, final_model_path)
            print(f"   Merged model: {final_model_path}")
        except Exception as e:
            print(f"   Warning: Could not merge LoRA weights: {e}")
            # Fall back to saving PEFT model state
            torch.save({
                'model_state_dict': trainer.model.state_dict(),
                'tokenizer_config': 'universal_tokenizer',
                'model_config': model_config,
                'training_config': training_config,
                'data_config': data_config,
                'split_info': split_info,
                'conditioning_enabled': args.enable_conditioning,
                'finetuned_from': args.pretrained_checkpoint,
                'is_peft_model': True,
            }, final_model_path)
    else:
        torch.save({
            'model_state_dict': trainer.model.state_dict(),
            'tokenizer_config': 'universal_tokenizer',
            'model_config': model_config,
            'training_config': training_config,
            'data_config': data_config,
            'split_info': split_info,
            'conditioning_enabled': args.enable_conditioning,
            'finetuned_from': args.pretrained_checkpoint,
        }, final_model_path)

    print(f"\nTraining complete!")
    print(f"   Model: {final_model_path}")
    print(f"   Checkpoints: {args.output_dir}")
    print(f"   Split info: {output_path / 'split_info.json'}")


if __name__ == "__main__":
    main(parse_args())
