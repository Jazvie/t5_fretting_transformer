#!/usr/bin/env python3
"""
Train on DadaGP acoustic guitar data with capo augmentation.

This script trains the fret-T5 model on DadaGP data using the same pipeline
as SynthTab, ensuring consistent TIME_SHIFT tokenization across datasets.

Usage:
    python train_dadagp.py                          # Train on DadaGP only
    python train_dadagp.py --use-pretrained         # Use pretrained T5
    python train_dadagp.py --no-conditioning        # Disable capo augmentation
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from fret_t5 import (
    MidiTabTokenizerV3,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    train_model,
    DEFAULT_CONDITIONING_TUNINGS,
    STANDARD_TUNING,
)
from fret_t5.data import SynthTabTokenDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train fret-T5 on DadaGP acoustic guitar data"
    )

    # Model options
    parser.add_argument(
        "--use-pretrained",
        action="store_true",
        help="Use pretrained T5 checkpoint instead of training from scratch",
    )
    parser.add_argument(
        "--model-name-or-path",
        default="t5-small",
        help="Model identifier when using --use-pretrained (default: t5-small)",
    )

    # Data options
    parser.add_argument(
        "--train-manifest",
        default="data/dadagp_acoustic_train.jsonl",
        help="Training manifest file (default: data/dadagp_acoustic_train.jsonl)",
    )
    parser.add_argument(
        "--val-manifest",
        default="data/dadagp_acoustic_val.jsonl",
        help="Validation manifest file (default: data/dadagp_acoustic_val.jsonl)",
    )

    # Conditioning/augmentation options
    parser.add_argument(
        "--no-conditioning",
        action="store_true",
        help="Disable capo/tuning conditioning (no augmentation)",
    )
    parser.add_argument(
        "--max-capo",
        type=int,
        default=7,
        help="Maximum capo value for augmentation (default: 7, range 0-7)",
    )

    # Training options
    parser.add_argument(
        "--output-dir",
        default="checkpoints_dadagp",
        help="Directory to save checkpoints (default: checkpoints_dadagp)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Per-device training batch size (default: 16)",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=8,
        help="Per-device evaluation batch size (default: 8)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=120,
        help="Number of training epochs (default: 120)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--eval-delay",
        type=int,
        default=5,
        help="Epochs to wait before first evaluation (default: 5)",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Train on DadaGP data."""

    print("=" * 70)
    print("DadaGP Acoustic Guitar Training")
    print("=" * 70)

    # Verify manifest files exist
    train_manifest = Path(args.train_manifest)
    val_manifest = Path(args.val_manifest)

    if not train_manifest.exists():
        print(f"ERROR: Training manifest not found: {train_manifest}")
        print("Run the DadaGP pipeline first: python scripts/dadagp_pipeline.py")
        sys.exit(1)

    if not val_manifest.exists():
        print(f"ERROR: Validation manifest not found: {val_manifest}")
        sys.exit(1)

    # Load tokenizer
    print("\nLoading universal tokenizer...")
    tokenizer = MidiTabTokenizerV3.load("universal_tokenizer")
    vocab_size = len(tokenizer.shared_token_to_id)
    print(f"  Vocab size: {vocab_size}")

    # Data configuration
    enable_conditioning = not args.no_conditioning

    if enable_conditioning:
        print(f"\nConditioning ENABLED (capo 0-{args.max_capo}, tuning augmentation)")
        data_config = DataConfig(
            max_encoder_length=512,
            max_decoder_length=512,
            acoustic_programs=(24, 25, 26),  # 24=Nylon, 25=Steel, 26=legacy
            train_on_time_shift=True,
            tab_loss_weight=1.2,
            enable_conditioning=True,
            conditioning_capo_values_train=tuple(range(args.max_capo + 1)),
            conditioning_capo_values_eval=(0,),  # Eval at capo 0 only
            conditioning_tunings_train=DEFAULT_CONDITIONING_TUNINGS,
            conditioning_tunings_eval=(STANDARD_TUNING,),
            randomize_tuning_per_sequence=True,
        )
    else:
        print("\nConditioning DISABLED (no augmentation)")
        data_config = DataConfig(
            max_encoder_length=512,
            max_decoder_length=512,
            acoustic_programs=(24, 25, 26),
            train_on_time_shift=True,
            tab_loss_weight=1.2,
            enable_conditioning=False,
        )

    # Load datasets
    print(f"\nLoading training data from: {train_manifest}")
    train_dataset = SynthTabTokenDataset(
        tokenizer=tokenizer,
        manifests=[train_manifest],
        data_config=data_config,
        split="train",
        preload=True,
    )

    print(f"Loading validation data from: {val_manifest}")
    val_dataset = SynthTabTokenDataset(
        tokenizer=tokenizer,
        manifests=[val_manifest],
        data_config=data_config,
        split="val",
        preload=True,
    )

    print(f"\n  Train examples: {len(train_dataset)}")
    print(f"  Val examples:   {len(val_dataset)}")

    # Model configuration
    if args.use_pretrained:
        print(f"\nUsing pretrained model: {args.model_name_or_path}")
        model_config = ModelConfig(
            use_pretrained=True,
            model_name_or_path=args.model_name_or_path,
        )
    else:
        print("\nUsing paper-spec tiny configuration (from scratch)")
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

    # Training configuration
    training_config = TrainingConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=4000,
        num_train_epochs=args.epochs,
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
        eval_delay=args.eval_delay,
    )

    print(f"\nTraining Configuration:")
    print(f"  Output dir: {training_config.output_dir}")
    print(f"  Learning rate: {training_config.learning_rate}")
    print(f"  Batch size: {training_config.batch_size}")
    print(f"  Epochs: {training_config.num_train_epochs}")
    print(f"  Warmup steps: {training_config.warmup_steps}")
    print(f"  Eval delay: {args.eval_delay} epochs")

    # Start training
    if args.resume_from_checkpoint:
        print(f"\nResuming from: {args.resume_from_checkpoint}")
    else:
        print("\nStarting training...")

    print("  NOTE: Evaluation uses autoregressive generation (slower but accurate)")
    print()

    trainer = train_model(
        tokenizer=tokenizer,
        model_config=model_config,
        training_config=training_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    # Save final model
    import torch
    final_model_path = Path(args.output_dir) / "best_model.pt"
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'tokenizer_config': 'universal_tokenizer',
        'model_config': model_config,
        'training_config': training_config,
        'data_config': {
            'enable_conditioning': enable_conditioning,
            'max_capo': args.max_capo if enable_conditioning else 0,
        },
    }, final_model_path)

    print(f"\nTraining complete!")
    print(f"  Final model: {final_model_path}")
    print(f"  Checkpoints: {args.output_dir}")


if __name__ == "__main__":
    main(parse_args())
