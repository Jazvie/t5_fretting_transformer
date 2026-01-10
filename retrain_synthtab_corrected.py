#!/usr/bin/env python3
"""
Retrain SynthTab Model with Corrected Evaluation

This retrains the model using the fixed training.py module
that properly evaluates using autoregressive generation instead of teacher forcing.
By default it matches the paper's tiny-from-scratch specification, but a pretrained
T5 checkpoint can be enabled via a command-line flag.

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
)
from fret_t5.data import SynthTabTokenDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain SynthTab with corrected evaluation")
    parser.add_argument(
        "--use-pretrained",
        action="store_true",
        help="Load a pretrained checkpoint (default: t5-small) instead of the paper's tiny scratch configuration.",
    )
    parser.add_argument(
        "--model-name-or-path",
        default="t5-small",
        help="Model identifier or local path used when --use-pretrained is supplied.",
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints_corrected",
        help="Directory to save checkpoints and final model.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Early stopping patience (0 = disabled). Use 10+ for scratch training.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from (e.g., checkpoints_scratch_conditioning/checkpoint-283322)",
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
        help="Per-device evaluation batch size (default: 8, smaller due to autoregressive generation)",
    )
    parser.add_argument(
        "--eval-delay",
        type=int,
        default=5,
        help="Number of epochs to wait before first evaluation (default: 5)",
    )
    parser.add_argument(
        "--train-manifest",
        default="data/synthtab_acoustic_train.jsonl",
        help="Training manifest file (default: data/synthtab_acoustic_train.jsonl)",
    )
    parser.add_argument(
        "--val-manifest",
        default="data/synthtab_acoustic_val.jsonl",
        help="Validation manifest file (default: data/synthtab_acoustic_val.jsonl)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=120,
        help="Number of training epochs (default: 120)",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Retrain SynthTab model with corrected evaluation."""

    print("SynthTab Retraining with Corrected Autoregressive Evaluation")
    print("=" * 70)
    print("Default mode trains the tiny scratch model with proper evaluation")
    print("Validation will be MUCH slower but metrics will be realistic")
    print("=" * 70)

    # Load tokenizer
    # Load tokenizer
    print("\nLoading universal tokenizer...")
    tokenizer = MidiTabTokenizerV3.load("universal_tokenizer")
    vocab_size = len(tokenizer.shared_token_to_id)
    print(f"   Loaded tokenizer: {vocab_size} vocab size")

    # Data configuration
    print("\nLoading pre-made train/val splits...")
    data_config = DataConfig(
        max_encoder_length=512,
        max_decoder_length=512,
        train_on_time_shift=True,
        tab_loss_weight=1.2,
        enable_conditioning=True,
        conditioning_capo_values_train=tuple(range(8)),
        conditioning_capo_values_eval=(0,),
        conditioning_tunings_train=DEFAULT_CONDITIONING_TUNINGS,
        conditioning_tunings_eval=(DEFAULT_CONDITIONING_TUNINGS[0],),
        randomize_tuning_per_sequence=True,
    )

    # Load train dataset from pre-made split
    print("   Loading training set...")
    train_dataset = SynthTabTokenDataset(
        tokenizer=tokenizer,
        manifests=[Path(args.train_manifest)],
        data_config=data_config,
        split="train",
        preload=True
    )

    # Load validation dataset from pre-made split
    print("   Loading validation set...")
    val_dataset = SynthTabTokenDataset(
        tokenizer=tokenizer,
        manifests=[Path(args.val_manifest)],
        data_config=data_config,
        split="val",
        preload=True
    )

    print(f"   Train: {len(train_dataset)} examples")
    print(f"   Val:   {len(val_dataset)} examples")

    # Model configuration (pretrained vs paper tiny)
    if args.use_pretrained:
        print("\nUsing pretrained T5 checkpoint")
        model_config = ModelConfig(
            use_pretrained=True,
            model_name_or_path=args.model_name_or_path,
        )
    else:
        print("\nUsing paper-spec tiny configuration")
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

    print(f"\nModel Configuration:")
    if args.use_pretrained:
        print(f"   model_name_or_path: {model_config.model_name_or_path}")
    else:
        print(f"   d_model: {model_config.d_model}")
        print(f"   d_ff: {model_config.d_ff}")
        print(f"   num_layers: {model_config.num_layers}")
        print(f"   num_heads: {model_config.num_heads}")

    # Training configuration with CORRECTED evaluation
    training_config = TrainingConfig(
        output_dir=args.output_dir,
        learning_rate=1e-4,
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
        eval_delay=args.eval_delay,       # Skip first N epochs of evaluation
    )

    print(f"\nTraining Configuration:")
    print(f"   Learning rate: {training_config.learning_rate}")
    print(f"   Batch size: {training_config.batch_size}")
    print(f"   Eval batch size: {training_config.eval_batch_size}")
    print(f"   Epochs: {training_config.num_train_epochs}")
    print(f"   Warmup steps: {training_config.warmup_steps}")
    print(f"   Early stopping patience: {training_config.early_stopping_patience}")
    print(f"   Early stopping patience: {training_config.early_stopping_patience}")
    print(f"   predict_with_generate: {training_config.predict_with_generate}")
    print(f"   eval_with_constraints: {training_config.eval_with_constraints}")
    print(f"   enable_conditioning: {data_config.enable_conditioning}")

    # Train!
    if args.resume_from_checkpoint:
        print(f"\nResuming training from checkpoint: {args.resume_from_checkpoint}")
    else:
        print("\nStarting training with corrected evaluation...")
    print("   NOTE: Evaluation will be much slower (autoregressive generation)")
    print("   NOTE: Initial metrics may be lower but will be realistic")
    print("   NOTE: Model will learn to actually generate sequences\n")

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
    }, final_model_path)

    print(f"\nTraining complete!")
    print(f"   Final model: {final_model_path}")
    print(f"   Checkpoints: {training_config.output_dir}")
    print(f"   This model was trained with autoregressive evaluation")
    print(f"   Metrics should match real-world performance")


if __name__ == "__main__":
    main(parse_args())
