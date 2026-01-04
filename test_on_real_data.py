#!/usr/bin/env python3
"""
Test checkpoint-614711 on actual training/validation data.

This script loads real data using the EXACT SAME configuration as training
and evaluates the model's predictions to sanity check if it trained well.

Key: This uses the EXACT data config and metrics from training.py!
"""

import sys
from pathlib import Path
import torch
import random
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from fret_t5.tokenization import MidiTabTokenizerV3, DEFAULT_CONDITIONING_TUNINGS
from fret_t5.data import DataConfig, create_song_level_splits
from fret_t5.metrics import compute_tab_accuracy
from transformers import T5ForConditionalGeneration


def load_model_and_tokenizer(checkpoint_path: str):
    """Load the trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    print("NOTE: This is a SCRATCH model (not pretrained)")

    # Load tokenizer with conditioning support (EXACT training config)
    tokenizer = MidiTabTokenizerV3.load("universal_tokenizer")
    tokenizer.ensure_conditioning_tokens(
        capo_values=tuple(range(8)),
        tuning_options=DEFAULT_CONDITIONING_TUNINGS
    )

    # Load model
    model = T5ForConditionalGeneration.from_pretrained(checkpoint_path)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        print(f"✓ Model loaded successfully (GPU)")
    else:
        print(f"✓ Model loaded successfully (CPU)")

    print(f"  - Vocab size: {model.config.vocab_size}")
    print(f"  - d_model: {model.config.d_model}")
    print(f"  - num_layers: {model.config.num_layers}")
    print(f"  - num_heads: {model.config.num_heads}")

    return model, tokenizer


def load_datasets(tokenizer):
    """Load the EXACT SAME train/val datasets used during training."""
    print("\nLoading datasets (EXACT training configuration)...")

    # EXACT data config from retrain_synthtab_corrected.py
    data_config = DataConfig(
        max_encoder_length=512,
        max_decoder_length=512,
        train_on_time_shift=True,  # Paper-faithful
        tab_loss_weight=1.2,       # Proven to work well
        enable_conditioning=True,
        conditioning_capo_values_train=tuple(range(8)),
        conditioning_capo_values_eval=(0,),
        conditioning_tunings_train=DEFAULT_CONDITIONING_TUNINGS,
        conditioning_tunings_eval=(DEFAULT_CONDITIONING_TUNINGS[0],),
        randomize_tuning_per_sequence=True,
    )

    # Create song-level splits (EXACT same as training)
    train_dataset, val_dataset, _ = create_song_level_splits(
        tokenizer=tokenizer,
        manifests=[Path("data/synthtab_acoustic_all.jsonl")],
        data_config=data_config,
        train_ratio=0.8,
        val_ratio=0.2,
        test_ratio=0.0
    )

    print(f"✓ Train dataset: {len(train_dataset)} examples")
    print(f"✓ Val dataset: {len(val_dataset)} examples")

    return train_dataset, val_dataset, tokenizer


def predict_batch(model, tokenizer, examples, max_length=512, pad_to_multiple_of=8):
    """Run model prediction on a batch of examples.

    This function replicates what TabSeq2SeqCollator does during training:
    1. Pads input_ids to max length in batch
    2. Generates predictions
    3. Pads BOTH predictions and labels to the same length
    4. Respects pad_to_multiple_of (training uses 8)

    This is necessary because compute_tab_accuracy expects predictions and labels
    to have the same shape for element-wise comparison.
    """
    # Prepare batch
    input_ids_list = [ex['input_ids'] for ex in examples]
    labels_list = [ex['labels'] for ex in examples]

    # Pad to max length in batch
    max_input_len = max(ids.shape[0] for ids in input_ids_list)

    input_ids_padded = []
    for ids in input_ids_list:
        if ids.shape[0] < max_input_len:
            padding = torch.zeros(max_input_len - ids.shape[0], dtype=torch.long)
            ids = torch.cat([ids, padding])
        input_ids_padded.append(ids)

    input_ids_batch = torch.stack(input_ids_padded)

    if torch.cuda.is_available():
        input_ids_batch = input_ids_batch.cuda()

    # Generate (same as training evaluation)
    with torch.no_grad():
        outputs = model.generate(
            input_ids_batch,
            max_length=max_length,
            num_beams=1,  # Same as training config
            do_sample=False,
            eos_token_id=model.config.eos_token_id,
            pad_token_id=model.config.pad_token_id,
        )

    outputs_cpu = outputs.cpu()

    # Pad predictions and labels to same length
    # This matches TabSeq2SeqCollator behavior (see collators.py lines 62-68)
    max_len = max(outputs_cpu.shape[1], max(lbl.shape[0] for lbl in labels_list))

    # Apply pad_to_multiple_of like the collator does (training uses pad_to_multiple_of=8)
    if pad_to_multiple_of is not None and max_len % pad_to_multiple_of != 0:
        max_len = ((max_len + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

    # Pad predictions to max_len (pad with 0s like decoder output padding)
    predictions_padded = []
    for pred in outputs_cpu:
        if pred.shape[0] < max_len:
            padding = torch.zeros(max_len - pred.shape[0], dtype=torch.long)
            pred = torch.cat([pred, padding])
        predictions_padded.append(pred)

    predictions_batch = torch.stack(predictions_padded)

    # Pad labels to max_len (pad with -100 like label_pad_token_id)
    labels_padded = []
    for lbl in labels_list:
        if lbl.shape[0] < max_len:
            padding = torch.full((max_len - lbl.shape[0],), -100, dtype=torch.long)
            lbl = torch.cat([lbl, padding])
        labels_padded.append(lbl)

    labels_batch = torch.stack(labels_padded)

    return predictions_batch.numpy(), labels_batch.numpy()


def evaluate_on_dataset(model, tokenizer, dataset, num_samples=50, dataset_name="Dataset"):
    """Evaluate model on samples from dataset using TRAINING METRICS."""
    print(f"\n{'='*80}")
    print(f"Evaluating on {num_samples} samples from {dataset_name}")
    print(f"Using EXACT SAME metrics as training (compute_tab_accuracy)")
    print(f"{'='*80}")

    # Randomly sample examples
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    examples = [dataset[i] for i in indices]

    # Predict in batches
    batch_size = 8
    all_predictions = []
    all_labels = []

    for i in range(0, len(examples), batch_size):
        batch = examples[i:i+batch_size]
        preds, labels = predict_batch(model, tokenizer, batch)
        all_predictions.append(preds)
        all_labels.append(labels)

    # Concatenate all predictions and labels
    predictions = np.concatenate(all_predictions, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # Compute metrics using EXACT training metric function
    metrics = compute_tab_accuracy(
        {"predictions": predictions, "label_ids": labels},
        tokenizer
    )

    # Print results
    print(f"\nMetrics on {len(examples)} examples:")
    print(f"  Tab Accuracy:        {metrics['tab_accuracy']:.2%}")
    print(f"  Pitch Accuracy:      {metrics['pitch_accuracy']:.2%}")
    print(f"  Time Shift Accuracy: {metrics['time_shift_accuracy']:.2%}")
    print(f"  Overall Accuracy:    {metrics['overall_accuracy']:.2%}")
    print(f"  Sequence Validity:   {metrics['sequence_validity']:.2%}")

    return metrics


def show_example_predictions(model, tokenizer, dataset, num_examples=3):
    """Show detailed predictions for a few examples."""
    print(f"\n{'='*80}")
    print(f"DETAILED EXAMPLE PREDICTIONS")
    print(f"{'='*80}")

    for i in range(min(num_examples, len(dataset))):
        example = dataset[i]
        input_ids = example['input_ids'].unsqueeze(0)
        labels = example['labels']

        print(f"\n--- Example {i+1} ---")

        # Decode input
        input_tokens = tokenizer.shared_to_encoder_tokens(input_ids[0].tolist())
        conditioning_tokens = [t for t in input_tokens if t.startswith('CAPO<') or t.startswith('TUNING<')]
        print(f"Conditioning: {conditioning_tokens}")

        # Predict
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=512,
                num_beams=1,
                do_sample=False,
                eos_token_id=model.config.eos_token_id,
                pad_token_id=model.config.pad_token_id,
            )

        # Decode predictions and labels
        pred_tokens = tokenizer.shared_to_decoder_tokens(outputs[0].cpu().tolist())
        label_tokens = tokenizer.shared_to_decoder_tokens(labels.tolist())

        # Extract TAB tokens
        pred_tabs = [t for t in pred_tokens if t.startswith('TAB<')][:10]
        label_tabs = [t for t in label_tokens if t.startswith('TAB<')][:10]

        print(f"First 10 predicted TABs: {pred_tabs}")
        print(f"First 10 expected TABs:  {label_tabs}")

        # Count matches
        matches = sum(1 for p, l in zip(pred_tabs, label_tabs) if p == l)
        print(f"Exact matches: {matches}/{min(len(pred_tabs), len(label_tabs))}")


def main():
    """Run comprehensive testing on real data."""
    checkpoint_path = "checkpoints_conditioning_scratch_retrain/checkpoint-642982"

    print("="*80)
    print("TESTING RETRAINED MODEL ON REAL DATA")
    print("="*80)
    print("This script uses the EXACT SAME configuration as training")
    print("to sanity check if the model learned the task correctly.")
    print("="*80)

    # Set seeds for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    # Load model
    model, tokenizer = load_model_and_tokenizer(checkpoint_path)

    # Load datasets (EXACT same config as training)
    train_dataset, val_dataset, tokenizer = load_datasets(tokenizer)

    # Evaluate on validation set (most important - these are unseen songs!)
    val_metrics = evaluate_on_dataset(
        model, tokenizer, val_dataset,
        num_samples=50,
        dataset_name="VALIDATION SET"
    )

    # Evaluate on training set (should do better if model trained)
    train_metrics = evaluate_on_dataset(
        model, tokenizer, train_dataset,
        num_samples=50,
        dataset_name="TRAINING SET"
    )

    # Show detailed examples
    show_example_predictions(model, tokenizer, val_dataset, num_examples=3)

    # Final summary
    print(f"\n\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"\nValidation Set Performance:")
    print(f"  Tab Accuracy:        {val_metrics['tab_accuracy']:.2%}")
    print(f"  Pitch Accuracy:      {val_metrics['pitch_accuracy']:.2%}")
    print(f"  Sequence Validity:   {val_metrics['sequence_validity']:.2%}")

    print(f"\nTraining Set Performance:")
    print(f"  Tab Accuracy:        {train_metrics['tab_accuracy']:.2%}")
    print(f"  Pitch Accuracy:      {train_metrics['pitch_accuracy']:.2%}")
    print(f"  Sequence Validity:   {train_metrics['sequence_validity']:.2%}")

    print(f"\n{'='*80}")
    print("Interpretation Guide:")
    print("  - Tab Accuracy: Exact string AND fret match")
    print("  - Pitch Accuracy: Correct MIDI pitch (any valid fingering)")
    print("  - Sequence Validity: Follows TAB↔TIME_SHIFT pattern")
    print()
    print("Expected Performance (scratch model with 614k steps):")
    print("  - Good: 40%+ tab accuracy, 60%+ pitch accuracy")
    print("  - Excellent: 60%+ tab accuracy, 75%+ pitch accuracy")
    print(f"{'='*80}")

    print("\n✓ Testing complete!")


if __name__ == "__main__":
    main()
