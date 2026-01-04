#!/usr/bin/env python3
"""
Test best_model.pt checkpoint on actual training/validation data.

Usage:
    python test_best_model.py <path_to_best_model.pt> [--num_samples 50] [--split val]

Example:
    python test_best_model.py model_outputs/guitarset_finetuned_h5splits_WITH_LEAKAGE/best_model.pt --num_samples 100 --split val
"""

import sys
import argparse
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
    """Load the trained model from best_model.pt checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    # Load tokenizer
    tokenizer = MidiTabTokenizerV3.load("universal_tokenizer")
    
    # Check if conditioning is enabled
    conditioning_enabled = checkpoint.get('conditioning_enabled', False)
    print(f"Conditioning enabled: {conditioning_enabled}")
    
    if conditioning_enabled:
        tokenizer.ensure_conditioning_tokens(
            capo_values=tuple(range(8)),
            tuning_options=DEFAULT_CONDITIONING_TUNINGS
        )
    
    # Get model config
    model_config = checkpoint.get('model_config')
    print(f"\nModel Configuration:")
    if hasattr(model_config, '__dict__'):
        for key, value in model_config.__dict__.items():
            print(f"  {key}: {value}")
    
    # Get vocab size from model state dict (embedding layer)
    state_dict = checkpoint['model_state_dict']
    vocab_size = state_dict['encoder.embed_tokens.weight'].shape[0]
    print(f"\nVocab size (from model): {vocab_size}")
    
    # Create model from config and load state dict
    from transformers import T5Config
    
    hf_config = T5Config(
        vocab_size=vocab_size,
        d_model=model_config.d_model,
        d_ff=model_config.d_ff,
        num_layers=model_config.num_layers,
        num_heads=model_config.num_heads,
        dropout_rate=model_config.dropout_rate,
        is_encoder_decoder=True,
        decoder_start_token_id=tokenizer.shared_token_to_id.get("<sos>", 0),
        eos_token_id=tokenizer.shared_token_to_id["<eos>"],
        pad_token_id=tokenizer.shared_token_to_id["<pad>"],
    )
    
    model = T5ForConditionalGeneration(hf_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        print(f"\n✓ Model loaded successfully (GPU)")
    else:
        print(f"\n✓ Model loaded successfully (CPU)")

    print(f"  - Vocab size: {model.config.vocab_size}")
    print(f"  - d_model: {model.config.d_model}")
    print(f"  - num_layers: {model.config.num_layers}")
    print(f"  - num_heads: {model.config.num_heads}")
    
    # Print finetuning info if available
    if 'finetuned_from' in checkpoint:
        print(f"  - Finetuned from: {checkpoint['finetuned_from']}")

    return model, tokenizer, checkpoint


def load_datasets(tokenizer, data_manifest, checkpoint):
    """Load datasets based on checkpoint configuration."""
    print("\nLoading datasets...")
    
    # Get data config from checkpoint
    data_config = checkpoint.get('data_config')
    conditioning_enabled = checkpoint.get('conditioning_enabled', False)
    
    if data_config is None:
        # Use default config
        print("Warning: No data_config in checkpoint, using defaults")
        data_config = DataConfig(
            max_encoder_length=512,
            max_decoder_length=512,
            train_on_time_shift=True,
            tab_loss_weight=1.2,
            enable_conditioning=conditioning_enabled,
        )
        
        if conditioning_enabled:
            data_config.conditioning_capo_values_train = tuple(range(8))
            data_config.conditioning_capo_values_eval = (0,)
            data_config.conditioning_tunings_train = DEFAULT_CONDITIONING_TUNINGS
            data_config.conditioning_tunings_eval = (DEFAULT_CONDITIONING_TUNINGS[0],)
            data_config.randomize_tuning_per_sequence = True
    
    print(f"Data config:")
    if hasattr(data_config, '__dict__'):
        for key, value in data_config.__dict__.items():
            print(f"  {key}: {value}")
    
    # Create song-level splits
    train_dataset, val_dataset, _ = create_song_level_splits(
        tokenizer=tokenizer,
        manifests=[Path(data_manifest)],
        data_config=data_config,
        train_ratio=0.8,
        val_ratio=0.2,
        test_ratio=0.0
    )

    print(f"✓ Train dataset: {len(train_dataset)} examples")
    print(f"✓ Val dataset: {len(val_dataset)} examples")

    return train_dataset, val_dataset, tokenizer


def predict_batch(model, tokenizer, examples, max_length=512, pad_to_multiple_of=8):
    """Run model prediction on a batch of examples."""
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

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids_batch,
            max_length=max_length,
            num_beams=1,
            do_sample=False,
            eos_token_id=model.config.eos_token_id,
            pad_token_id=model.config.pad_token_id,
        )

    outputs_cpu = outputs.cpu()

    # Pad predictions and labels to same length
    max_len = max(outputs_cpu.shape[1], max(lbl.shape[0] for lbl in labels_list))

    if pad_to_multiple_of is not None and max_len % pad_to_multiple_of != 0:
        max_len = ((max_len + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

    # Pad predictions
    predictions_padded = []
    for pred in outputs_cpu:
        if pred.shape[0] < max_len:
            padding = torch.zeros(max_len - pred.shape[0], dtype=torch.long)
            pred = torch.cat([pred, padding])
        predictions_padded.append(pred)

    predictions_batch = torch.stack(predictions_padded)

    # Pad labels
    labels_padded = []
    for lbl in labels_list:
        if lbl.shape[0] < max_len:
            padding = torch.full((max_len - lbl.shape[0],), -100, dtype=torch.long)
            lbl = torch.cat([lbl, padding])
        labels_padded.append(lbl)

    labels_batch = torch.stack(labels_padded)

    return predictions_batch.numpy(), labels_batch.numpy()


def evaluate_on_dataset(model, tokenizer, dataset, num_samples=50, dataset_name="Dataset"):
    """Evaluate model on samples from dataset."""
    print(f"\n{'='*80}")
    print(f"Evaluating on {num_samples} samples from {dataset_name}")
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

    # Compute metrics
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
        if conditioning_tokens:
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
    parser = argparse.ArgumentParser(description='Test best_model.pt on real data')
    parser.add_argument('checkpoint_path', type=str, help='Path to best_model.pt')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of samples to evaluate')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'both'], default='val',
                        help='Which split to evaluate on')
    parser.add_argument('--data_manifest', type=str, default='data/synthtab_acoustic_all.jsonl',
                        help='Path to data manifest file')
    parser.add_argument('--show_examples', type=int, default=3, help='Number of detailed examples to show')
    
    args = parser.parse_args()

    print("="*80)
    print("TESTING BEST_MODEL.PT ON REAL DATA")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Data manifest: {args.data_manifest}")
    print(f"Split: {args.split}")
    print(f"Num samples: {args.num_samples}")
    print("="*80)

    # Set seeds for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    # Load model
    model, tokenizer, checkpoint = load_model_and_tokenizer(args.checkpoint_path)

    # Load datasets
    train_dataset, val_dataset, tokenizer = load_datasets(tokenizer, args.data_manifest, checkpoint)

    # Evaluate based on split argument
    if args.split == 'val' or args.split == 'both':
        val_metrics = evaluate_on_dataset(
            model, tokenizer, val_dataset,
            num_samples=args.num_samples,
            dataset_name="VALIDATION SET"
        )

    if args.split == 'train' or args.split == 'both':
        train_metrics = evaluate_on_dataset(
            model, tokenizer, train_dataset,
            num_samples=args.num_samples,
            dataset_name="TRAINING SET"
        )

    # Show detailed examples
    if args.show_examples > 0:
        dataset_for_examples = val_dataset if args.split != 'train' else train_dataset
        show_example_predictions(model, tokenizer, dataset_for_examples, num_examples=args.show_examples)

    # Final summary
    print(f"\n\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    
    if args.split == 'val' or args.split == 'both':
        print(f"\nValidation Set Performance:")
        print(f"  Tab Accuracy:        {val_metrics['tab_accuracy']:.2%}")
        print(f"  Pitch Accuracy:      {val_metrics['pitch_accuracy']:.2%}")
        print(f"  Sequence Validity:   {val_metrics['sequence_validity']:.2%}")

    if args.split == 'train' or args.split == 'both':
        print(f"\nTraining Set Performance:")
        print(f"  Tab Accuracy:        {train_metrics['tab_accuracy']:.2%}")
        print(f"  Pitch Accuracy:      {train_metrics['pitch_accuracy']:.2%}")
        print(f"  Sequence Validity:   {train_metrics['sequence_validity']:.2%}")

    print(f"\n{'='*80}")
    print("Interpretation Guide:")
    print("  - Tab Accuracy: Exact string AND fret match")
    print("  - Pitch Accuracy: Correct MIDI pitch (any valid fingering)")
    print("  - Sequence Validity: Follows TAB↔TIME_SHIFT pattern")
    print(f"{'='*80}")

    print("\n✓ Testing complete!")


if __name__ == "__main__":
    main()
