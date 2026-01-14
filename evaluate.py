#!/usr/bin/env python3
"""
Unified Evaluation Script for Fretting-Transformer

Evaluates trained models on various datasets with optional post-processing.

Usage Examples:
    # Evaluate on SynthTab validation set
    python evaluate.py checkpoints/best_model.pt --dataset synthtab

    # Evaluate on GuitarSet test set
    python evaluate.py checkpoints/best_model.pt --dataset guitarset --split test

    # Evaluate on DadaGP with custom number of samples
    python evaluate.py checkpoints/best_model.pt --dataset dadagp --num-samples 100

    # Evaluate with post-processing disabled
    python evaluate.py checkpoints/best_model.pt --dataset synthtab --no-postprocess

    # Custom manifest
    python evaluate.py checkpoints/best_model.pt --dataset custom --manifest data/my_test.jsonl
"""

import argparse
import json
import sys
import numpy as np
import torch
from dataclasses import replace
from pathlib import Path
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from fret_t5 import (
    MidiTabTokenizerV3,
    DataConfig,
    SynthTabTokenDataset,
    chunk_tokenized_track,
    V3ConstrainedProcessor,
    DEFAULT_CONDITIONING_TUNINGS,
    STANDARD_TUNING,
)
from fret_t5.postprocess import (
    parse_capo_token,
    parse_tuning_token,
    extract_input_notes,
    extract_output_tabs,
    align_sequences_with_window,
    tab_to_midi_pitch,
    find_alternative_fingerings,
    select_best_fingering,
)
from transformers import T5Config, T5ForConditionalGeneration


# =============================================================================
# Dataset Presets
# =============================================================================

DATASET_PRESETS = {
    "synthtab": {
        "manifest_template": "data/synthtab_acoustic_{split}.jsonl",
        "description": "SynthTab synthetic guitar tablature",
    },
    "dadagp": {
        "manifest_template": "data/dadagp_acoustic_{split}.jsonl",
        "description": "DadaGP Guitar Pro extracted data",
    },
    "guitarset": {
        "guitarset_dir": "/data/akshaj/MusicAI/GuitarSet/annotation",
        "description": "GuitarSet real guitar recordings",
    },
    "custom": {
        "description": "Custom dataset from user-provided manifest",
    },
}


# =============================================================================
# Difficulty Calculation
# =============================================================================

def transition_difficulty(p: Tuple[int, int], q: Tuple[int, int]) -> float:
    """Calculate transition difficulty between two fret positions."""
    p_string, p_fret = p
    q_string, q_fret = q

    # Fret stretch
    delta_fret = q_fret - p_fret
    fret_stretch = 0.50 * abs(delta_fret) if delta_fret > 0 else 0.75 * abs(delta_fret)

    # Locality (higher frets harder)
    locality = 0.25 * (p_fret + q_fret)

    # Vertical stretch
    delta_string = abs(q_string - p_string)
    vert_stretch = 0.25 if delta_string <= 1 else 0.50

    return fret_stretch + locality + vert_stretch


def calculate_sequence_difficulty(positions: List[Tuple[int, int]]) -> float:
    """Calculate mean difficulty score for a tablature sequence."""
    if len(positions) < 2:
        return 0.0

    total = sum(
        transition_difficulty(positions[i], positions[i + 1])
        for i in range(len(positions) - 1)
    )
    return total / (len(positions) - 1)


# =============================================================================
# Model Loading
# =============================================================================

def load_model_and_tokenizer(checkpoint_path: str) -> Tuple[T5ForConditionalGeneration, MidiTabTokenizerV3]:
    """Load model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Load tokenizer
    tokenizer = MidiTabTokenizerV3.load("universal_tokenizer")
    tokenizer.ensure_conditioning_tokens(
        capo_values=tuple(range(8)),
        tuning_options=DEFAULT_CONDITIONING_TUNINGS
    )

    # Get model config
    model_config = checkpoint.get('model_config')
    state_dict = checkpoint['model_state_dict']
    vocab_size = state_dict['encoder.embed_tokens.weight'].shape[0]

    # Create model
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
    model.load_state_dict(state_dict)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"   Model loaded on {device}")

    return model, tokenizer


# =============================================================================
# Post-Processing
# =============================================================================

def extract_conditioning(encoder_tokens: List[str]) -> Tuple[int, Tuple[int, ...]]:
    """Extract capo and tuning from encoder tokens."""
    capo = 0
    tuning = STANDARD_TUNING

    for token in encoder_tokens[:5]:
        if token.startswith('CAPO<'):
            parsed = parse_capo_token(token)
            if parsed is not None:
                capo = parsed
        elif token.startswith('TUNING<'):
            parsed = parse_tuning_token(token)
            if parsed is not None:
                tuning = parsed

    return capo, tuning


def postprocess_predictions(
    encoder_tokens: List[str],
    decoder_tokens: List[str],
    capo: int = 0,
    tuning: Tuple[int, ...] = STANDARD_TUNING,
    pitch_window: int = 5,
    alignment_window: int = 5,
) -> Tuple[List[str], Dict[str, int]]:
    """Apply post-processing to correct pitch and time shift errors."""
    input_notes = extract_input_notes(encoder_tokens)
    output_tabs = extract_output_tabs(decoder_tokens)

    stats = {
        'pitch_corrections': 0,
        'time_shift_corrections': 0,
        'pitch_too_far': 0,
        'unaligned_outputs': 0,
    }

    if not output_tabs:
        return decoder_tokens, stats

    alignments = align_sequences_with_window(input_notes, output_tabs, alignment_window)
    corrected_tokens = []

    for input_idx, output_idx in alignments:
        out_string, out_fret, out_time_shift = output_tabs[output_idx]

        if input_idx is None:
            stats['unaligned_outputs'] += 1
            corrected_tokens.extend([
                f"TAB<{out_string},{out_fret}>",
                f"TIME_SHIFT<{out_time_shift}>"
            ])
            continue

        input_pitch, input_time_shift = input_notes[input_idx]
        predicted_pitch = tab_to_midi_pitch(out_string, out_fret, capo, tuning)
        pitch_diff = abs(input_pitch - predicted_pitch)

        if pitch_diff == 0:
            corrected_string, corrected_fret = out_string, out_fret
        elif pitch_diff <= pitch_window:
            alternatives = find_alternative_fingerings(input_pitch, capo, tuning)
            if alternatives:
                corrected_string, corrected_fret = select_best_fingering(
                    alternatives, out_string, out_fret
                )
                if (corrected_string, corrected_fret) != (out_string, out_fret):
                    stats['pitch_corrections'] += 1
            else:
                corrected_string, corrected_fret = out_string, out_fret
        else:
            corrected_string, corrected_fret = out_string, out_fret
            stats['pitch_too_far'] += 1

        if input_time_shift != out_time_shift:
            stats['time_shift_corrections'] += 1

        corrected_tokens.extend([
            f"TAB<{corrected_string},{corrected_fret}>",
            f"TIME_SHIFT<{input_time_shift}>"
        ])

    if decoder_tokens and decoder_tokens[-1] == "<eos>":
        corrected_tokens.append("<eos>")

    return corrected_tokens, stats


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_metrics(
    encoder_tokens: List[str],
    decoder_tokens: List[str],
    capo: int = 0,
    tuning: Tuple[int, ...] = STANDARD_TUNING,
    ground_truth_tokens: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    input_notes = extract_input_notes(encoder_tokens)
    output_tabs = extract_output_tabs(decoder_tokens)

    if not input_notes:
        return {'pitch_accuracy': 0.0, 'time_shift_accuracy': 0.0, 'tab_accuracy': 0.0, 'difficulty': 0.0}

    min_len = min(len(input_notes), len(output_tabs))
    if min_len == 0:
        return {'pitch_accuracy': 0.0, 'time_shift_accuracy': 0.0, 'tab_accuracy': 0.0, 'difficulty': 0.0}

    pitch_matches = 0
    time_matches = 0
    tab_matches = 0

    ground_truth_tabs = extract_output_tabs(ground_truth_tokens) if ground_truth_tokens else None

    for i in range(min_len):
        input_pitch, input_time = input_notes[i]
        out_string, out_fret, out_time = output_tabs[i]

        pred_pitch = tab_to_midi_pitch(out_string, out_fret, capo, tuning)

        if input_pitch == pred_pitch:
            pitch_matches += 1
        if input_time == out_time:
            time_matches += 1

        if ground_truth_tabs and i < len(ground_truth_tabs):
            gt_string, gt_fret, _ = ground_truth_tabs[i]
            if out_string == gt_string and out_fret == gt_fret:
                tab_matches += 1

    positions = [(s, f) for s, f, _ in output_tabs[:min_len]]
    difficulty = calculate_sequence_difficulty(positions)

    metrics = {
        'pitch_accuracy': (pitch_matches / min_len) * 100,
        'time_shift_accuracy': (time_matches / min_len) * 100,
        'difficulty': difficulty,
        'total_notes': min_len,
    }

    if ground_truth_tabs:
        metrics['tab_accuracy'] = (tab_matches / min(min_len, len(ground_truth_tabs))) * 100
        gt_positions = [(s, f) for s, f, _ in ground_truth_tabs[:min_len]]
        metrics['gt_difficulty'] = calculate_sequence_difficulty(gt_positions)
    else:
        metrics['tab_accuracy'] = 0.0
        metrics['gt_difficulty'] = 0.0

    return metrics


# =============================================================================
# Dataset Loading
# =============================================================================

def load_guitarset_examples(
    guitarset_dir: Path,
    tokenizer: MidiTabTokenizerV3,
    split: str = "val",
    split_file: Optional[Path] = None,
) -> List[Dict]:
    """Load GuitarSet examples with proper chunking."""
    from guitarset_loader import load_guitarset_jams, extract_tablature_from_guitarset_jams

    # Get file list
    if split_file and split_file.exists():
        with open(split_file, 'r') as f:
            split_data = json.load(f)
        jams_files = [Path(p) for p in split_data.get(f"{split}_files", [])]
        print(f"   Using split file: {split_file}")
    else:
        jams_files = list(guitarset_dir.glob("*.jams"))
        print(f"   No split file, using all {len(jams_files)} files")

    print(f"   Found {len(jams_files)} files for {split} split")

    # Setup chunking config
    data_config = DataConfig(
        max_encoder_length=512,
        max_decoder_length=512,
        enable_conditioning=True,
    )

    capo_val = 0
    tuning_val = STANDARD_TUNING
    sample_prefix = tokenizer.build_conditioning_prefix(capo_val, tuning_val)
    chunk_config = replace(data_config, max_encoder_length=512 - len(sample_prefix))

    examples = []
    skipped = 0

    for file_path in jams_files:
        try:
            jams_data = load_guitarset_jams(file_path)
            tab_events = extract_tablature_from_guitarset_jams(jams_data, auto_detect_tuning=False)

            if len(tab_events) < 10:
                skipped += 1
                continue

            # Convert to tokenizer format
            jams_events = [{
                "string": float(e["string"]),
                "fret": float(e["fret"]),
                "duration_ms": float(e.get("duration", 0.5) * 1000),
                "time_ticks": float(e.get("time", 0) * 1000)
            } for e in tab_events]

            tokenized = tokenizer.tokenize_track_from_jams(jams_events)
            chunks = list(chunk_tokenized_track(tokenized, chunk_config))

            for enc_tokens, dec_tokens, _ in chunks:
                prefix = tokenizer.build_conditioning_prefix(capo_val, tuning_val)
                examples.append({
                    'encoder_tokens': prefix + list(enc_tokens),
                    'decoder_tokens': list(dec_tokens),
                    'source': file_path.name,
                })

        except Exception:
            skipped += 1

    print(f"   Created {len(examples)} examples ({skipped} files skipped)")
    return examples


# =============================================================================
# Main Evaluation
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified evaluation script for Fretting-Transformer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint (best_model.pt)")

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["synthtab", "guitarset", "dadagp", "custom"],
        default="synthtab",
        help="Dataset to evaluate on"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Data split to use (default: val)"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Custom manifest file (for --dataset custom)"
    )
    parser.add_argument(
        "--guitarset-dir",
        type=str,
        default=None,
        help="GuitarSet directory (overrides preset)"
    )
    parser.add_argument(
        "--split-file",
        type=str,
        default=None,
        help="GuitarSet split JSON file"
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of samples to evaluate (0 for all)"
    )

    parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="Disable post-processing"
    )
    parser.add_argument(
        "--pitch-window",
        type=int,
        default=5,
        help="Pitch correction window in MIDI notes"
    )
    parser.add_argument(
        "--alignment-window",
        type=int,
        default=5,
        help="Alignment window for input/output matching"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sample results"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    preset = DATASET_PRESETS[args.dataset]

    print("=" * 70)
    print(f"Fretting-Transformer Evaluation")
    print(f"Dataset: {args.dataset} - {preset['description']}")
    print(f"Split: {args.split}")
    print(f"Checkpoint: {args.checkpoint}")
    print("=" * 70)

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.checkpoint)
    constrained_processor = V3ConstrainedProcessor(tokenizer)
    device = next(model.parameters()).device

    # Load examples
    examples = []

    if args.dataset == "guitarset":
        guitarset_dir = Path(args.guitarset_dir or preset["guitarset_dir"])
        split_file = Path(args.split_file) if args.split_file else None
        examples = load_guitarset_examples(
            guitarset_dir, tokenizer, args.split, split_file
        )

    elif args.dataset in ("synthtab", "dadagp"):
        manifest_path = preset["manifest_template"].format(split=args.split)
        if not Path(manifest_path).exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        print(f"\nLoading {args.dataset} {args.split} dataset...")
        data_config = DataConfig(
            max_encoder_length=512,
            max_decoder_length=512,
            enable_conditioning=True,
            conditioning_capo_values_eval=(0,),
            conditioning_tunings_eval=(STANDARD_TUNING,),
        )

        dataset = SynthTabTokenDataset(
            tokenizer=tokenizer,
            manifests=[Path(manifest_path)],
            data_config=data_config,
            split=args.split,
            preload=True
        )

        for i in range(len(dataset)):
            example = dataset[i]
            enc_ids = [id for id in example['input_ids'].tolist() if id != tokenizer.shared_token_to_id["<pad>"]]
            lab_ids = [id for id in example['labels'].tolist() if id != tokenizer.shared_token_to_id["<pad>"] and id != -100]

            examples.append({
                'encoder_tokens': tokenizer.shared_to_encoder_tokens(enc_ids),
                'decoder_tokens': tokenizer.shared_to_decoder_tokens(lab_ids),
                'source': f"example_{i}",
            })

    elif args.dataset == "custom":
        if not args.manifest:
            raise ValueError("--manifest is required for custom dataset")
        # Similar to synthtab loading
        data_config = DataConfig(
            max_encoder_length=512,
            max_decoder_length=512,
            enable_conditioning=True,
        )
        dataset = SynthTabTokenDataset(
            tokenizer=tokenizer,
            manifests=[Path(args.manifest)],
            data_config=data_config,
            split="all",
            preload=True
        )
        for i in range(len(dataset)):
            example = dataset[i]
            enc_ids = [id for id in example['input_ids'].tolist() if id != tokenizer.shared_token_to_id["<pad>"]]
            lab_ids = [id for id in example['labels'].tolist() if id not in (tokenizer.shared_token_to_id["<pad>"], -100)]
            examples.append({
                'encoder_tokens': tokenizer.shared_to_encoder_tokens(enc_ids),
                'decoder_tokens': tokenizer.shared_to_decoder_tokens(lab_ids),
                'source': f"example_{i}",
            })

    print(f"   Total examples: {len(examples)}")

    # Limit samples
    num_samples = args.num_samples if args.num_samples > 0 else len(examples)
    num_samples = min(num_samples, len(examples))
    print(f"\nEvaluating on {num_samples} samples...")

    # Evaluate
    all_original = []
    all_postprocessed = []

    for i in range(num_samples):
        example = examples[i]
        encoder_tokens = example['encoder_tokens']
        ground_truth = example['decoder_tokens']

        # Encode
        encoder_ids = tokenizer.encode_encoder_tokens_shared(encoder_tokens)
        input_tensor = torch.tensor([encoder_ids], dtype=torch.long, device=device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_tensor,
                max_length=512,
                num_beams=1,
                do_sample=False,
                eos_token_id=tokenizer.shared_token_to_id["<eos>"],
                pad_token_id=tokenizer.shared_token_to_id["<pad>"],
                logits_processor=[constrained_processor],
            )

        pred_tokens = tokenizer.shared_to_decoder_tokens(outputs[0].cpu().tolist())

        # Extract conditioning
        capo, tuning = extract_conditioning(encoder_tokens)

        # Original metrics
        orig_metrics = compute_metrics(encoder_tokens, pred_tokens, capo, tuning, ground_truth)
        all_original.append(orig_metrics)

        # Post-processed metrics
        if not args.no_postprocess:
            post_tokens, stats = postprocess_predictions(
                encoder_tokens, pred_tokens, capo, tuning,
                args.pitch_window, args.alignment_window
            )
            post_metrics = compute_metrics(encoder_tokens, post_tokens, capo, tuning, ground_truth)
        else:
            post_metrics = orig_metrics
            stats = {}

        all_postprocessed.append(post_metrics)

        if args.verbose:
            print(f"[{i+1}/{num_samples}] {example['source'][:30]}")
            print(f"   Original:  Tab={orig_metrics['tab_accuracy']:.1f}%, Pitch={orig_metrics['pitch_accuracy']:.1f}%")
            if not args.no_postprocess:
                print(f"   PostProc:  Tab={post_metrics['tab_accuracy']:.1f}%, Pitch={post_metrics['pitch_accuracy']:.1f}%")

    # Aggregate results
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    def avg(metrics_list, key):
        values = [m.get(key, 0) for m in metrics_list]
        return np.mean(values) if values else 0

    print(f"\nOriginal Model ({num_samples} samples):")
    print(f"   Tab Accuracy:        {avg(all_original, 'tab_accuracy'):.2f}%")
    print(f"   Pitch Accuracy:      {avg(all_original, 'pitch_accuracy'):.2f}%")
    print(f"   Time Shift Accuracy: {avg(all_original, 'time_shift_accuracy'):.2f}%")
    print(f"   Difficulty:          {avg(all_original, 'difficulty'):.3f}")

    if not args.no_postprocess:
        print(f"\nPost-Processed:")
        print(f"   Tab Accuracy:        {avg(all_postprocessed, 'tab_accuracy'):.2f}%")
        print(f"   Pitch Accuracy:      {avg(all_postprocessed, 'pitch_accuracy'):.2f}%")
        print(f"   Time Shift Accuracy: {avg(all_postprocessed, 'time_shift_accuracy'):.2f}%")
        print(f"   Difficulty:          {avg(all_postprocessed, 'difficulty'):.3f}")

        print(f"\nImprovement:")
        print(f"   Tab Accuracy:        +{avg(all_postprocessed, 'tab_accuracy') - avg(all_original, 'tab_accuracy'):.2f}%")
        print(f"   Pitch Accuracy:      +{avg(all_postprocessed, 'pitch_accuracy') - avg(all_original, 'pitch_accuracy'):.2f}%")

    gt_diff = avg(all_original, 'gt_difficulty')
    if gt_diff > 0:
        print(f"\nGround Truth Difficulty: {gt_diff:.3f}")

    print("\n" + "=" * 70)
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
