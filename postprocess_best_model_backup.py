#!/usr/bin/env python3
"""
Post-processing evaluation for best_model.pt checkpoints.

Usage:
    python postprocess_best_model.py <path_to_best_model.pt> --dataset synthtab --num_pieces 50
    python postprocess_best_model.py <path_to_best_model.pt> --dataset synthtab --split val --num_pieces 100

Example:
    python postprocess_best_model.py model_outputs/guitarset_finetuned_h5splits_WITH_LEAKAGE/best_model.pt --dataset synthtab --num_pieces 50
"""

import os
import sys
import argparse
import torch
import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from fret_t5.tokenization import MidiTabTokenizerV3, DEFAULT_CONDITIONING_TUNINGS
from transformers import T5ForConditionalGeneration
from fret_t5.data import DataConfig, SynthTabTokenDataset

# Standard tuning (high E to low E)
STANDARD_TUNING = (64, 59, 55, 50, 45, 40)

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


def load_model_and_tokenizer(checkpoint_path: str):
    """Load model from best_model.pt checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return None, None
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Load tokenizer
    tokenizer = MidiTabTokenizerV3.load("universal_tokenizer")
    
    # Check conditioning
    conditioning_enabled = checkpoint.get('conditioning_enabled', False)
    print(f"Conditioning enabled: {conditioning_enabled}")
    
    if conditioning_enabled:
        tokenizer.ensure_conditioning_tokens(
            capo_values=tuple(range(8)),
            tuning_options=DEFAULT_CONDITIONING_TUNINGS
        )
    
    # Get model config
    model_config = checkpoint.get('model_config')
    
    # Get vocab size from model state dict (embedding layer)
    state_dict = checkpoint['model_state_dict']
    vocab_size = state_dict['encoder.embed_tokens.weight'].shape[0]
    
    # Create model
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
        print("✓ Using CUDA")
    else:
        print("✓ Using CPU")
    
    return model, tokenizer


def extract_conditioning_from_encoder(encoder_tokens: List[str]) -> Tuple[int, Tuple[int, ...]]:
    """Extract capo and tuning from encoder tokens."""
    capo = 0
    tuning = STANDARD_TUNING

    for token in encoder_tokens[:5]:
        if token.startswith('CAPO<'):
            parsed_capo = parse_capo_token(token)
            if parsed_capo is not None:
                capo = parsed_capo
        elif token.startswith('TUNING<'):
            parsed_tuning = parse_tuning_token(token)
            if parsed_tuning is not None:
                tuning = parsed_tuning

    return capo, tuning


def postprocess_predictions(
    encoder_tokens: List[str],
    decoder_tokens: List[str],
    capo: int = 0,
    tuning: Tuple[int, ...] = STANDARD_TUNING,
    pitch_window: int = 5,
    alignment_window: int = 5,
    debug: bool = False
) -> Tuple[List[str], Dict[str, int]]:
    """Apply post-processing to correct pitch and time shift errors."""
    input_notes = extract_input_notes(encoder_tokens)
    output_tabs = extract_output_tabs(decoder_tokens)

    stats = {
        'pitch_corrections': 0,
        'time_shift_corrections': 0,
        'pitch_too_far': 0,
        'unaligned_outputs': 0,
        'input_length': len(input_notes),
        'output_length': len(output_tabs)
    }

    if len(output_tabs) == 0:
        return decoder_tokens, stats

    alignments = align_sequences_with_window(input_notes, output_tabs, alignment_window)

    if debug and len(input_notes) != len(output_tabs):
        print(f"  DEBUG: Length mismatch - input={len(input_notes)}, output={len(output_tabs)}")
        print(f"  DEBUG: Aligned {len([a for a in alignments if a[0] is not None])}/{len(output_tabs)} outputs")

    corrected_tokens = []

    for input_idx, output_idx in alignments:
        out_string, out_fret, out_time_shift = output_tabs[output_idx]

        if input_idx is None:
            stats['unaligned_outputs'] += 1
            corrected_tokens.append(f"TAB<{out_string},{out_fret}>")
            corrected_tokens.append(f"TIME_SHIFT<{out_time_shift}>")
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
                    if debug:
                        print(f"  DEBUG: Corrected pitch at output_idx={output_idx}: "
                              f"TAB<{out_string},{out_fret}> (pitch={predicted_pitch}) -> "
                              f"TAB<{corrected_string},{corrected_fret}> (pitch={input_pitch})")
            else:
                corrected_string, corrected_fret = out_string, out_fret
        else:
            corrected_string, corrected_fret = out_string, out_fret
            stats['pitch_too_far'] += 1
            if debug:
                print(f"  DEBUG: Pitch difference too large at output_idx={output_idx}: "
                      f"{pitch_diff} MIDI notes (input={input_pitch}, pred={predicted_pitch})")

        if input_time_shift != out_time_shift:
            stats['time_shift_corrections'] += 1

        corrected_time_shift = input_time_shift

        corrected_tokens.append(f"TAB<{corrected_string},{corrected_fret}>")
        corrected_tokens.append(f"TIME_SHIFT<{corrected_time_shift}>")

    if decoder_tokens and decoder_tokens[-1] == "<eos>":
        corrected_tokens.append("<eos>")

    return corrected_tokens, stats


def compute_accuracy_metrics(
    encoder_tokens: List[str],
    decoder_tokens: List[str],
    capo: int = 0,
    tuning: Tuple[int, ...] = STANDARD_TUNING,
    ground_truth_tokens: List[str] = None
) -> Dict[str, float]:
    """Compute pitch, time shift, and tab accuracy metrics."""
    input_notes = extract_input_notes(encoder_tokens)
    output_tabs = extract_output_tabs(decoder_tokens)

    if len(input_notes) == 0:
        return {
            'pitch_accuracy': 0.0,
            'time_shift_accuracy': 0.0,
            'tab_accuracy': 0.0,
            'total_notes': 0
        }

    min_len = min(len(input_notes), len(output_tabs))

    if min_len == 0:
        return {
            'pitch_accuracy': 0.0,
            'time_shift_accuracy': 0.0,
            'tab_accuracy': 0.0,
            'total_notes': 0,
            'input_length': len(input_notes),
            'output_length': len(output_tabs)
        }

    pitch_matches = 0
    time_shift_matches = 0
    tab_matches = 0

    ground_truth_tabs = None
    if ground_truth_tokens:
        ground_truth_tabs = extract_output_tabs(ground_truth_tokens)

    for i in range(min_len):
        input_pitch, input_time_shift = input_notes[i]
        out_string, out_fret, out_time_shift = output_tabs[i]

        predicted_pitch = tab_to_midi_pitch(out_string, out_fret, capo, tuning)

        if input_pitch == predicted_pitch:
            pitch_matches += 1

        if input_time_shift == out_time_shift:
            time_shift_matches += 1

        if ground_truth_tabs and i < len(ground_truth_tabs):
            gt_string, gt_fret, _ = ground_truth_tabs[i]
            if out_string == gt_string and out_fret == gt_fret:
                tab_matches += 1

    metrics = {
        'pitch_accuracy': (pitch_matches / min_len) * 100,
        'time_shift_accuracy': (time_shift_matches / min_len) * 100,
        'total_notes': min_len,
        'input_length': len(input_notes),
        'output_length': len(output_tabs)
    }

    if ground_truth_tabs:
        metrics['tab_accuracy'] = (tab_matches / min_len) * 100
    else:
        metrics['tab_accuracy'] = 0.0

    return metrics


def main():
    """Main evaluation loop."""
    parser = argparse.ArgumentParser(description='Post-process predictions from best_model.pt')
    parser.add_argument('checkpoint_path', type=str, help='Path to best_model.pt')
    parser.add_argument('--dataset', type=str, choices=['synthtab'], default='synthtab',
                        help='Dataset to evaluate on')
    parser.add_argument('--split', type=str, choices=['train', 'val'], default='val',
                        help='Which split to use')
    parser.add_argument('--num_pieces', type=int, default=50,
                        help='Number of pieces to evaluate (0 for all)')
    parser.add_argument('--pitch_window', type=int, default=5,
                        help='Pitch correction window in MIDI notes')
    parser.add_argument('--alignment_window', type=int, default=5,
                        help='Alignment window for matching input to output')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"Post-Processing Evaluation on {args.dataset.upper()} ({args.split} split)")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Num pieces: {args.num_pieces if args.num_pieces > 0 else 'all'}")
    print(f"Pitch window: {args.pitch_window}")
    print(f"Alignment window: {args.alignment_window}")
    print("=" * 80)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.checkpoint_path)
    
    if model is None or tokenizer is None:
        return

    # Load dataset
    if args.dataset.lower() == "synthtab":
        data_config = DataConfig(
            max_encoder_length=512,
            max_decoder_length=512,
            enable_conditioning=True,
            conditioning_capo_values_eval=(0,),
            conditioning_tunings_eval=(STANDARD_TUNING,),
        )

        manifest_file = f"data/synthtab_acoustic_{args.split}.jsonl"
        
        if not Path(manifest_file).exists():
            print(f"ERROR: Manifest file not found: {manifest_file}")
            return

        print(f"\nLoading SynthTab {args.split} dataset...")
        dataset = SynthTabTokenDataset(
            tokenizer=tokenizer,
            manifests=[Path(manifest_file)],
            data_config=data_config,
            split=args.split,
            preload=True
        )

        if len(dataset) == 0:
            print(f"ERROR: No examples in dataset")
            return

        print(f"Found {len(dataset)} examples in {args.split} set")
        
        num_pieces = args.num_pieces if args.num_pieces > 0 else len(dataset)
        num_pieces = min(num_pieces, len(dataset))
        
        print(f"\nEvaluating on {num_pieces} pieces...")

        all_original_metrics = []
        all_postprocessed_metrics = []

        for i in range(num_pieces):
            example = dataset[i]
            print(f"\n[{i+1}/{num_pieces}] Processing example {i}...")

            # Extract tokens
            encoder_ids = example['input_ids'].tolist()
            label_ids = example['labels'].tolist()

            # Filter padding
            encoder_ids = [id for id in encoder_ids if id != tokenizer.shared_token_to_id["<pad>"]]
            label_ids = [id for id in label_ids if id != tokenizer.shared_token_to_id["<pad>"] and id != -100]

            encoder_tokens = tokenizer.shared_to_encoder_tokens(encoder_ids)
            ground_truth_tokens = tokenizer.shared_to_decoder_tokens(label_ids)

            print(f"  Loaded {len(extract_input_notes(encoder_tokens))} notes")

            # Generate prediction
            input_ids = torch.tensor(encoder_ids, dtype=torch.long).unsqueeze(0)

            if torch.cuda.is_available():
                input_ids = input_ids.cuda()

            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_length=512,
                    num_beams=1,
                    do_sample=False,
                    eos_token_id=tokenizer.shared_token_to_id["<eos>"],
                    pad_token_id=tokenizer.shared_token_to_id["<pad>"],
                )

            pred_ids = outputs[0].cpu().tolist()
            pred_tokens = tokenizer.shared_to_decoder_tokens(pred_ids)

            # Extract conditioning
            capo, tuning = extract_conditioning_from_encoder(encoder_tokens)

            # Compute original metrics
            original_metrics = compute_accuracy_metrics(
                encoder_tokens, pred_tokens, capo, tuning,
                ground_truth_tokens=ground_truth_tokens
            )
            all_original_metrics.append(original_metrics)

            # Check if debug needed
            has_errors = (original_metrics['pitch_accuracy'] < 100 or
                         original_metrics['time_shift_accuracy'] < 100)

            # Post-process
            postprocessed_tokens, correction_stats = postprocess_predictions(
                encoder_tokens, pred_tokens, capo, tuning,
                pitch_window=args.pitch_window,
                alignment_window=args.alignment_window,
                debug=has_errors
            )

            # Compute post-processed metrics
            postprocessed_metrics = compute_accuracy_metrics(
                encoder_tokens, postprocessed_tokens, capo, tuning,
                ground_truth_tokens=ground_truth_tokens
            )
            all_postprocessed_metrics.append(postprocessed_metrics)

            # Print results
            print(f"  Original:       Tab={original_metrics['tab_accuracy']:.1f}%, "
                  f"Pitch={original_metrics['pitch_accuracy']:.1f}%, "
                  f"TimeShift={original_metrics['time_shift_accuracy']:.1f}%")
            print(f"  Post-processed: Tab={postprocessed_metrics['tab_accuracy']:.1f}%, "
                  f"Pitch={postprocessed_metrics['pitch_accuracy']:.1f}%, "
                  f"TimeShift={postprocessed_metrics['time_shift_accuracy']:.1f}%")

            if has_errors or correction_stats['pitch_corrections'] > 0 or correction_stats['time_shift_corrections'] > 0:
                print(f"  Corrections: Pitch={correction_stats['pitch_corrections']}, "
                      f"TimeShift={correction_stats['time_shift_corrections']}, "
                      f"PitchTooFar={correction_stats['pitch_too_far']}, "
                      f"Unaligned={correction_stats['unaligned_outputs']}")

        # Aggregate results
        print("\n" + "=" * 80)
        print("AGGREGATE RESULTS")
        print("=" * 80)

        if all_original_metrics:
            avg_orig_pitch = np.mean([m['pitch_accuracy'] for m in all_original_metrics])
            avg_orig_time = np.mean([m['time_shift_accuracy'] for m in all_original_metrics])
            avg_post_pitch = np.mean([m['pitch_accuracy'] for m in all_postprocessed_metrics])
            avg_post_time = np.mean([m['time_shift_accuracy'] for m in all_postprocessed_metrics])

            avg_orig_tab = np.mean([m['tab_accuracy'] for m in all_original_metrics])
            avg_post_tab = np.mean([m['tab_accuracy'] for m in all_postprocessed_metrics])

            print(f"\nOriginal Model:")
            print(f"  Tab Accuracy:        {avg_orig_tab:.2f}%")
            print(f"  Pitch Accuracy:      {avg_orig_pitch:.2f}%")
            print(f"  Time Shift Accuracy: {avg_orig_time:.2f}%")

            print(f"\nPost-Processed:")
            print(f"  Tab Accuracy:        {avg_post_tab:.2f}%")
            print(f"  Pitch Accuracy:      {avg_post_pitch:.2f}%")
            print(f"  Time Shift Accuracy: {avg_post_time:.2f}%")

            print(f"\nImprovement:")
            print(f"  Tab Accuracy:        +{avg_post_tab - avg_orig_tab:.2f}%")
            print(f"  Pitch Accuracy:      +{avg_post_pitch - avg_orig_pitch:.2f}%")
            print(f"  Time Shift Accuracy: +{avg_post_time - avg_orig_time:.2f}%")

        print("\n" + "=" * 80)
        print("✓ Evaluation complete!")

    else:
        print(f"ERROR: Unknown dataset '{args.dataset}'")


if __name__ == "__main__":
    main()
