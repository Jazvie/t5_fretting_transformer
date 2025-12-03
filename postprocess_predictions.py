#!/usr/bin/env python3
"""
Post-processing algorithm for tablature predictions.

This script implements the post-processing algorithm described in section 3.5:
- Corrects pitch errors by matching to nearest valid fingering within ±5 MIDI notes
- Corrects time shift discrepancies to match input
- Uses fret_stretch metric to select best alternative fingering
"""

import os
import torch
import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import jams

from src.fret_t5.tokenization import MidiTabTokenizerV3, DEFAULT_CONDITIONING_TUNINGS
from transformers import T5ForConditionalGeneration


# Standard tuning (high E to low E)
STANDARD_TUNING = (64, 59, 55, 50, 45, 40)
OPEN_STRINGS = {
    1: 64,  # High E (E4)
    2: 59,  # B  (B3)
    3: 55,  # G  (G3)
    4: 50,  # D  (D3)
    5: 45,  # A  (A2)
    6: 40   # Low E (E2)
}

# GuitarSet string mapping (from test_guitarset_zero_shot.py)
GUITARSET_STRING_MAP = {
    "0": (6, 40),  # Low E
    "1": (5, 45),  # A
    "2": (4, 50),  # D
    "3": (3, 55),  # G
    "4": (2, 59),  # B
    "5": (1, 64),  # High E
}


from src.fret_t5.postprocess import (
    parse_tab_token,
    parse_time_shift_token,
    parse_note_on_token,
    parse_capo_token,
    parse_tuning_token,
    tuning_to_open_strings,
    tab_to_midi_pitch,
    fret_stretch,
    find_alternative_fingerings,
    select_best_fingering,
    extract_input_notes,
    extract_output_tabs,
    align_sequences_with_window,
)


def extract_conditioning_from_encoder(encoder_tokens: List[str]) -> Tuple[int, Tuple[int, ...]]:
    """
    Extract capo and tuning from encoder tokens.

    Returns:
        (capo, tuning) tuple with defaults if not found
    """
    capo = 0
    tuning = STANDARD_TUNING

    # Conditioning tokens are at the start
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
    """
    Apply post-processing to correct pitch and time shift errors.

    Args:
        encoder_tokens: Input MIDI tokens
        decoder_tokens: Model's predicted tablature tokens
        capo: Capo position
        pitch_window: Only correct if pitch difference <= this value (MIDI notes)
        alignment_window: Position window for matching input to output notes
        debug: Print debugging information

    Returns:
        (corrected_tokens, stats) tuple where stats contains correction counts
    """
    # Extract notes and tabs
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

    # Align sequences
    alignments = align_sequences_with_window(input_notes, output_tabs, alignment_window)

    if debug and len(input_notes) != len(output_tabs):
        print(f"  DEBUG: Length mismatch - input={len(input_notes)}, output={len(output_tabs)}")
        print(f"  DEBUG: Aligned {len([a for a in alignments if a[0] is not None])}/{len(output_tabs)} outputs")

    # Apply corrections
    corrected_tokens = []

    for input_idx, output_idx in alignments:
        out_string, out_fret, out_time_shift = output_tabs[output_idx]

        # If no aligned input, keep output as-is
        if input_idx is None:
            stats['unaligned_outputs'] += 1
            corrected_tokens.append(f"TAB<{out_string},{out_fret}>")
            corrected_tokens.append(f"TIME_SHIFT<{out_time_shift}>")
            continue

        input_pitch, input_time_shift = input_notes[input_idx]

        # Calculate predicted MIDI pitch
        predicted_pitch = tab_to_midi_pitch(out_string, out_fret, capo, tuning)
        pitch_diff = abs(input_pitch - predicted_pitch)

        # Determine correct fingering
        pitch_corrected = False
        if pitch_diff == 0:
            # Pitch already correct, keep original fingering
            corrected_string, corrected_fret = out_string, out_fret
        elif pitch_diff <= pitch_window:
            # Pitch within window, find best alternative fingering
            alternatives = find_alternative_fingerings(input_pitch, capo, tuning)
            if alternatives:
                corrected_string, corrected_fret = select_best_fingering(
                    alternatives, out_string, out_fret
                )
                if (corrected_string, corrected_fret) != (out_string, out_fret):
                    stats['pitch_corrections'] += 1
                    pitch_corrected = True
                    if debug:
                        print(f"  DEBUG: Corrected pitch at output_idx={output_idx}: "
                              f"TAB<{out_string},{out_fret}> (pitch={predicted_pitch}) -> "
                              f"TAB<{corrected_string},{corrected_fret}> (pitch={input_pitch})")
            else:
                # No valid alternative, keep original
                corrected_string, corrected_fret = out_string, out_fret
                if debug:
                    print(f"  DEBUG: No alternatives found for pitch {input_pitch}")
        else:
            # Pitch difference too large, keep original
            corrected_string, corrected_fret = out_string, out_fret
            stats['pitch_too_far'] += 1
            if debug:
                print(f"  DEBUG: Pitch difference too large at output_idx={output_idx}: "
                      f"{pitch_diff} MIDI notes (input={input_pitch}, pred={predicted_pitch})")

        # Correct time shift if different
        if input_time_shift != out_time_shift:
            stats['time_shift_corrections'] += 1
            if debug:
                print(f"  DEBUG: Corrected time shift at output_idx={output_idx}: "
                      f"{out_time_shift}ms -> {input_time_shift}ms")

        corrected_time_shift = input_time_shift

        # Add corrected tokens
        corrected_tokens.append(f"TAB<{corrected_string},{corrected_fret}>")
        corrected_tokens.append(f"TIME_SHIFT<{corrected_time_shift}>")

    # Add EOS token if present in original
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
    """
    Compute pitch, time shift, and tab accuracy metrics.

    Args:
        encoder_tokens: Input MIDI tokens
        decoder_tokens: Model's predicted tablature tokens
        capo: Capo position
        tuning: Guitar tuning
        ground_truth_tokens: Optional ground truth decoder tokens for tab accuracy

    Returns:
        Dictionary with accuracy percentages
    """
    input_notes = extract_input_notes(encoder_tokens)
    output_tabs = extract_output_tabs(decoder_tokens)

    if len(input_notes) == 0:
        return {
            'pitch_accuracy': 0.0,
            'time_shift_accuracy': 0.0,
            'tab_accuracy': 0.0,
            'total_notes': 0
        }

    # Handle length mismatch
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

    # Extract ground truth tabs if provided
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

        # Check tab accuracy if ground truth provided
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

    # Add tab accuracy if ground truth was provided
    if ground_truth_tabs:
        metrics['tab_accuracy'] = (tab_matches / min_len) * 100
    else:
        metrics['tab_accuracy'] = 0.0

    return metrics


def load_guitarset_piece(jams_path: str) -> List[Dict]:
    """
    Load a GuitarSet piece and extract tab events.

    Returns:
        List of tab events with time, duration, string, fret, midi_pitch
    """
    jam = jams.load(jams_path)

    # Find note_midi annotations
    note_annos = [a for a in jam.annotations if a.namespace == 'note_midi']

    if not note_annos:
        return []

    tab_events = []

    for anno in note_annos:
        data_source = anno.annotation_metadata.data_source

        if data_source not in GUITARSET_STRING_MAP:
            continue

        string_num, open_pitch = GUITARSET_STRING_MAP[data_source]

        for obs in anno.data:
            time = float(obs.time)
            duration = float(obs.duration)
            midi_pitch = int(obs.value)

            # Calculate fret
            fret = midi_pitch - open_pitch

            # Validate
            if 0 <= fret <= 24 and 20 <= midi_pitch <= 108:
                tab_events.append({
                    'time': time,
                    'duration': duration,
                    'string': string_num,
                    'fret': fret,
                    'midi_pitch': midi_pitch,
                })

    # Sort by time
    tab_events.sort(key=lambda x: x['time'])

    return tab_events


def load_synthtab_piece(jams_path: str) -> List[Dict]:
    """
    Load a SynthTab piece and extract tab events.

    SynthTab JAMS files use note_tab annotations with sandbox.string_index.

    Returns:
        List of tab events with time, duration, string, fret, midi_pitch
    """
    import json

    # Load JAMS as raw JSON (avoids validation errors with note_tab namespace)
    with open(jams_path, 'r') as f:
        jams_data = json.load(f)

    tab_events = []
    open_strings = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}

    # Find note_tab annotations
    for anno in jams_data.get('annotations', []):
        if anno.get('namespace') != 'note_tab':
            continue

        # Get string number from sandbox
        string_num = anno.get('sandbox', {}).get('string_index')
        if string_num is None:
            continue

        # Extract notes
        for note in anno.get('data', []):
            time = float(note.get('time', 0))
            duration = float(note.get('duration', 0))
            fret = int(note.get('value', {}).get('fret', 0))

            # Calculate MIDI pitch
            if string_num in open_strings:
                open_pitch = open_strings[string_num]
                midi_pitch = open_pitch + fret

                if 0 <= fret <= 24 and 20 <= midi_pitch <= 108:
                    tab_events.append({
                        'time': time,
                        'duration': duration,
                        'string': string_num,
                        'fret': fret,
                        'midi_pitch': midi_pitch,
                    })

    # Sort by time
    tab_events.sort(key=lambda x: x['time'])

    return tab_events


def create_encoder_decoder_tokens(
    tab_events: List[Dict],
    capo: int = 0,
    tuning: str = "64,59,55,50,45,40"
) -> Tuple[List[str], List[str]]:
    """
    Create encoder and decoder tokens from tab events.

    Returns:
        (encoder_tokens, decoder_tokens) tuple
    """
    encoder_tokens = [f"CAPO<{capo}>", f"TUNING<{tuning}>"]
    decoder_tokens = []

    for event in tab_events:
        midi_pitch = event['midi_pitch']
        duration_ms = int(round(event['duration'] * 1000))
        duration_ms = min(duration_ms, 5000)  # Cap at 5 seconds
        duration_ms = int(round(duration_ms / 100)) * 100  # Quantize to 100ms

        # Ensure minimum duration
        if duration_ms == 0:
            duration_ms = 100

        # Encoder tokens
        encoder_tokens.extend([
            f"NOTE_ON<{midi_pitch}>",
            f"TIME_SHIFT<{duration_ms}>",
            f"NOTE_OFF<{midi_pitch}>"
        ])

        # Decoder tokens (ground truth)
        decoder_tokens.extend([
            f"TAB<{event['string']},{event['fret']}>",
            f"TIME_SHIFT<{duration_ms}>"
        ])

    decoder_tokens.append("<eos>")

    return encoder_tokens, decoder_tokens


def main(dataset: str = "guitarset"):
    """Main evaluation loop.

    Args:
        dataset: "guitarset" or "synthtab"
    """
    print("=" * 80)
    print(f"Post-Processing Evaluation on {dataset.upper()}")
    print("=" * 80)

    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    checkpoint_path = "checkpoints_conditioning_scratch_retrain/checkpoint-642982"

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return

    tokenizer = MidiTabTokenizerV3.load("universal_tokenizer")
    tokenizer.ensure_conditioning_tokens(
        capo_values=tuple(range(8)),
        tuning_options=DEFAULT_CONDITIONING_TUNINGS
    )

    model = T5ForConditionalGeneration.from_pretrained(checkpoint_path)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        print("Using CUDA")
    else:
        print("Using CPU")

    # Load dataset files
    if dataset.lower() == "guitarset":
        guitarset_dir = Path("/data/akshaj/MusicAI/GuitarSet/annotation")

        if not guitarset_dir.exists():
            print(f"ERROR: GuitarSet directory not found at {guitarset_dir}")
            return

        jams_files = list(guitarset_dir.glob("*.jams"))

        if not jams_files:
            print(f"ERROR: No JAMS files found in {guitarset_dir}")
            return

        print(f"Found {len(jams_files)} GuitarSet pieces")
        load_piece_fn = load_guitarset_piece

    elif dataset.lower() == "synthtab":
        # Load SynthTab validation set from pre-made split
        from src.fret_t5.data import DataConfig, SynthTabTokenDataset

        data_config = DataConfig(
            max_encoder_length=512,
            max_decoder_length=512,
            enable_conditioning=True,
            conditioning_capo_values_eval=(0,),
            conditioning_tunings_eval=(STANDARD_TUNING,),
        )

        print("Loading SynthTab validation dataset from pre-made split...")
        val_dataset = SynthTabTokenDataset(
            tokenizer=tokenizer,
            manifests=[Path("data/synthtab_acoustic_val.jsonl")],
            data_config=data_config,
            split="val",
            preload=True
        )

        if len(val_dataset) == 0:
            print(f"ERROR: No examples in SynthTab val dataset")
            return

        print(f"Found {len(val_dataset)} SynthTab validation examples")

        # For SynthTab, we'll iterate through the dataset directly
        jams_files = None  # Signal to use dataset instead
        synthtab_dataset = val_dataset

    else:
        print(f"ERROR: Unknown dataset '{dataset}'. Use 'guitarset' or 'synthtab'")
        return

    # Evaluate on entire dataset
    if dataset.lower() == "synthtab":
        num_pieces = len(synthtab_dataset)  # Entire validation set
        data_source = synthtab_dataset
    else:
        num_pieces = len(jams_files)  # All ~360 GuitarSet files
        data_source = jams_files

    print(f"\nEvaluating on {num_pieces} pieces...")

    all_original_metrics = []
    all_postprocessed_metrics = []

    for i in range(num_pieces):
        if dataset.lower() == "synthtab":
            # For SynthTab, get example from dataset
            example = synthtab_dataset[i]
            print(f"\n[{i+1}/{num_pieces}] Processing SynthTab example {i}...")

            # Extract encoder and decoder tokens
            encoder_ids = example['input_ids'].tolist()
            label_ids = example['labels'].tolist()

            # Filter out padding tokens
            encoder_ids = [id for id in encoder_ids if id != tokenizer.shared_token_to_id["<pad>"]]
            label_ids = [id for id in label_ids if id != tokenizer.shared_token_to_id["<pad>"] and id != -100]

            encoder_tokens = tokenizer.shared_to_encoder_tokens(encoder_ids)
            ground_truth_tokens = tokenizer.shared_to_decoder_tokens(label_ids)

            print(f"  Loaded {len(extract_input_notes(encoder_tokens))} notes")

            # Encode input for model
            input_ids = torch.tensor(encoder_ids, dtype=torch.long).unsqueeze(0)

            if torch.cuda.is_available():
                input_ids = input_ids.cuda()

            # Generate prediction
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_length=512,
                    num_beams=1,
                    do_sample=False,
                    eos_token_id=tokenizer.shared_token_to_id["<eos>"],
                    pad_token_id=tokenizer.shared_token_to_id["<pad>"],
                )

            # Decode prediction
            pred_ids = outputs[0].cpu().tolist()
            pred_tokens = tokenizer.shared_to_decoder_tokens(pred_ids)

        else:
            # For GuitarSet, load from JAMS file
            jams_path = jams_files[i]
            print(f"\n[{i+1}/{num_pieces}] Processing {jams_path.name}...")

            # Load piece using appropriate loader
            tab_events = load_piece_fn(str(jams_path))

            if not tab_events:
                print(f"  Skipping (no valid events)")
                continue

            print(f"  Loaded {len(tab_events)} notes")

            # Create tokens
            capo = 0
            encoder_tokens, ground_truth_tokens = create_encoder_decoder_tokens(tab_events, capo)

            # Encode input
            encoder_ids = tokenizer.encode_encoder_tokens_shared(encoder_tokens)
            input_ids = torch.tensor(encoder_ids, dtype=torch.long).unsqueeze(0)

            if torch.cuda.is_available():
                input_ids = input_ids.cuda()

            # Generate prediction
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_length=512,
                    num_beams=1,
                    do_sample=False,
                    eos_token_id=tokenizer.shared_token_to_id["<eos>"],
                    pad_token_id=tokenizer.shared_token_to_id["<pad>"],
                )

            # Decode prediction
            pred_ids = outputs[0].cpu().tolist()
            pred_tokens = tokenizer.shared_to_decoder_tokens(pred_ids)

        capo, tuning = extract_conditioning_from_encoder(encoder_tokens)

        # Compute original metrics (pass ground_truth_tokens if available)
        original_metrics = compute_accuracy_metrics(
            encoder_tokens, pred_tokens, capo, tuning,
            ground_truth_tokens=ground_truth_tokens
        )
        all_original_metrics.append(original_metrics)

        # Check if we should enable debug mode (for pieces with errors)
        has_errors = (original_metrics['pitch_accuracy'] < 100 or
                     original_metrics['time_shift_accuracy'] < 100)


        postprocessed_tokens, correction_stats = postprocess_predictions(
            encoder_tokens, pred_tokens, capo, tuning, pitch_window=5, alignment_window=5, debug=has_errors
        )

        # Compute post-processed metrics (pass ground_truth_tokens if available)
        postprocessed_metrics = compute_accuracy_metrics(
            encoder_tokens, postprocessed_tokens, capo, tuning,
            ground_truth_tokens=ground_truth_tokens
        )
        all_postprocessed_metrics.append(postprocessed_metrics)

        # Print piece-level results
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


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    dataset = "guitarset"  # default
    if len(sys.argv) > 1:
        dataset = sys.argv[1]

    main(dataset)
