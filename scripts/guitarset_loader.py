#!/usr/bin/env python3
"""
GuitarSet data loader and tokenizer adapter.
Converts GuitarSet JAMS note_midi annotations to our tablature format.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fret_t5 import TokenizerConfig


def load_guitarset_jams(jams_path: Path) -> Dict:
    """Load and parse a GuitarSet JAMS file."""
    with open(jams_path, 'r') as f:
        return json.load(f)


def hz_to_midi_int(f):
    """Convert frequency in Hz to MIDI note number.

    Args:
        f: Frequency in Hz

    Returns:
        MIDI note number (int) or None if invalid
    """
    import math
    if f <= 0 or not math.isfinite(f):
        return None
    return int(round(69 + 12 * math.log2(f / 440.0)))


def extract_midi_notes_from_guitarset(jams_data: Dict) -> List[Dict]:
    """Extract MIDI note events from GuitarSet JAMS annotations.

    GuitarSet stores pitch data as frequencies in Hz within pitch_contour annotations.
    We segment these contours into discrete notes and convert to MIDI note numbers.

    Returns:
        List of note events with time (seconds), duration (seconds), pitch (MIDI)
    """
    notes = []

    for ann in jams_data.get("annotations", []):
        if ann.get("namespace") != "pitch_contour":
            continue

        data = ann.get("data", [])

        frames = []
        # Handle both dict-of-arrays and list-of-dicts formats
        if isinstance(data, dict):
            # Dict format: {"time": [...], "value": [...]}
            times = data.get("time", [])
            values = data.get("value", [])
            for t, v in zip(times, values):
                # Accept dict {"frequency": ...} or direct float
                f_hz = v.get("frequency") if isinstance(v, dict) else v
                if f_hz is None:
                    continue
                midi = hz_to_midi_int(float(f_hz))
                if midi is not None:
                    frames.append((float(t), midi))
        elif isinstance(data, list):
            # List format: [{"time": ..., "value": ...}, ...]
            for d in data:
                t = d.get("time")
                v = d.get("value")
                if t is None or v is None:
                    continue
                f_hz = v.get("frequency") if isinstance(v, dict) else v
                midi = hz_to_midi_int(float(f_hz))
                if midi is not None:
                    frames.append((float(t), midi))

        if not frames:
            continue

        # Sort and segment frames into notes by stable MIDI & small gaps
        frames.sort()
        MAX_GAP_S = 0.06       # merge within 60 ms
        TOL_SEMITONES = 0.49   # treat +/- <0.5 semitone as same

        start_t = frames[0][0]
        last_t  = start_t
        last_m  = frames[0][1]

        def flush(s, e, m):
            # Range gate; guitar range with some headroom
            if 30 <= m <= 100 and e > s:
                notes.append({"time": s, "duration": e - s, "pitch": m})

        for t, m in frames[1:]:
            same_pitch = abs(m - last_m) <= TOL_SEMITONES
            small_gap  = (t - last_t) <= MAX_GAP_S
            if same_pitch and small_gap:
                last_t = t
            else:
                flush(start_t, last_t + MAX_GAP_S, last_m)
                start_t, last_t, last_m = t, t, m

        flush(start_t, last_t + MAX_GAP_S, last_m)

    # Fallback: note_midi if available
    if not notes:
        for ann in jams_data.get("annotations", []):
            if ann.get("namespace") == "note_midi":
                for d in ann.get("data", []):
                    t = float(d.get("time", 0))
                    dur = float(d.get("duration", 0.1))
                    val = d.get("value", 0)
                    # Detect Hz vs MIDI
                    if isinstance(val, (int, float)):
                        midi = val if (0 <= val <= 127 and float(val).is_integer()) else hz_to_midi_int(float(val))
                        if midi is not None and 0 <= midi <= 127:
                            notes.append({"time": t, "duration": dur, "pitch": int(midi)})

    # Sort by time and validate
    notes.sort(key=lambda x: x["time"])

    # Data validation - fail fast on invalid ranges
    for note in notes:
        assert 0 <= note["pitch"] <= 127, f"Out-of-range MIDI {note['pitch']}"

    # Debug logging
    if notes:
        pitches = [n['pitch'] for n in notes]
        durations = [n['duration'] for n in notes]
        print(f"   Extracted {len(notes)} notes from GuitarSet")
        print(f"   Pitch range: {min(pitches)} to {max(pitches)} (MIDI)")
        print(f"   Duration range: {min(durations):.3f}s to {max(durations):.3f}s")
        sample_notes = [(n['pitch'], f"{n['duration']:.3f}s") for n in notes[:3]]
        print(f"   Sample notes: {sample_notes}")

    return notes


def debug_guitarset_string_mapping(jams_data: Dict):
    """Debug function to verify GuitarSet string mapping is correct."""
    note_midi_annotations = [ann for ann in jams_data.get("annotations", [])
                            if ann.get("namespace") == "note_midi"]

    print(f"Debug: Found {len(note_midi_annotations)} note_midi annotations")

    for i, annotation in enumerate(note_midi_annotations):
        data_source = annotation.get("annotation_metadata", {}).get("data_source", "")
        data_count = len(annotation.get("data", []))

        if data_count > 0:
            pitches = [round(note["value"]) for note in annotation.get("data", [])]
            min_pitch, max_pitch = min(pitches), max(pitches)
            print(f"  annotation {i}: data_source='{data_source}', notes={data_count}, pitch_range={min_pitch}-{max_pitch}")


def extract_tablature_from_guitarset_jams(
    jams_data: Dict, 
    auto_detect_tuning: bool = False,
    inverted_string_convention: bool = False,
) -> List[Dict]:
    """Extract tablature events directly from GuitarSet JAMS per-string annotations.

    Uses the 6 separate note_midi annotations to get ground truth string assignments.
    GuitarSet data_source field maps to strings as follows:
    - data_source "0" -> String 6 (Low E) [or String 1 if inverted]
    - data_source "1" -> String 5 (A)     [or String 2 if inverted]
    - data_source "2" -> String 4 (D)     [or String 3 if inverted]
    - data_source "3" -> String 3 (G)     [or String 4 if inverted]
    - data_source "4" -> String 2 (B)     [or String 5 if inverted]
    - data_source "5" -> String 1 (High E)[or String 6 if inverted]

    Args:
        jams_data: GuitarSet JAMS data
        auto_detect_tuning: If True, detect actual tuning from data instead of assuming standard.
                           NOTE: GuitarSet is in standard tuning, so this should be False.
                           The auto-detection is flawed as it uses min(pitches) which finds
                           the lowest note played, not the open string pitch.
        inverted_string_convention: If True, use inverted string numbering where
                           String 1 = Low E (40) instead of High E (64).
                           Default False uses standard guitar convention.
    """
    tab_events = []

    # String mapping depends on convention choice
    # Standard: String 1 = High E (64), String 6 = Low E (40)
    # Inverted: String 1 = Low E (40), String 6 = High E (64)
    if inverted_string_convention:
        standard_tuning = {
            "0": (1, 40),  # Low E  -> String 1 (inverted)
            "1": (2, 45),  # A      -> String 2
            "2": (3, 50),  # D      -> String 3
            "3": (4, 55),  # G      -> String 4
            "4": (5, 59),  # B      -> String 5
            "5": (6, 64)   # High E -> String 6 (inverted)
        }
    else:
        # Standard guitar string mapping to match SynthTab training data conventions
        # CRITICAL: Must match SynthTab's string_index mapping for consistent MIDI-to-TAB tokens
        standard_tuning = {
            "0": (6, 40),  # Low E  -> String 6 (standard)
            "1": (5, 45),  # A      -> String 5
            "2": (4, 50),  # D      -> String 4
            "3": (3, 55),  # G      -> String 3
            "4": (2, 59),  # B      -> String 2
            "5": (1, 64)   # High E -> String 1 (standard)
        }

    # Find all note_midi annotations (should be 6, one per string)
    note_midi_annotations = [ann for ann in jams_data.get("annotations", [])
                            if ann.get("namespace") == "note_midi"]

    if len(note_midi_annotations) != 6:
        print(f"Warning: Expected 6 note_midi annotations, found {len(note_midi_annotations)}")
        return []

    # Auto-detect tuning if enabled
    detected_tuning = {}
    if auto_detect_tuning:
        print("   🔍 Auto-detecting guitar tuning...")
        for annotation in note_midi_annotations:
            data_source = annotation.get("annotation_metadata", {}).get("data_source", "")
            if data_source not in standard_tuning:
                continue

            notes_data = annotation.get("data", [])
            if not notes_data:
                continue

            # GuitarSet stores MIDI note numbers directly, not frequencies
            midi_pitches = []
            for note_data in notes_data:
                midi_val = note_data["value"]
                # GuitarSet values are already MIDI notes (typically 28-84 for guitar)
                midi_pitch = round(midi_val)  # Round to nearest integer

                if 20 <= midi_pitch <= 108:  # Guitar range
                    midi_pitches.append(midi_pitch)

            if midi_pitches:
                string_num = standard_tuning[data_source][0]
                detected_open = min(midi_pitches)  # Lowest note likely open string
                standard_open = standard_tuning[data_source][1]

                tuning_offset = detected_open - standard_open
                detected_tuning[data_source] = (string_num, detected_open, tuning_offset)

                tuning_diff = detected_open - standard_open
                status = "OK" if abs(tuning_diff) <= 1 else "WARN"
                print(f"     {status} String {string_num}: detected {detected_open} vs standard {standard_open} (diff: {tuning_diff:+d})")

    # Use detected tuning if available, otherwise fall back to standard
    actual_tuning = detected_tuning if detected_tuning else standard_tuning

    # Process each string's annotations
    for annotation in note_midi_annotations:
        data_source = annotation.get("annotation_metadata", {}).get("data_source", "")

        # Skip annotations without proper data_source or with empty data
        if data_source not in actual_tuning or not annotation.get("data"):
            continue

        if auto_detect_tuning and data_source in detected_tuning:
            string_num, _, _ = detected_tuning[data_source]
            open_pitch = standard_tuning[data_source][1]  # Use standard open pitch for calculations
        else:
            string_num, open_pitch = actual_tuning[data_source]

        for note_data in annotation.get("data", []):
            time = float(note_data["time"])
            duration = float(note_data["duration"])
            midi_val = note_data["value"]

            # GuitarSet stores MIDI note numbers directly
            midi_pitch = round(midi_val)  # Round to nearest integer

            # Validate MIDI pitch is in reasonable guitar range
            if not (20 <= midi_pitch <= 108):
                continue

            # Apply tuning correction for consistency with SynthTab standard tuning
            # This normalizes all files to standard tuning to prevent MIDI-to-TAB conflicts
            corrected_midi = midi_pitch  # Default to original

            if auto_detect_tuning and data_source in detected_tuning:
                # Use detected tuning with correction to standard
                _, detected_open, tuning_offset = detected_tuning[data_source]
                corrected_midi = midi_pitch - tuning_offset  # Normalize to standard tuning
                fret = corrected_midi - open_pitch  # Calculate fret from standard tuning
            else:
                # Use original calculation for standard tuning
                fret = midi_pitch - open_pitch

            # Validate fret range
            if -2 <= fret <= 24:
                # Clamp negative frets to 0 (treat as open string)
                fret = max(0, fret)

                tab_events.append({
                    'time': time,
                    'duration': duration,
                    'string': string_num,
                    'fret': fret,
                    'midi_pitch': corrected_midi,  # Use corrected MIDI for consistency
                    'original_value': midi_val,  # Keep original for debugging
                    'original_midi': midi_pitch,  # Keep original MIDI for debugging
                    'tuning_corrected': auto_detect_tuning and data_source in detected_tuning
                })
            else:
                # Still warn about extreme outliers
                if abs(fret) > 10:  # Only warn for major outliers
                    print(f"Warning: Extreme fret {fret} for string {string_num}, pitch {midi_pitch}")

    # Sort by time
    tab_events.sort(key=lambda x: x['time'])

    if tab_events and auto_detect_tuning:
        print(f"   Extracted {len(tab_events)} valid tablature events")

    return tab_events



def quantize_duration(duration_ms: float, quantum_ms: int = 100, max_duration_ms: int = 5000) -> int:
    """Apply same quantization as SynthTab tokenizer.

    Uses linear quantization strategy:
    quantized = round(duration_ms / quantum_ms) * quantum_ms
    """
    # Cap at maximum duration
    duration_ms = min(duration_ms, max_duration_ms)

    # Linear quantization (same as SynthTab)
    return int(round(duration_ms / quantum_ms)) * quantum_ms


def convert_to_tokens(tab_events: List[Dict]) -> Tuple[List[str], List[str]]:
    """Convert tablature events to encoder/decoder token sequences with proper quantization."""

    encoder_tokens = []  # MIDI-style tokens for input
    decoder_tokens = []  # TAB tokens for output

    for event in tab_events:
        # Apply quantization (same as SynthTab tokenizer)
        duration_ms_raw = event['duration'] * 1000
        duration_ms_quantized = quantize_duration(duration_ms_raw)

        # Encoder: NOTE_ON/TIME_SHIFT/NOTE_OFF sequence
        pitch = (event['string'] - 1) * 25 + event['fret'] + 40  # Reconstruct pitch

        encoder_tokens.extend([
            f"NOTE_ON<{pitch}>",
            f"TIME_SHIFT<{duration_ms_quantized}>",
            f"NOTE_OFF<{pitch}>"
        ])

        # Decoder: TAB<string,fret> TIME_SHIFT<duration>
        decoder_tokens.extend([
            f"TAB<{event['string']},{event['fret']}>",
            f"TIME_SHIFT<{duration_ms_quantized}>"
        ])

    return encoder_tokens, decoder_tokens


def test_guitarset_tokenization(guitarset_dir: Path):
    """Test GuitarSet data loading and tokenization."""

    print("Testing GuitarSet Tokenization")
    print("=" * 50)

    # 1. Find a GuitarSet JAMS file
    jams_files = list(guitarset_dir.glob("*.jams"))

    if not jams_files:
        print("ERROR: No GuitarSet JAMS files found!")
        return

    print(f"Found {len(jams_files)} GuitarSet JAMS files")

    # Test with first 3 files
    for i, jams_file in enumerate(jams_files[:3]):
        print(f"\nTesting file {i+1}: {jams_file.name}")
        print("-" * 40)

        try:
            # Load JAMS data
            jams_data = load_guitarset_jams(jams_file)
            print(f"   JAMS loaded successfully")

            # Extract tablature using ground truth string assignments
            tab_events = extract_tablature_from_guitarset_jams(jams_data, auto_detect_tuning=False)
            print(f"   Found {len(tab_events)} tablature events with ground truth strings")

            if len(tab_events) == 0:
                print("   WARNING: No tablature events found in this file")
                continue

            # Also extract MIDI notes for comparison
            midi_notes = extract_midi_notes_from_guitarset(jams_data)
            print(f"   Found {len(midi_notes)} MIDI notes")

            # Show first few notes
            print(f"   First 5 tablature events:")
            for j, tab in enumerate(tab_events[:5]):
                print(f"     {j+1}. String: {tab['string']}, Fret: {tab['fret']}, Time: {tab['time']:.3f}s")

            # Use first 20 events for token conversion test
            tab_events = tab_events[:20]
            print(f"   Using first {len(tab_events)} tablature events for tokenization test")

            # Convert to tokens
            encoder_tokens, decoder_tokens = convert_to_tokens(tab_events)
            print(f"   Generated {len(encoder_tokens)} encoder tokens, {len(decoder_tokens)} decoder tokens")

            # Show quantization examples
            print(f"   Duration quantization examples:")
            for j, tab in enumerate(tab_events[:3]):
                raw_ms = tab['duration'] * 1000
                quantized_ms = quantize_duration(raw_ms)
                print(f"     {j+1}. {raw_ms:.0f}ms → {quantized_ms}ms")

            # Show token examples
            print(f"   Encoder tokens (first 10): {encoder_tokens[:10]}")
            print(f"   Decoder tokens (first 10): {decoder_tokens[:10]}")

        except Exception as e:
            print(f"   ERROR: Error processing {jams_file.name}: {e}")
            import traceback
            traceback.print_exc()

    # 2. Test with our trained tokenizer
    print(f"\nTesting with Trained Tokenizer")
    print("-" * 40)

    try:
        from fret_t5 import MidiTabTokenizerV3

        tokenizer = MidiTabTokenizerV3.load("universal_tokenizer")
        print(f"   Tokenizer loaded: {len(tokenizer.shared_token_to_id)} vocab size")

        # Test tokenization on GuitarSet data
        if encoder_tokens and decoder_tokens:
            # Encode tokens
            encoder_ids = tokenizer.encode_encoder_tokens_shared(encoder_tokens[:50])
            decoder_ids = tokenizer.encode_decoder_tokens_shared(decoder_tokens[:50])

            print(f"   Encoded {len(encoder_ids)} encoder IDs, {len(decoder_ids)} decoder IDs")

            # Check for unknown tokens
            unk_id = tokenizer.shared_token_to_id["<unk>"]
            encoder_unks = sum(1 for id in encoder_ids if id == unk_id)
            decoder_unks = sum(1 for id in decoder_ids if id == unk_id)

            print(f"   Unknown tokens: {encoder_unks} encoder, {decoder_unks} decoder")

            if encoder_unks > 0 or decoder_unks > 0:
                print("   WARNING: Some tokens not in vocabulary - might need vocabulary expansion")
            else:
                print("   OK: All tokens successfully encoded!")

    except Exception as e:
        print(f"   ERROR: Error testing tokenizer: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test GuitarSet data loading and tokenization")
    parser.add_argument(
        "--guitarset-dir",
        type=str,
        required=True,
        help="Path to GuitarSet annotation directory"
    )
    args = parser.parse_args()

    test_guitarset_tokenization(Path(args.guitarset_dir))
