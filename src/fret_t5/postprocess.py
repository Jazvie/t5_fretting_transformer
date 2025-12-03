"""
Post-processing for tablature predictions.

This module implements the post-processing algorithm described in section 3.5:
- Corrects pitch errors by matching to nearest valid fingering within ±N MIDI notes
- Corrects time shift discrepancies to match input
- Uses fret_stretch metric to select best alternative fingering
"""

from typing import List, Tuple, Optional, Dict

# Standard tuning (string 1 to string 6)
STANDARD_TUNING = (64, 59, 55, 50, 45, 40)


def tuning_to_open_strings(tuning: Tuple[int, ...]) -> Dict[int, int]:
    """Convert tuning tuple to open strings dictionary mapping string number to MIDI pitch."""
    return {i + 1: pitch for i, pitch in enumerate(tuning)}


def tab_to_midi_pitch(
    string: int,
    fret: int,
    capo: int = 0,
    tuning: Tuple[int, ...] = STANDARD_TUNING
) -> int:
    """
    Convert guitar tablature to MIDI pitch.

    Args:
        string: String number (1-6, where 1 is high E)
        fret: Fret number (0-24)
        capo: Capo position (0-7)
        tuning: Tuple of 6 open string pitches

    Returns:
        MIDI pitch value
    """
    open_strings = tuning_to_open_strings(tuning)
    return open_strings[string] + fret + capo


def fret_stretch(p_fret: int, q_fret: int) -> float:
    """
    Calculate fret stretch difficulty between two fret positions.

    Moving up the fretboard (positive delta) is easier (0.50x weight),
    moving down (negative delta) is harder (0.75x weight).

    Args:
        p_fret: Starting fret position
        q_fret: Ending fret position

    Returns:
        Fret stretch difficulty score
    """
    delta_fret = q_fret - p_fret
    if delta_fret > 0:
        return 0.50 * abs(delta_fret)
    else:
        return 0.75 * abs(delta_fret)


def find_alternative_fingerings(
    midi_pitch: int,
    capo: int = 0,
    tuning: Tuple[int, ...] = STANDARD_TUNING
) -> List[Tuple[int, int]]:
    """
    Find all valid (string, fret) combinations for a given MIDI pitch.

    Args:
        midi_pitch: Target MIDI pitch
        capo: Capo position (0-7)
        tuning: Tuple of 6 open string pitches

    Returns:
        List of (string, fret) tuples that produce the target pitch
    """
    alternatives = []
    open_strings = tuning_to_open_strings(tuning)

    for string in range(1, 7):
        open_pitch = open_strings[string]
        fret = midi_pitch - open_pitch - capo

        if 0 <= fret <= 24:
            alternatives.append((string, fret))

    return alternatives


def select_best_fingering(
    alternatives: List[Tuple[int, int]],
    predicted_string: int,
    predicted_fret: int
) -> Tuple[int, int]:
    """
    Select the best alternative fingering using fret_stretch metric.

    Chooses the fingering that minimizes fret stretch distance from
    the model's predicted fingering.

    Args:
        alternatives: List of valid (string, fret) combinations
        predicted_string: Model's predicted string
        predicted_fret: Model's predicted fret

    Returns:
        (string, fret) tuple with minimal fret stretch distance
    """
    if not alternatives:
        return (predicted_string, predicted_fret)

    if len(alternatives) == 1:
        return alternatives[0]

    # Calculate fret_stretch distance for each alternative
    best_fingering = alternatives[0]
    min_distance = fret_stretch(predicted_fret, alternatives[0][1])

    for string, fret in alternatives[1:]:
        distance = fret_stretch(predicted_fret, fret)
        if distance < min_distance:
            min_distance = distance
            best_fingering = (string, fret)

    return best_fingering


def parse_tab_token(tab_token: str) -> Optional[Tuple[int, int]]:
    """Parse TAB<string,fret> token to (string, fret) tuple."""
    try:
        content = tab_token[4:-1]  # Remove "TAB<" and ">"
        parts = content.split(',')
        if len(parts) == 2:
            return (int(parts[0]), int(parts[1]))
    except ValueError:
        return None
    return None


def parse_time_shift_token(time_shift_token: str) -> Optional[int]:
    """Parse TIME_SHIFT<ms> token to integer milliseconds."""
    try:
        content = time_shift_token[11:-1]  # Remove "TIME_SHIFT<" and ">"
        return int(content)
    except ValueError:
        return None
    return None


def parse_note_on_token(note_on_token: str) -> Optional[int]:
    """Parse NOTE_ON<pitch> token to integer MIDI pitch."""
    try:
        content = note_on_token[8:-1]  # Remove "NOTE_ON<" and ">"
        return int(content)
    except ValueError:
        return None
    return None


def parse_capo_token(capo_token: str) -> Optional[int]:
    """Parse CAPO<n> token to integer."""
    try:
        content = capo_token[5:-1]  # Remove "CAPO<" and ">"
        return int(content)
    except ValueError:
        return None
    return None


def parse_tuning_token(tuning_token: str) -> Optional[Tuple[int, ...]]:
    """Parse TUNING<p1,p2,p3,p4,p5,p6> token to tuple of pitches."""
    try:
        content = tuning_token[7:-1]  # Remove "TUNING<" and ">"
        pitches = tuple(int(p) for p in content.split(','))
        if len(pitches) == 6:
            return pitches
    except ValueError:
        return None
    return None


def extract_input_notes(encoder_tokens: List[str]) -> List[Tuple[int, int]]:
    """
    Extract (midi_pitch, time_shift) pairs from encoder tokens.

    Args:
        encoder_tokens: List of encoder tokens including NOTE_ON, TIME_SHIFT, NOTE_OFF

    Returns:
        List of (midi_pitch, time_shift_ms) tuples
    """
    notes = []
    i = 0

    # Skip conditioning tokens (CAPO and TUNING)
    while i < len(encoder_tokens):
        token = encoder_tokens[i]
        if token.startswith('CAPO<') or token.startswith('TUNING<'):
            i += 1
            continue
        break

    # Parse NOTE_ON, TIME_SHIFT, NOTE_OFF triplets
    while i < len(encoder_tokens) - 2:
        if encoder_tokens[i].startswith('NOTE_ON<'):
            pitch = parse_note_on_token(encoder_tokens[i])
            time_shift = parse_time_shift_token(encoder_tokens[i + 1])

            if pitch is not None and time_shift is not None:
                notes.append((pitch, time_shift))

            i += 3  # Skip NOTE_ON, TIME_SHIFT, NOTE_OFF
        else:
            i += 1

    return notes


def extract_output_tabs(decoder_tokens: List[str]) -> List[Tuple[int, int, int]]:
    """
    Extract (string, fret, time_shift) tuples from decoder tokens.

    Args:
        decoder_tokens: List of decoder tokens (TAB and TIME_SHIFT pairs)

    Returns:
        List of (string, fret, time_shift_ms) tuples
    """
    tabs = []
    i = 0

    while i < len(decoder_tokens) - 1:
        if decoder_tokens[i].startswith('TAB<'):
            tab = parse_tab_token(decoder_tokens[i])
            time_shift = parse_time_shift_token(decoder_tokens[i + 1])

            if tab is not None and time_shift is not None:
                string, fret = tab
                tabs.append((string, fret, time_shift))

            i += 2  # Skip TAB and TIME_SHIFT
        else:
            i += 1

    return tabs


def align_sequences_with_window(
    input_notes: List[Tuple[int, int]],
    output_tabs: List[Tuple[int, int, int]],
    window_size: int = 5
) -> List[Tuple[Optional[int], Optional[int]]]:
    """
    Align input notes to output tabs using a sliding window.

    For each output position, finds the best matching input within ±window_size positions.

    Args:
        input_notes: List of (midi_pitch, time_shift) tuples
        output_tabs: List of (string, fret, time_shift) tuples
        window_size: Maximum position difference for matching

    Returns:
        List of (input_idx, output_idx) pairs, with None for unmatched positions
    """
    # Simple 1-to-1 alignment when lengths match
    if len(input_notes) == len(output_tabs):
        return [(i, i) for i in range(len(input_notes))]

    # For length mismatches, use greedy nearest-neighbor alignment
    alignments = []
    used_inputs = set()

    for out_idx in range(len(output_tabs)):
        best_input_idx = None
        best_score = float('inf')

        # Search within window
        search_start = max(0, out_idx - window_size)
        search_end = min(len(input_notes), out_idx + window_size + 1)

        for in_idx in range(search_start, search_end):
            if in_idx in used_inputs:
                continue

            # Score = position difference (prefer nearby positions)
            score = abs(in_idx - out_idx)

            if score < best_score:
                best_score = score
                best_input_idx = in_idx

        if best_input_idx is not None:
            alignments.append((best_input_idx, out_idx))
            used_inputs.add(best_input_idx)
        else:
            # No match found, output will be left as-is
            alignments.append((None, out_idx))

    return alignments


def postprocess_decoder_tokens(
    encoder_tokens: List[str],
    decoder_tokens: List[str],
    capo: int = 0,
    tuning: Tuple[int, ...] = STANDARD_TUNING,
    pitch_window: int = 5,
    alignment_window: int = 5
) -> List[str]:
    """
    Apply post-processing to correct pitch and time shift errors.

    This function:
    1. Aligns input notes to output tabs using a sliding window
    2. Corrects pitch errors within ±pitch_window MIDI notes
    3. Matches time shifts from input to output
    4. Uses fret_stretch metric to select best alternative fingerings

    Args:
        encoder_tokens: Input MIDI tokens (NOTE_ON, TIME_SHIFT, NOTE_OFF sequence)
        decoder_tokens: Model's predicted tablature tokens (TAB, TIME_SHIFT pairs)
        capo: Capo position (0-7)
        tuning: Tuple of 6 open string pitches (string 1 to string 6)
        pitch_window: Only correct if pitch difference <= this value (MIDI notes)
        alignment_window: Position window for matching input to output notes

    Returns:
        Corrected decoder tokens with pitch and time shift corrections applied
    """
    # Extract notes and tabs
    input_notes = extract_input_notes(encoder_tokens)
    output_tabs = extract_output_tabs(decoder_tokens)

    if len(output_tabs) == 0:
        return decoder_tokens

    # Align sequences
    alignments = align_sequences_with_window(input_notes, output_tabs, alignment_window)

    # Apply corrections
    corrected_tokens = []

    for input_idx, output_idx in alignments:
        out_string, out_fret, out_time_shift = output_tabs[output_idx]

        # If no aligned input, keep output as-is
        if input_idx is None:
            corrected_tokens.append(f"TAB<{out_string},{out_fret}>")
            corrected_tokens.append(f"TIME_SHIFT<{out_time_shift}>")
            continue

        input_pitch, input_time_shift = input_notes[input_idx]

        # Calculate predicted MIDI pitch
        predicted_pitch = tab_to_midi_pitch(out_string, out_fret, capo, tuning)
        pitch_diff = abs(input_pitch - predicted_pitch)

        # Determine correct fingering
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
            else:
                # No valid alternative, keep original (shouldn't happen)
                corrected_string, corrected_fret = out_string, out_fret
        else:
            # Pitch difference too large, keep original
            corrected_string, corrected_fret = out_string, out_fret

        # Always use input time shift
        corrected_time_shift = input_time_shift

        # Add corrected tokens
        corrected_tokens.append(f"TAB<{corrected_string},{corrected_fret}>")
        corrected_tokens.append(f"TIME_SHIFT<{corrected_time_shift}>")

    # Add EOS token if present in original
    if decoder_tokens and decoder_tokens[-1] == "<eos>":
        corrected_tokens.append("<eos>")

    return corrected_tokens
