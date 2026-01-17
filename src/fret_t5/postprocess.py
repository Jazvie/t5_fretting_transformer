"""
Post-processing for tablature predictions.

This module implements the post-processing algorithm described in section 3.5:
- Corrects pitch errors by matching to nearest valid fingering within ±N MIDI notes
- Corrects time shift discrepancies to match input
- Uses fret_stretch metric to select best alternative fingering

Additionally provides timing reconstruction functionality for audio-to-tab pipelines:
- Preserves original continuous timing from MIDI input through quantized model inference
- Reconstructs absolute timestamps for tab events during postprocessing
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Sequence


# ---------------------------------------------------------------------------
# Timing Context for preserving original MIDI timing through inference
# ---------------------------------------------------------------------------

@dataclass
class NoteTimingInfo:
    """Original timing information for a single note from MIDI input.
    
    Attributes:
        onset_sec: Original onset time in seconds from MIDI
        duration_sec: Original duration in seconds from MIDI  
        pitch: MIDI pitch (for alignment verification)
        quantized_duration_ms: The quantized TIME_SHIFT value used in tokens
    """
    onset_sec: float
    duration_sec: float
    pitch: int
    quantized_duration_ms: int = 0


@dataclass 
class TimingContext:
    """Carries original MIDI timing through the inference pipeline.
    
    This allows postprocessing to reconstruct continuous timestamps
    after the model generates quantized TIME_SHIFT tokens.
    
    Attributes:
        note_timings: List of NoteTimingInfo, one per input note in sequence order
        time_shift_quantum_ms: The quantization step used (default 100ms)
    """
    note_timings: List[NoteTimingInfo] = field(default_factory=list)
    time_shift_quantum_ms: int = 100
    
    def __len__(self) -> int:
        return len(self.note_timings)
    
    def add_note(
        self, 
        onset_sec: float, 
        duration_sec: float, 
        pitch: int,
        quantized_duration_ms: int = 0
    ) -> None:
        """Add a note's timing information."""
        self.note_timings.append(NoteTimingInfo(
            onset_sec=onset_sec,
            duration_sec=duration_sec,
            pitch=pitch,
            quantized_duration_ms=quantized_duration_ms
        ))


@dataclass
class TabEvent:
    """A single tablature event with continuous timing.
    
    Attributes:
        string: Guitar string number (1-6, where 1 is high E)
        fret: Fret number (0-24)
        onset_sec: Onset time in seconds (from original MIDI)
        duration_sec: Duration in seconds (from original MIDI)
        midi_pitch: The MIDI pitch this tab position produces
    """
    string: int
    fret: int
    onset_sec: float
    duration_sec: float
    midi_pitch: int = 0

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
) -> List[Tuple[Optional[int], int]]:
    """
    Align input notes to output tabs using a sliding window.

    For each output position, finds the best matching input within ±window_size positions.

    Args:
        input_notes: List of (midi_pitch, time_shift) tuples
        output_tabs: List of (string, fret, time_shift) tuples
        window_size: Maximum position difference for matching

    Returns:
        List of (input_idx, output_idx) pairs. input_idx may be None for unmatched outputs,
        but output_idx is always a valid index.
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


# ---------------------------------------------------------------------------
# Timing-Aware Postprocessing for Audio-to-Tab Pipelines
# ---------------------------------------------------------------------------

def midi_notes_to_encoder_tokens_with_timing(
    midi_notes: List[Dict],
    time_shift_quantum_ms: int = 100,
    max_duration_ms: int = 5000,
) -> Tuple[List[str], TimingContext]:
    """Convert MIDI notes to encoder tokens while preserving original timing.
    
    This is the key function for the audio-to-tab pipeline. It:
    1. Sorts notes by onset time
    2. Detects chords (notes with same onset)
    3. Creates quantized encoder tokens for the model
    4. Preserves original continuous timing in TimingContext
    
    Args:
        midi_notes: List of dicts with keys:
            - 'pitch': MIDI pitch (int, 0-127)
            - 'start': onset time in seconds (float)
            - 'duration': duration in seconds (float)
            Or alternatively:
            - 'pitch': MIDI pitch
            - 'onset': onset time in seconds (alias for 'start')
            - 'offset': end time in seconds (duration = offset - onset)
        time_shift_quantum_ms: Quantization step in milliseconds (default 100)
        max_duration_ms: Maximum duration cap in milliseconds (default 5000)
        
    Returns:
        Tuple of (encoder_tokens, timing_context):
        - encoder_tokens: List of NOTE_ON, TIME_SHIFT, NOTE_OFF token strings
        - timing_context: TimingContext with original timing for each note
        
    Example:
        >>> notes = [
        ...     {'pitch': 60, 'start': 0.0, 'duration': 0.5},
        ...     {'pitch': 64, 'start': 0.0, 'duration': 0.5},  # chord with above
        ...     {'pitch': 67, 'start': 0.55, 'duration': 0.3},
        ... ]
        >>> tokens, timing = midi_notes_to_encoder_tokens_with_timing(notes)
        >>> # tokens ready for model, timing preserves original 0.0, 0.0, 0.55 onsets
    """
    # Normalize note format
    normalized_notes = []
    for note in midi_notes:
        pitch = int(note['pitch'])
        
        # Handle different key names for onset time
        if 'start' in note:
            onset = float(note['start'])
        elif 'onset' in note:
            onset = float(note['onset'])
        else:
            raise ValueError("Note must have 'start' or 'onset' key")
        
        # Handle different key names for duration
        if 'duration' in note:
            duration = float(note['duration'])
        elif 'offset' in note:
            duration = float(note['offset']) - onset
        else:
            raise ValueError("Note must have 'duration' or 'offset' key")
            
        normalized_notes.append({
            'pitch': pitch,
            'onset': onset,
            'duration': duration
        })
    
    # Sort by onset time, then by pitch for consistent ordering
    sorted_notes = sorted(normalized_notes, key=lambda n: (n['onset'], n['pitch']))
    
    # Group notes by onset time to detect chords
    # Notes within 10ms of each other are considered simultaneous
    CHORD_THRESHOLD_SEC = 0.01
    
    onset_groups: List[List[Dict]] = []
    current_group: List[Dict] = []
    current_onset: Optional[float] = None
    
    for note in sorted_notes:
        if current_onset is None or abs(note['onset'] - current_onset) <= CHORD_THRESHOLD_SEC:
            current_group.append(note)
            if current_onset is None:
                current_onset = note['onset']
        else:
            if current_group:
                onset_groups.append(current_group)
            current_group = [note]
            current_onset = note['onset']
    
    if current_group:
        onset_groups.append(current_group)
    
    # Build encoder tokens and timing context
    encoder_tokens: List[str] = []
    timing_context = TimingContext(time_shift_quantum_ms=time_shift_quantum_ms)
    
    for group_idx, group in enumerate(onset_groups):
        for note_idx, note in enumerate(group):
            pitch = note['pitch']
            onset = note['onset']
            duration = note['duration']
            
            # Determine if this is the last note in a chord group
            is_chord_note = len(group) > 1
            is_last_in_chord = note_idx == len(group) - 1
            
            # Calculate duration for TIME_SHIFT token
            duration_ms = duration * 1000
            duration_ms = min(duration_ms, max_duration_ms)
            
            # For chord notes (except last), use TIME_SHIFT<0>
            if is_chord_note and not is_last_in_chord:
                quantized_ms = 0
            else:
                # Quantize to nearest step
                quantized_ms = int(round(duration_ms / time_shift_quantum_ms)) * time_shift_quantum_ms
                # Ensure minimum duration for non-chord notes
                if quantized_ms == 0:
                    quantized_ms = time_shift_quantum_ms
            
            # Add encoder tokens
            encoder_tokens.append(f"NOTE_ON<{pitch}>")
            encoder_tokens.append(f"TIME_SHIFT<{quantized_ms}>")
            encoder_tokens.append(f"NOTE_OFF<{pitch}>")
            
            # Preserve original timing
            timing_context.add_note(
                onset_sec=onset,
                duration_sec=duration,
                pitch=pitch,
                quantized_duration_ms=quantized_ms
            )
    
    return encoder_tokens, timing_context


def postprocess_with_timing(
    encoder_tokens: List[str],
    decoder_tokens: List[str],
    timing_context: TimingContext,
    capo: int = 0,
    tuning: Tuple[int, ...] = STANDARD_TUNING,
    pitch_window: int = 5,
    alignment_window: int = 5,
) -> List[TabEvent]:
    """Postprocess model output and reconstruct original timing.
    
    This is the main function for getting tabs with continuous timing from
    the audio-to-tab pipeline. It:
    1. Applies standard pitch correction
    2. Aligns output tabs to input notes  
    3. Reconstructs original continuous timing from TimingContext
    
    Args:
        encoder_tokens: Input encoder tokens (may include CAPO/TUNING prefix)
        decoder_tokens: Model's predicted decoder tokens
        timing_context: TimingContext with original MIDI timing
        capo: Capo position (0-7)
        tuning: Guitar tuning as tuple of 6 MIDI pitches
        pitch_window: Max pitch difference for correction (semitones)
        alignment_window: Window size for sequence alignment
        
    Returns:
        List of TabEvent objects with:
        - string, fret: The tablature position
        - onset_sec: Original onset time from MIDI (continuous)
        - duration_sec: Original duration from MIDI (continuous)
        - midi_pitch: The MIDI pitch produced by this tab position
        
    Example:
        >>> # After model inference
        >>> tab_events = postprocess_with_timing(
        ...     encoder_tokens, decoder_tokens, timing_context
        ... )
        >>> for event in tab_events:
        ...     print(f"String {event.string}, Fret {event.fret} at {event.onset_sec:.3f}s")
    """
    # Extract notes and tabs using existing functions
    input_notes = extract_input_notes(encoder_tokens)
    output_tabs = extract_output_tabs(decoder_tokens)
    
    if len(output_tabs) == 0:
        return []
    
    # Validate timing context matches input notes
    if len(timing_context.note_timings) != len(input_notes):
        raise ValueError(
            f"Timing context has {len(timing_context.note_timings)} notes but "
            f"encoder tokens have {len(input_notes)} notes. "
            f"These must match for timing reconstruction."
        )
    
    # Align sequences
    alignments = align_sequences_with_window(input_notes, output_tabs, alignment_window)
    
    # Build tab events with reconstructed timing
    tab_events: List[TabEvent] = []
    
    for input_idx, output_idx in alignments:
        out_string, out_fret, out_time_shift = output_tabs[output_idx]
        
        # Skip outputs with no matching input (model generated more tabs than input notes)
        # This can happen when the model hallucinates extra chord notes
        if input_idx is None:
            continue
        
        if input_idx >= len(timing_context.note_timings):
            raise ValueError(
                f"Timing context mismatch: input_idx={input_idx} but timing_context "
                f"only has {len(timing_context.note_timings)} entries. "
                f"This indicates a bug in the tokenization pipeline."
            )
        
        # Apply pitch correction
        input_pitch, _ = input_notes[input_idx]
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
            else:
                corrected_string, corrected_fret = out_string, out_fret
        else:
            corrected_string, corrected_fret = out_string, out_fret
        
        # Reconstruct timing from TimingContext
        timing_info = timing_context.note_timings[input_idx]
        onset_sec = timing_info.onset_sec
        duration_sec = timing_info.duration_sec
        
        tab_events.append(TabEvent(
            string=corrected_string,
            fret=corrected_fret,
            onset_sec=onset_sec,
            duration_sec=duration_sec,
            midi_pitch=input_pitch
        ))
    
    return tab_events


def tab_events_to_dict_list(tab_events: List[TabEvent]) -> List[Dict]:
    """Convert TabEvent objects to list of dictionaries.
    
    Useful for serialization or integration with other systems.
    
    Args:
        tab_events: List of TabEvent objects
        
    Returns:
        List of dicts with keys: string, fret, onset_sec, duration_sec, midi_pitch
    """
    return [
        {
            'string': event.string,
            'fret': event.fret,
            'onset_sec': event.onset_sec,
            'duration_sec': event.duration_sec,
            'midi_pitch': event.midi_pitch,
        }
        for event in tab_events
    ]


def postprocess_to_timed_tabs(
    midi_notes: List[Dict],
    decoder_tokens: List[str],
    capo: int = 0,
    tuning: Tuple[int, ...] = STANDARD_TUNING,
    pitch_window: int = 5,
    alignment_window: int = 5,
    time_shift_quantum_ms: int = 100,
) -> List[TabEvent]:
    """Convenience function: postprocess decoder tokens using original MIDI notes.
    
    This combines midi_notes_to_encoder_tokens_with_timing and postprocess_with_timing
    into a single call for simpler integration.
    
    Args:
        midi_notes: Original MIDI notes with timing (as passed to inference)
        decoder_tokens: Model's predicted decoder tokens
        capo: Capo position (0-7)
        tuning: Guitar tuning
        pitch_window: Max pitch difference for correction
        alignment_window: Window size for alignment
        time_shift_quantum_ms: Quantization step used during tokenization
        
    Returns:
        List of TabEvent objects with reconstructed timing
        
    Example:
        >>> # Full pipeline usage
        >>> midi_notes = extract_notes_from_midi(midi_file)  # Your MIDI loader
        >>> encoder_tokens, timing = midi_notes_to_encoder_tokens_with_timing(midi_notes)
        >>> decoder_tokens = model.predict(encoder_tokens)  # Your model
        >>> tab_events = postprocess_to_timed_tabs(midi_notes, decoder_tokens)
    """
    # Recreate encoder tokens and timing context
    encoder_tokens, timing_context = midi_notes_to_encoder_tokens_with_timing(
        midi_notes,
        time_shift_quantum_ms=time_shift_quantum_ms,
    )
    
    return postprocess_with_timing(
        encoder_tokens=encoder_tokens,
        decoder_tokens=decoder_tokens,
        timing_context=timing_context,
        capo=capo,
        tuning=tuning,
        pitch_window=pitch_window,
        alignment_window=alignment_window,
    )
