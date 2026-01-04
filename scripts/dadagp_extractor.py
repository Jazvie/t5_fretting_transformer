"""
DadaGP Extractor - Core GuitarPro parsing and note extraction.

Extracts tablature notes from GuitarPro files with:
- String/fret positions
- Timing in ticks
- MIDI pitch calculation from tuning
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Iterator
import logging

try:
    import guitarpro as gp
except ImportError:
    raise ImportError("PyGuitarPro is required. Install with: pip install PyGuitarPro")

logger = logging.getLogger(__name__)

# Standard 6-string guitar tuning (MIDI pitches): High E to Low E
# String 1 = High E (64), String 6 = Low E (40)
STANDARD_TUNING = (64, 59, 55, 50, 45, 40)

# Skip these derivative files
SKIP_PATTERNS = ['.tokens', '.pygp.gp5', '.gp2tokens2gp.gp5', '.tokens.txt']


@dataclass
class TabNote:
    """A single tablature note."""
    time_ticks: float           # Onset in ticks
    duration_ticks: float       # Duration in ticks
    string: int                 # 1-6 (1=high E, 6=low E)
    fret: int                   # 0-24 (can be negative for drop tunings)
    pitch: int                  # MIDI pitch
    velocity: int = 80          # Default velocity


@dataclass
class ExtractedTrack:
    """Extracted data from a GuitarPro track."""
    source_file: str
    track_index: int
    track_name: str
    instrument: int             # MIDI program number
    notes: List[TabNote] = field(default_factory=list)
    tuning: Tuple[int, ...] = field(default_factory=lambda: STANDARD_TUNING)
    tuning_strings: List[str] = field(default_factory=list)  # e.g., ['E5', 'B4', ...]
    tempo: float = 120.0
    is_standard_tuning: bool = True
    tuning_offset: int = 0      # Global semitone offset from standard


@dataclass
class TempoChange:
    """A tempo change event."""
    tick: float
    tempo: float  # BPM


def find_guitarpro_files(root_dir: str,
                         extensions: Tuple[str, ...] = ('.gp3', '.gp4', '.gp5', '.gpx')) -> Iterator[str]:
    """
    Find all GuitarPro files in a directory tree.

    Skips derivative files (tokens, pygp conversions, etc.)
    """
    root_path = Path(root_dir)

    for filepath in root_path.rglob('*'):
        if not filepath.is_file():
            continue

        # Check extension
        if not any(filepath.name.lower().endswith(ext) for ext in extensions):
            continue

        # Skip derivative files
        if any(pattern in filepath.name.lower() for pattern in SKIP_PATTERNS):
            continue

        yield str(filepath)


def note_string_to_midi(note_str: str) -> int:
    """
    Convert a note string like 'E5' or 'C#4' to MIDI pitch.

    Based on dadagp.py noteNumber function.
    """
    # Parse octave (last character)
    octave = int(note_str[-1])
    pitch_class = note_str[:-1]

    # Map pitch class to semitones above C
    pitch_map = {
        'C': 0, 'C#': 1, 'Db': 1,
        'D': 2, 'D#': 3, 'Eb': 3,
        'E': 4, 'Fb': 4, 'E#': 5,
        'F': 5, 'F#': 6, 'Gb': 6,
        'G': 7, 'G#': 8, 'Ab': 8,
        'A': 9, 'A#': 10, 'Bb': 10,
        'B': 11, 'Cb': 11, 'B#': 0
    }

    pitch_value = pitch_map.get(pitch_class, 0)
    midi_number = octave * 12 + pitch_value
    return midi_number


def get_tuning_midi_pitches(track: gp.Track) -> Tuple[int, ...]:
    """
    Extract MIDI pitches for each string's open note.

    Returns tuple of MIDI pitches for strings 1-6 (high to low).
    """
    pitches = []
    for string in track.strings[:6]:  # Only first 6 strings
        # string.value is the MIDI pitch of the open string
        pitches.append(string.value)
    return tuple(pitches)


def get_tuning_note_names(track: gp.Track) -> List[str]:
    """
    Extract note names for each string (e.g., ['E5', 'B4', 'G4', 'D4', 'A3', 'E3']).
    """
    names = []
    for string in track.strings[:6]:
        # Convert MIDI pitch back to note name
        midi = string.value
        octave = midi // 12
        pitch_class = midi % 12

        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        note_name = f"{note_names[pitch_class]}{octave}"
        names.append(note_name)
    return names


def calculate_string_intervals(tuning: Tuple[int, ...]) -> List[int]:
    """
    Calculate intervals between adjacent strings.

    Standard tuning: [-5, -4, -5, -5, -5]
    Drop D: [-5, -4, -5, -5, -7]
    """
    intervals = []
    for i in range(len(tuning) - 1):
        intervals.append(tuning[i + 1] - tuning[i])
    return intervals


def is_standard_tuning_pattern(intervals: List[int]) -> bool:
    """Check if intervals match standard tuning pattern."""
    return intervals == [-5, -4, -5, -5, -5]


def is_drop_tuning_pattern(intervals: List[int]) -> bool:
    """Check if intervals match drop tuning pattern (like Drop D)."""
    return intervals == [-5, -4, -5, -5, -7]


def is_valid_guitar_tuning(tuning: Tuple[int, ...]) -> Tuple[bool, str]:
    """
    Check if tuning is valid for our processing.

    Returns (is_valid, tuning_type) where tuning_type is one of:
    - 'standard': Standard tuning intervals (may be transposed)
    - 'drop': Drop tuning (6th string lowered 2 semitones)
    - 'invalid': Not processable
    """
    if len(tuning) != 6:
        return False, 'invalid'

    intervals = calculate_string_intervals(tuning)

    if is_standard_tuning_pattern(intervals):
        return True, 'standard'
    elif is_drop_tuning_pattern(intervals):
        return True, 'drop'
    else:
        return False, 'invalid'


def calculate_tuning_offset(tuning: Tuple[int, ...]) -> int:
    """
    Calculate semitone offset from standard tuning.

    Positive = tuned up, Negative = tuned down.
    e.g., Half-step down returns -1
    """
    if not tuning:
        return 0
    # Compare high E string to standard (64)
    return tuning[0] - STANDARD_TUNING[0]


def normalize_to_standard_tuning(notes: List[TabNote],
                                  tuning: Tuple[int, ...],
                                  tuning_type: str) -> List[TabNote]:
    """
    Normalize notes to standard tuning at capo 0.

    Adjusts MIDI pitches so they represent what would be played
    in standard tuning at the same fret positions.
    """
    if tuning_type == 'invalid':
        return notes

    # Calculate offset per string
    offsets = []
    for i, actual_pitch in enumerate(tuning[:6]):
        standard_pitch = STANDARD_TUNING[i]
        # For drop tuning, string 6 is lowered 2 more semitones
        if tuning_type == 'drop' and i == 5:
            # The drop note should be represented at fret -2 in standard
            # So we need to adjust differently
            pass
        offsets.append(actual_pitch - standard_pitch)

    # All offsets should be equal for a simple transpose
    global_offset = offsets[0] if offsets else 0

    normalized = []
    for note in notes:
        string_idx = note.string - 1  # Convert to 0-indexed
        if 0 <= string_idx < len(offsets):
            # Adjust pitch to standard tuning
            new_pitch = note.pitch - offsets[string_idx]

            # For drop tuning, handle string 6 specially
            if tuning_type == 'drop' and string_idx == 5:
                # Frets on string 6 need adjustment
                # In drop D, fret 0 = D (38), but we want it as fret -2 on E (40)
                new_fret = note.fret - 2
            else:
                new_fret = note.fret

            normalized.append(TabNote(
                time_ticks=note.time_ticks,
                duration_ticks=note.duration_ticks,
                string=note.string,
                fret=new_fret,
                pitch=new_pitch,
                velocity=note.velocity
            ))
        else:
            normalized.append(note)

    return normalized


def extract_tempo_changes(song: gp.Song) -> List[TempoChange]:
    """
    Extract tempo changes from a GuitarPro song.

    Returns list of (tick, tempo) pairs.
    """
    tempo_changes = [TempoChange(tick=0, tempo=float(song.tempo))]

    # Search for tempo changes in beat effects
    current_tick = 0
    ticks_per_beat = 960  # Standard GP resolution

    if not song.tracks:
        return tempo_changes

    # Use first track for structure
    track = song.tracks[0]

    for measure in track.measures:
        for voice in measure.voices:
            beat_tick = current_tick
            for beat in voice.beats:
                # Check for tempo change in mix table
                if hasattr(beat, 'effect') and beat.effect:
                    if hasattr(beat.effect, 'mixTableChange') and beat.effect.mixTableChange:
                        mtc = beat.effect.mixTableChange
                        if hasattr(mtc, 'tempo') and mtc.tempo:
                            tempo_changes.append(TempoChange(
                                tick=beat_tick,
                                tempo=float(mtc.tempo.value)
                            ))

                beat_tick += beat.duration.time if hasattr(beat.duration, 'time') else ticks_per_beat

        # Update for next measure
        header = measure.header
        numerator = header.timeSignature.numerator if hasattr(header, 'timeSignature') else 4
        current_tick += numerator * ticks_per_beat

    return tempo_changes


def extract_notes_from_track(track: gp.Track, song: gp.Song) -> List[TabNote]:
    """
    Extract all notes from a GuitarPro track.

    Returns list of TabNote objects with timing in ticks.
    """
    notes = []
    current_tick = 0
    ticks_per_beat = 960  # Standard GP resolution

    # Get tuning for pitch calculation
    tuning = get_tuning_midi_pitches(track)

    for measure in track.measures:
        measure_start_tick = current_tick

        for voice in measure.voices:
            beat_tick = measure_start_tick

            for beat in voice.beats:
                # Get beat duration
                if hasattr(beat.duration, 'time'):
                    beat_duration = beat.duration.time
                else:
                    # Fallback: calculate from duration value
                    beat_duration = ticks_per_beat // (2 ** (beat.duration.value - 1)) if beat.duration.value > 0 else ticks_per_beat

                # Extract notes from this beat
                for note in beat.notes:
                    # Skip ties, dead notes, etc. for tablature - only normal notes
                    if hasattr(note, 'type'):
                        if note.type != gp.NoteType.normal and note.type != gp.NoteType.tie:
                            continue

                    string_num = note.string  # 1-indexed
                    fret = note.value

                    # Skip invalid string numbers
                    if string_num < 1 or string_num > 6:
                        continue

                    # Skip unreasonably high frets
                    if fret > 24:
                        continue

                    # Calculate MIDI pitch
                    string_idx = string_num - 1
                    if string_idx < len(tuning):
                        open_pitch = tuning[string_idx]
                        midi_pitch = open_pitch + fret
                    else:
                        midi_pitch = 60 + fret  # Fallback

                    # Get velocity
                    velocity = note.velocity if hasattr(note, 'velocity') and note.velocity else 80

                    notes.append(TabNote(
                        time_ticks=beat_tick,
                        duration_ticks=beat_duration,
                        string=string_num,
                        fret=fret,
                        pitch=midi_pitch,
                        velocity=velocity
                    ))

                beat_tick += beat_duration

        # Update position for next measure
        header = measure.header
        if hasattr(header, 'timeSignature'):
            numerator = header.timeSignature.numerator
            denominator = header.timeSignature.denominator.value if hasattr(header.timeSignature.denominator, 'value') else 4
            measure_duration = int(numerator * ticks_per_beat * 4 / denominator)
        else:
            measure_duration = 4 * ticks_per_beat  # Assume 4/4

        current_tick += measure_duration

    # Sort by time
    notes.sort(key=lambda n: (n.time_ticks, n.string))

    return notes


def load_guitarpro_file(filepath: str) -> Optional[gp.Song]:
    """
    Load a GuitarPro file.

    Returns None if loading fails.
    """
    try:
        song = gp.parse(filepath)
        return song
    except Exception as e:
        logger.warning(f"Failed to load {filepath}: {e}")
        return None


def extract_track(track: gp.Track, song: gp.Song,
                  source_file: str, track_index: int) -> ExtractedTrack:
    """
    Extract all data from a single track.
    """
    # Get tuning info
    tuning = get_tuning_midi_pitches(track)
    tuning_names = get_tuning_note_names(track)
    is_valid, tuning_type = is_valid_guitar_tuning(tuning)
    offset = calculate_tuning_offset(tuning)

    # Extract notes
    notes = extract_notes_from_track(track, song)

    return ExtractedTrack(
        source_file=source_file,
        track_index=track_index,
        track_name=track.name,
        instrument=track.channel.instrument if hasattr(track.channel, 'instrument') else -1,
        notes=notes,
        tuning=tuning,
        tuning_strings=tuning_names,
        tempo=float(song.tempo),
        is_standard_tuning=(tuning_type == 'standard' and offset == 0),
        tuning_offset=offset
    )


if __name__ == "__main__":
    # Quick test
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dadagp_extractor.py <path_to_gp_file>")
        sys.exit(1)

    filepath = sys.argv[1]
    song = load_guitarpro_file(filepath)

    if song is None:
        print("Failed to load file")
        sys.exit(1)

    print(f"Loaded: {filepath}")
    print(f"Tempo: {song.tempo} BPM")
    print(f"Tracks: {len(song.tracks)}")

    for i, track in enumerate(song.tracks):
        print(f"\n  Track {i}: {track.name}")
        print(f"    Instrument: {track.channel.instrument}")
        print(f"    Strings: {len(track.strings)}")

        if len(track.strings) == 6:
            tuning = get_tuning_midi_pitches(track)
            tuning_names = get_tuning_note_names(track)
            is_valid, tuning_type = is_valid_guitar_tuning(tuning)
            offset = calculate_tuning_offset(tuning)

            print(f"    Tuning: {tuning_names} ({tuning_type}, offset={offset})")

            notes = extract_notes_from_track(track, song)
            print(f"    Notes: {len(notes)}")

            if notes:
                print(f"    First 5 notes:")
                for note in notes[:5]:
                    print(f"      {note}")
