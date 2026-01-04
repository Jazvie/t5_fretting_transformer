"""
DadaGP to JAMS/MIDI - Convert extracted notes to JAMS and MIDI files.

Creates output files compatible with the fret_t5 training pipeline.
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import logging

try:
    import pretty_midi
except ImportError:
    raise ImportError("pretty_midi is required. Install with: pip install pretty_midi")

from dadagp_extractor import (
    TabNote,
    ExtractedTrack,
    TempoChange,
    STANDARD_TUNING,
    extract_tempo_changes,
    get_tuning_midi_pitches,
)
from dadagp_filters import (
    normalize_notes_to_standard,
    is_valid_guitar_tuning,
)

logger = logging.getLogger(__name__)

# Ticks per beat (quarter note) - standard for GuitarPro
TICKS_PER_BEAT = 960


@dataclass
class TempoMapEntry:
    """A tempo change for the tempo map."""
    tick: float
    tempo: float


def build_tempo_map(tempo_changes: List[TempoChange]) -> Dict:
    """
    Build a tempo map dictionary for JAMS sandbox.
    """
    return {
        'ticks_per_beat': TICKS_PER_BEAT,
        'tempo_changes': [[tc.tick, tc.tempo] for tc in tempo_changes]
    }


def tick_to_seconds(tick: float, tempo_changes: List[TempoChange],
                    ticks_per_beat: int = TICKS_PER_BEAT) -> float:
    """
    Convert tick position to seconds using tempo map.
    """
    if not tempo_changes:
        # Default 120 BPM
        return tick / ticks_per_beat * (60.0 / 120.0)

    total_seconds = 0.0
    prev_tick = 0.0
    prev_tempo = tempo_changes[0].tempo

    for tc in tempo_changes:
        if tc.tick > tick:
            break

        # Add time from previous segment
        tick_diff = tc.tick - prev_tick
        seconds_per_tick = 60.0 / (prev_tempo * ticks_per_beat)
        total_seconds += tick_diff * seconds_per_tick

        prev_tick = tc.tick
        prev_tempo = tc.tempo

    # Add remaining ticks
    remaining_ticks = tick - prev_tick
    seconds_per_tick = 60.0 / (prev_tempo * ticks_per_beat)
    total_seconds += remaining_ticks * seconds_per_tick

    return total_seconds


def create_jams_annotation(notes: List[TabNote],
                           tempo_changes: List[TempoChange] = None,
                           source: str = "dadagp") -> Dict:
    """
    Create JAMS annotation dictionary from notes.

    Format matches SynthTab JAMS files.
    """
    # Build annotation data - time in TICKS (like SynthTab)
    data = []
    for note in notes:
        data.append({
            "time": note.time_ticks,
            "duration": note.duration_ticks,
            "value": {
                "fret": note.fret,
                "string": note.string
            }
        })

    # Build tempo map for sandbox
    if tempo_changes is None:
        tempo_changes = [TempoChange(tick=0, tempo=120.0)]

    tempo_map = build_tempo_map(tempo_changes)

    return {
        "annotations": [
            {
                "namespace": "note_tab",
                "data": data,
                "sandbox": {
                    "tempo_map": tempo_map,
                    "source": source,
                    "ticks_per_beat": TICKS_PER_BEAT
                }
            }
        ]
    }


def save_jams_file(jams_data: Dict, filepath: str) -> None:
    """Save JAMS data to file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(jams_data, f, indent=2)


def create_midi_from_notes(notes: List[TabNote],
                           tempo_changes: List[TempoChange],
                           instrument_program: int = 25) -> pretty_midi.PrettyMIDI:
    """
    Create a PrettyMIDI object from notes.

    Args:
        notes: List of TabNote objects with timing in ticks
        tempo_changes: List of tempo changes
        instrument_program: MIDI program (24=Nylon, 25=Steel)
    """
    # Determine initial tempo
    initial_tempo = tempo_changes[0].tempo if tempo_changes else 120.0

    # Create PrettyMIDI with initial tempo
    midi = pretty_midi.PrettyMIDI(initial_tempo=initial_tempo)

    # Add tempo changes
    for tc in tempo_changes[1:]:
        # Convert tick to seconds for this tempo change
        time_sec = tick_to_seconds(tc.tick, tempo_changes)
        # Calculate microseconds per beat
        microseconds_per_beat = int(60000000 / tc.tempo)
        # Note: pretty_midi doesn't have a direct API for tempo changes
        # They are added implicitly through the timing

    # Create instrument
    instrument_name = "Acoustic Guitar (nylon)" if instrument_program == 24 else "Acoustic Guitar (steel)"
    instrument = pretty_midi.Instrument(
        program=instrument_program,
        is_drum=False,
        name=instrument_name
    )

    # Add notes
    for note in notes:
        # Convert ticks to seconds
        start_time = tick_to_seconds(note.time_ticks, tempo_changes)
        end_time = tick_to_seconds(note.time_ticks + note.duration_ticks, tempo_changes)

        # Ensure minimum duration
        if end_time <= start_time:
            end_time = start_time + 0.1

        # Create MIDI note
        midi_note = pretty_midi.Note(
            velocity=note.velocity,
            pitch=note.pitch,
            start=start_time,
            end=end_time
        )
        instrument.notes.append(midi_note)

    midi.instruments.append(instrument)
    return midi


def save_midi_file(midi: pretty_midi.PrettyMIDI, filepath: str) -> None:
    """Save MIDI to file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    midi.write(filepath)


@dataclass
class ProcessedTrack:
    """A fully processed track ready for manifest."""
    track_id: str
    source_file: str
    track_index: int
    track_name: str
    instrument_type: str  # "nylon" or "steel"
    instrument_program: int  # 24 or 25
    midi_path: str
    jams_path: str
    note_count: int
    original_tuning: Tuple[int, ...]
    tuning_type: str
    tempo: float


def process_track_to_files(notes: List[TabNote],
                           tuning: Tuple[int, ...],
                           tuning_type: str,
                           tempo_changes: List[TempoChange],
                           instrument_type: str,
                           output_dir: str,
                           track_id: str) -> Tuple[str, str]:
    """
    Process notes and save to JAMS and MIDI files.

    Returns (midi_path, jams_path).
    """
    # Normalize notes to standard tuning
    normalized_notes = normalize_notes_to_standard(notes, tuning, tuning_type)

    # Determine instrument program
    instrument_program = 24 if instrument_type == "nylon" else 25

    # Create output paths
    output_path = Path(output_dir) / track_id
    output_path.mkdir(parents=True, exist_ok=True)

    midi_path = str(output_path / "combined.mid")
    jams_path = str(output_path / "track.jams")

    # Create and save JAMS
    jams_data = create_jams_annotation(normalized_notes, tempo_changes)
    save_jams_file(jams_data, jams_path)

    # Create and save MIDI
    midi = create_midi_from_notes(normalized_notes, tempo_changes, instrument_program)
    save_midi_file(midi, midi_path)

    return midi_path, jams_path


def generate_track_id(source_file: str, track_index: int, track_name: str,
                      instrument_type: str) -> str:
    """
    Generate a unique track ID.

    Format: "Artist - Song - format__trackN - Instrument Type"
    """
    # Extract filename without extension
    filename = Path(source_file).stem

    # Clean track name
    clean_name = track_name.replace('/', '_').replace('\\', '_')
    clean_name = clean_name.replace(':', '_').replace('__', '_')

    # Get format from extension
    ext = Path(source_file).suffix.lower().replace('.', '')

    # Build instrument name
    if instrument_type == "nylon":
        inst_name = "Acoustic Nylon Guitar"
    else:
        inst_name = "Acoustic Steel Guitar"

    return f"{filename}__{track_index} - {inst_name}"


if __name__ == "__main__":
    import sys
    import tempfile
    from dadagp_extractor import load_guitarpro_file, extract_notes_from_track
    from dadagp_filters import filter_song_tracks, FilterConfig

    if len(sys.argv) < 2:
        print("Usage: python dadagp_to_jams.py <path_to_gp_file> [output_dir]")
        sys.exit(1)

    filepath = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else tempfile.mkdtemp(prefix="dadagp_test_")

    song = load_guitarpro_file(filepath)
    if song is None:
        print("Failed to load file")
        sys.exit(1)

    print(f"Loaded: {filepath}")
    print(f"Output dir: {output_dir}")

    # Filter for acoustic tracks
    config = FilterConfig()
    results = filter_song_tracks(song, config)
    passed = [r for r in results if r.passed]

    if not passed:
        print("No acoustic tracks found")
        sys.exit(0)

    # Get tempo changes
    tempo_changes = extract_tempo_changes(song)
    print(f"Tempo changes: {len(tempo_changes)}")

    # Process each passed track
    for r in passed:
        track = r.track
        tuning = get_tuning_midi_pitches(track)
        _, tuning_type = is_valid_guitar_tuning(tuning)

        notes = extract_notes_from_track(track, song)
        track_id = generate_track_id(filepath, r.track_index, track.name, r.instrument_type)

        print(f"\nProcessing: {track_id}")
        print(f"  Notes: {len(notes)}")
        print(f"  Tuning type: {tuning_type}")

        midi_path, jams_path = process_track_to_files(
            notes=notes,
            tuning=tuning,
            tuning_type=tuning_type,
            tempo_changes=tempo_changes,
            instrument_type=r.instrument_type,
            output_dir=output_dir,
            track_id=track_id
        )

        print(f"  MIDI: {midi_path}")
        print(f"  JAMS: {jams_path}")

        # Verify files
        if os.path.exists(jams_path):
            with open(jams_path) as f:
                jams = json.load(f)
                note_count = len(jams['annotations'][0]['data'])
                print(f"  JAMS notes: {note_count}")

        if os.path.exists(midi_path):
            midi = pretty_midi.PrettyMIDI(midi_path)
            midi_notes = sum(len(inst.notes) for inst in midi.instruments)
            print(f"  MIDI notes: {midi_notes}")
