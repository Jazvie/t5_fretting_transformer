"""
DadaGP Filters - Acoustic track filtering and tuning normalization.

Filters GuitarPro tracks to find acoustic guitar tracks using:
1. MIDI instrument program numbers (24=Nylon, 25=Steel)
2. Keyword matching (multilingual guitar terms)
3. Tuning validation and normalization
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import logging

try:
    import guitarpro as gp
except ImportError:
    raise ImportError("PyGuitarPro is required. Install with: pip install PyGuitarPro")

from dadagp_extractor import (
    TabNote,
    ExtractedTrack,
    STANDARD_TUNING,
    get_tuning_midi_pitches,
    get_tuning_note_names,
    is_valid_guitar_tuning,
    calculate_tuning_offset,
    extract_notes_from_track,
    extract_tempo_changes,
)

logger = logging.getLogger(__name__)

# MIDI Program numbers for acoustic guitars
# Per General MIDI standard:
#   24 = Acoustic Guitar (nylon)
#   25 = Acoustic Guitar (steel)
ACOUSTIC_PROGRAMS = [24, 25]

# Extended programs that might contain guitar (for keyword fallback)
GUITAR_PROGRAMS = [24, 25, 26, 27, 28]  # 26=Jazz, 27=Clean, 28=Muted

# Guitar-related keywords in multiple languages
GUITAR_KEYWORDS = [
    # English
    'guitar', 'acoustic', 'nylon', 'steel', 'fingerstyle', 'classical',
    'fingerpicking', 'arpeggio', 'strumming',
    # Spanish
    'guitarra', 'acustica', 'clasica', 'acústica', 'clásica',
    # French
    'guitare', 'acoustique', 'classique',
    # German
    'gitarre', 'akustisch', 'akustik', 'klassisch', 'klassische',
    # Italian
    'chitarra', 'acustica', 'classica',
    # Portuguese
    'violao', 'violão', 'acustico', 'acústico',
]

# Keywords that suggest the track is NOT an acoustic guitar
EXCLUDE_KEYWORDS = [
    'bass', 'bajo', 'basse', 'baixo',
    'drum', 'percussion', 'bateria', 'batterie', 'schlagzeug',
    'vocal', 'voice', 'voz', 'voix', 'stimme', 'canto',
    'keyboard', 'piano', 'organ', 'synth',
    'electric', 'distortion', 'overdrive', 'lead',
]


@dataclass
class FilterConfig:
    """Configuration for track filtering."""
    acoustic_programs: List[int] = None
    guitar_programs: List[int] = None
    guitar_keywords: List[str] = None
    exclude_keywords: List[str] = None
    min_notes: int = 10
    max_notes: int = 2000
    require_6_strings: bool = True
    allow_non_standard_tuning: bool = True  # Allow drop D, etc.
    max_tuning_offset: int = 3  # Max semitones from standard (e.g., 3 = Eb tuning)

    def __post_init__(self):
        if self.acoustic_programs is None:
            self.acoustic_programs = ACOUSTIC_PROGRAMS
        if self.guitar_programs is None:
            self.guitar_programs = GUITAR_PROGRAMS
        if self.guitar_keywords is None:
            self.guitar_keywords = GUITAR_KEYWORDS
        if self.exclude_keywords is None:
            self.exclude_keywords = EXCLUDE_KEYWORDS


@dataclass
class FilterResult:
    """Result of filtering a track."""
    track: gp.Track
    track_index: int
    passed: bool
    instrument_match: bool
    keyword_match: bool
    tuning_valid: bool
    note_count_valid: bool
    rejection_reason: Optional[str] = None
    instrument_type: str = "unknown"  # "nylon", "steel", "unknown"


def is_acoustic_by_instrument(track: gp.Track, config: FilterConfig) -> Tuple[bool, str]:
    """
    Check if track is acoustic guitar by MIDI instrument number.

    Returns (is_acoustic, instrument_type)
    """
    if not hasattr(track.channel, 'instrument'):
        return False, "unknown"

    instrument = track.channel.instrument

    if instrument == 24:
        return True, "nylon"
    elif instrument == 25:
        return True, "steel"
    elif instrument in config.guitar_programs:
        return False, "guitar_but_not_acoustic"
    else:
        return False, "unknown"


def is_acoustic_by_keywords(track: gp.Track, config: FilterConfig) -> bool:
    """
    Check if track name contains acoustic guitar keywords.

    Uses multilingual keyword matching.
    """
    track_name = track.name.lower()

    # First check exclusion keywords
    for keyword in config.exclude_keywords:
        if keyword in track_name:
            return False

    # Then check inclusion keywords
    for keyword in config.guitar_keywords:
        if keyword in track_name:
            return True

    return False


def is_valid_string_count(track: gp.Track, config: FilterConfig) -> bool:
    """Check if track has valid number of strings."""
    string_count = len(track.strings)

    if config.require_6_strings:
        return string_count == 6
    else:
        return 4 <= string_count <= 7  # Support 4-7 string guitars


def is_valid_tuning_for_processing(track: gp.Track, config: FilterConfig) -> Tuple[bool, str]:
    """
    Check if track tuning can be processed.

    Returns (is_valid, tuning_type) where tuning_type is:
    - 'standard': Standard tuning (possibly transposed)
    - 'drop': Drop tuning
    - 'invalid': Cannot process
    """
    if len(track.strings) != 6:
        return False, 'invalid'

    tuning = get_tuning_midi_pitches(track)
    is_valid, tuning_type = is_valid_guitar_tuning(tuning)

    if not is_valid:
        return False, 'invalid'

    # Check if tuning offset is within acceptable range
    if not config.allow_non_standard_tuning:
        offset = calculate_tuning_offset(tuning)
        if abs(offset) > config.max_tuning_offset:
            return False, 'invalid'

    return True, tuning_type


def is_note_count_valid(notes: List[TabNote], config: FilterConfig) -> bool:
    """Check if note count is within acceptable range."""
    count = len(notes)
    return config.min_notes <= count <= config.max_notes


def filter_track(track: gp.Track, track_index: int, song: gp.Song,
                 config: FilterConfig) -> FilterResult:
    """
    Apply all filters to a single track.

    Returns FilterResult with pass/fail status and reasons.
    """
    # Skip percussion tracks immediately
    if hasattr(track, 'isPercussionTrack') and track.isPercussionTrack:
        return FilterResult(
            track=track,
            track_index=track_index,
            passed=False,
            instrument_match=False,
            keyword_match=False,
            tuning_valid=False,
            note_count_valid=False,
            rejection_reason="percussion_track"
        )

    # Check string count
    if not is_valid_string_count(track, config):
        return FilterResult(
            track=track,
            track_index=track_index,
            passed=False,
            instrument_match=False,
            keyword_match=False,
            tuning_valid=False,
            note_count_valid=False,
            rejection_reason=f"invalid_string_count_{len(track.strings)}"
        )

    # Check instrument
    is_acoustic, instrument_type = is_acoustic_by_instrument(track, config)

    # Check keywords (as backup or confirmation)
    keyword_match = is_acoustic_by_keywords(track, config)

    # A track passes instrument filter if:
    # 1. It has acoustic instrument program (24/25), OR
    # 2. It has general guitar program AND matches keywords
    instrument_match = is_acoustic or (
        instrument_type == "guitar_but_not_acoustic" and keyword_match
    )

    # Check tuning
    tuning_valid, tuning_type = is_valid_tuning_for_processing(track, config)

    # Extract notes to check count
    notes = extract_notes_from_track(track, song)
    note_count_valid = is_note_count_valid(notes, config)

    # Determine overall pass/fail
    passed = instrument_match and tuning_valid and note_count_valid

    # Determine rejection reason if failed
    rejection_reason = None
    if not passed:
        if not instrument_match:
            rejection_reason = f"not_acoustic_instrument_{track.channel.instrument if hasattr(track.channel, 'instrument') else 'unknown'}"
        elif not tuning_valid:
            rejection_reason = f"invalid_tuning_{tuning_type}"
        elif not note_count_valid:
            rejection_reason = f"note_count_{len(notes)}"

    # Determine instrument type for passed tracks
    if passed:
        if instrument_type == "nylon":
            final_type = "nylon"
        elif instrument_type == "steel":
            final_type = "steel"
        else:
            # Try to determine from track name
            track_lower = track.name.lower()
            if 'nylon' in track_lower or 'classical' in track_lower:
                final_type = "nylon"
            elif 'steel' in track_lower:
                final_type = "steel"
            else:
                final_type = "steel"  # Default to steel if unknown
    else:
        final_type = "unknown"

    return FilterResult(
        track=track,
        track_index=track_index,
        passed=passed,
        instrument_match=instrument_match,
        keyword_match=keyword_match,
        tuning_valid=tuning_valid,
        note_count_valid=note_count_valid,
        rejection_reason=rejection_reason,
        instrument_type=final_type
    )


def filter_song_tracks(song: gp.Song, config: FilterConfig = None) -> List[FilterResult]:
    """
    Filter all tracks in a song for acoustic guitars.

    Returns list of FilterResults for each track.
    """
    if config is None:
        config = FilterConfig()

    results = []
    for i, track in enumerate(song.tracks):
        result = filter_track(track, i, song, config)
        results.append(result)

    return results


def get_acoustic_tracks(song: gp.Song, config: FilterConfig = None) -> List[FilterResult]:
    """
    Get only the tracks that pass acoustic guitar filtering.
    """
    results = filter_song_tracks(song, config)
    return [r for r in results if r.passed]


def normalize_notes_to_standard(notes: List[TabNote],
                                 tuning: Tuple[int, ...],
                                 tuning_type: str) -> List[TabNote]:
    """
    Normalize notes from non-standard tuning to standard tuning representation.

    For tracks with transposed tuning (e.g., Eb), we adjust the MIDI pitches
    so they represent what would be played in standard tuning at capo 0.

    For drop tunings, we handle the 6th string specially.
    """
    if tuning_type == 'invalid':
        return notes

    # Calculate per-string offset from standard tuning
    offsets = []
    for i in range(min(6, len(tuning))):
        offset = tuning[i] - STANDARD_TUNING[i]
        offsets.append(offset)

    # Pad with zeros if needed
    while len(offsets) < 6:
        offsets.append(0)

    normalized = []
    for note in notes:
        string_idx = note.string - 1  # Convert to 0-indexed

        if 0 <= string_idx < 6:
            # Normalize pitch to standard tuning
            normalized_pitch = note.pitch - offsets[string_idx]

            # For drop tuning, handle fret adjustment on string 6
            if tuning_type == 'drop' and string_idx == 5:
                # In drop D, fret 0 produces D (38)
                # In standard, fret 0 produces E (40)
                # We want to represent drop D's open string as fret -2
                normalized_fret = note.fret - 2
            else:
                normalized_fret = note.fret

            normalized.append(TabNote(
                time_ticks=note.time_ticks,
                duration_ticks=note.duration_ticks,
                string=note.string,
                fret=normalized_fret,
                pitch=normalized_pitch,
                velocity=note.velocity
            ))
        else:
            # Keep note as-is if string out of range
            normalized.append(note)

    return normalized


@dataclass
class FilterStats:
    """Statistics from filtering a collection of files."""
    total_files: int = 0
    files_with_acoustic: int = 0
    total_tracks: int = 0
    acoustic_tracks: int = 0
    nylon_tracks: int = 0
    steel_tracks: int = 0
    rejected_by_instrument: int = 0
    rejected_by_tuning: int = 0
    rejected_by_note_count: int = 0
    rejected_by_string_count: int = 0
    rejected_percussion: int = 0

    def update(self, results: List[FilterResult]):
        """Update stats with filter results from one file."""
        self.total_tracks += len(results)

        passed_any = False
        for r in results:
            if r.passed:
                passed_any = True
                self.acoustic_tracks += 1
                if r.instrument_type == "nylon":
                    self.nylon_tracks += 1
                elif r.instrument_type == "steel":
                    self.steel_tracks += 1
            else:
                reason = r.rejection_reason or ""
                if "percussion" in reason:
                    self.rejected_percussion += 1
                elif "not_acoustic" in reason:
                    self.rejected_by_instrument += 1
                elif "tuning" in reason:
                    self.rejected_by_tuning += 1
                elif "note_count" in reason:
                    self.rejected_by_note_count += 1
                elif "string_count" in reason:
                    self.rejected_by_string_count += 1

        if passed_any:
            self.files_with_acoustic += 1
        self.total_files += 1

    def __str__(self) -> str:
        return f"""Filter Statistics:
  Files processed: {self.total_files}
  Files with acoustic tracks: {self.files_with_acoustic} ({100*self.files_with_acoustic/max(1,self.total_files):.1f}%)
  Total tracks: {self.total_tracks}
  Acoustic tracks: {self.acoustic_tracks} ({100*self.acoustic_tracks/max(1,self.total_tracks):.1f}%)
    - Nylon: {self.nylon_tracks}
    - Steel: {self.steel_tracks}
  Rejected:
    - By instrument: {self.rejected_by_instrument}
    - By tuning: {self.rejected_by_tuning}
    - By note count: {self.rejected_by_note_count}
    - By string count: {self.rejected_by_string_count}
    - Percussion: {self.rejected_percussion}"""


if __name__ == "__main__":
    import sys
    from dadagp_extractor import load_guitarpro_file

    if len(sys.argv) < 2:
        print("Usage: python dadagp_filters.py <path_to_gp_file>")
        sys.exit(1)

    filepath = sys.argv[1]
    song = load_guitarpro_file(filepath)

    if song is None:
        print("Failed to load file")
        sys.exit(1)

    print(f"Loaded: {filepath}")
    print(f"Tracks: {len(song.tracks)}")

    config = FilterConfig()
    results = filter_song_tracks(song, config)

    print("\nFilter Results:")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] Track {r.track_index}: {r.track.name}")
        print(f"         Instrument: {r.track.channel.instrument if hasattr(r.track.channel, 'instrument') else '?'} -> {r.instrument_type}")
        print(f"         Matches: inst={r.instrument_match}, kw={r.keyword_match}, tuning={r.tuning_valid}, notes={r.note_count_valid}")
        if r.rejection_reason:
            print(f"         Reason: {r.rejection_reason}")

    # Show passed tracks with normalized notes
    passed = [r for r in results if r.passed]
    if passed:
        print(f"\n{len(passed)} acoustic track(s) found:")
        for r in passed:
            tuning = get_tuning_midi_pitches(r.track)
            tuning_names = get_tuning_note_names(r.track)
            _, tuning_type = is_valid_guitar_tuning(tuning)

            notes = extract_notes_from_track(r.track, song)
            normalized = normalize_notes_to_standard(notes, tuning, tuning_type)

            print(f"\n  Track {r.track_index}: {r.track.name} ({r.instrument_type})")
            print(f"  Tuning: {tuning_names} ({tuning_type})")
            print(f"  Notes: {len(notes)}")
            print(f"  First 3 notes (original):")
            for n in notes[:3]:
                print(f"    string={n.string}, fret={n.fret}, pitch={n.pitch}")
            print(f"  First 3 notes (normalized):")
            for n in normalized[:3]:
                print(f"    string={n.string}, fret={n.fret}, pitch={n.pitch}")
