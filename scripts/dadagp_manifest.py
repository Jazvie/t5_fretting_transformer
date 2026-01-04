"""
DadaGP Manifest Builder - Create JSONL manifests and handle deduplication.

Creates manifest files matching the SynthTab format for training.
"""

import json
import hashlib
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import logging

from dadagp_extractor import TabNote

logger = logging.getLogger(__name__)


@dataclass
class TrackInfo:
    """Information about a processed track for manifest generation."""
    track_id: str
    source_file: str
    artist: str
    song: str
    track_index: int
    track_name: str
    instrument_type: str  # "nylon" or "steel"
    instrument_program: int  # 24 or 25
    midi_path: str
    jams_path: str
    note_count: int
    is_validation: bool = False
    content_hash: str = ""  # Hash of note content for deduplication


@dataclass
class ManifestEntry:
    """A single manifest entry matching SynthTab format."""
    track_id: str
    midi_path: str
    tab_path: str
    program: int
    split: str
    instrument: str

    def to_dict(self) -> Dict:
        return {
            "track_id": self.track_id,
            "midi_path": self.midi_path,
            "tab_path": self.tab_path,
            "program": self.program,
            "split": self.split,
            "instrument": self.instrument
        }


def load_dadagp_metadata(dadagp_path: str) -> Dict[str, bool]:
    """
    Load DadaGP metadata to determine train/validation splits.

    Returns dict mapping filename patterns to is_validation bool.
    """
    metadata = {}

    training_path = Path(dadagp_path) / "_DadaGP_training.json"
    validation_path = Path(dadagp_path) / "_DadaGP_validation.json"

    # Load training files
    if training_path.exists():
        with open(training_path) as f:
            training_data = json.load(f)
            for item in training_data:
                # Extract original filename from tokens.txt path
                tokens_path = item.get('tokens.txt', '')
                # Convert tokens path back to gp path
                # e.g., "1/1 God/1 God - Grace.gp4.tokens.txt" -> "1 God - Grace.gp4"
                filename = tokens_path.split('/')[-1].replace('.tokens.txt', '')
                if filename:
                    metadata[filename] = False  # Not validation

    # Load validation files
    if validation_path.exists():
        with open(validation_path) as f:
            validation_data = json.load(f)
            for item in validation_data:
                tokens_path = item.get('tokens.txt', '')
                filename = tokens_path.split('/')[-1].replace('.tokens.txt', '')
                if filename:
                    metadata[filename] = True  # Is validation

    logger.info(f"Loaded metadata: {len(metadata)} files ({sum(metadata.values())} validation)")
    return metadata


def is_validation_file(source_file: str, metadata: Dict[str, bool]) -> bool:
    """Check if a source file is in the validation set."""
    filename = Path(source_file).name

    # Try exact match first
    if filename in metadata:
        return metadata[filename]

    # Try without extension variations
    base_name = Path(source_file).stem
    for ext in ['.gp3', '.gp4', '.gp5', '.gpx']:
        test_name = base_name + ext
        if test_name in metadata:
            return metadata[test_name]

    # Default to training if not found
    return False


def compute_content_hash(notes: List[TabNote]) -> str:
    """
    Compute a hash of note content for deduplication.

    Uses sorted (time, string, fret) tuples.
    """
    if not notes:
        return ""

    # Create a stable representation
    note_data = sorted([
        (int(n.time_ticks), n.string, n.fret)
        for n in notes
    ])

    # Convert to string and hash
    data_str = json.dumps(note_data, sort_keys=True)
    return hashlib.md5(data_str.encode()).hexdigest()


def normalize_song_name(name: str) -> str:
    """
    Normalize song name for duplicate detection.

    Removes version numbers, extra spaces, etc.
    """
    # Convert to lowercase
    name = name.lower()

    # Remove common suffixes like (2), (v2), (acoustic), etc.
    name = re.sub(r'\s*\([^)]*\)\s*$', '', name)
    name = re.sub(r'\s*\[[^\]]*\]\s*$', '', name)

    # Remove extra whitespace
    name = ' '.join(name.split())

    # Remove common suffixes
    for suffix in [' - acoustic', ' acoustic', ' - live', ' live', ' - tab', ' tab']:
        if name.endswith(suffix):
            name = name[:-len(suffix)]

    return name.strip()


def extract_artist_song(source_file: str) -> Tuple[str, str]:
    """
    Extract artist and song from file path.

    DadaGP format: "Artist Name - Song Title.gp4"
    Directory structure: DadaGP-v1.1/A/Artist Name/Artist Name - Song Title.gp4
    """
    filename = Path(source_file).stem

    # Try to split by " - "
    if ' - ' in filename:
        parts = filename.split(' - ', 1)
        artist = parts[0].strip()
        song = parts[1].strip() if len(parts) > 1 else filename
    else:
        # Use parent directory as artist
        artist = Path(source_file).parent.name
        song = filename

    return artist, song


@dataclass
class DuplicateDetector:
    """Detects and tracks duplicate tracks."""
    seen_hashes: Dict[str, TrackInfo] = field(default_factory=dict)
    seen_normalized_names: Dict[str, TrackInfo] = field(default_factory=dict)
    duplicates: List[Tuple[TrackInfo, TrackInfo, str]] = field(default_factory=list)

    def check_duplicate(self, track: TrackInfo) -> Optional[TrackInfo]:
        """
        Check if track is a duplicate.

        Returns the original TrackInfo if duplicate, None otherwise.
        """
        # Method 1: Content hash (most reliable)
        if track.content_hash and track.content_hash in self.seen_hashes:
            original = self.seen_hashes[track.content_hash]
            self.duplicates.append((original, track, "content_hash"))
            return original

        # Method 2: Normalized artist+song name
        normalized = f"{normalize_song_name(track.artist)}|{normalize_song_name(track.song)}"
        if normalized in self.seen_normalized_names:
            original = self.seen_normalized_names[normalized]
            self.duplicates.append((original, track, "normalized_name"))
            return original

        # Not a duplicate - register it
        if track.content_hash:
            self.seen_hashes[track.content_hash] = track
        self.seen_normalized_names[normalized] = track

        return None

    def get_stats(self) -> Dict:
        """Get duplicate detection statistics."""
        hash_dups = sum(1 for _, _, reason in self.duplicates if reason == "content_hash")
        name_dups = sum(1 for _, _, reason in self.duplicates if reason == "normalized_name")

        return {
            "total_duplicates": len(self.duplicates),
            "by_content_hash": hash_dups,
            "by_normalized_name": name_dups,
            "unique_tracks": len(self.seen_hashes) + len(self.seen_normalized_names) - len(self.seen_hashes)
        }


def create_manifest_entry(track: TrackInfo) -> ManifestEntry:
    """Create a manifest entry from track info."""
    # Determine split
    split = "val" if track.is_validation else "train"

    # Determine instrument name
    if track.instrument_type == "nylon":
        instrument = "Acoustic Nylon Guitar"
    else:
        instrument = "Acoustic Steel Guitar"

    return ManifestEntry(
        track_id=track.track_id,
        midi_path=track.midi_path,
        tab_path=track.jams_path,
        program=track.instrument_program,
        split=split,
        instrument=instrument
    )


def write_manifests(tracks: List[TrackInfo],
                    output_dir: str,
                    prefix: str = "dadagp_acoustic") -> Dict[str, str]:
    """
    Write manifest files for train/val/all splits.

    Returns dict of split_name -> filepath.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Group by split
    train_tracks = [t for t in tracks if not t.is_validation]
    val_tracks = [t for t in tracks if t.is_validation]

    manifests = {
        'train': train_tracks,
        'val': val_tracks,
        'all': tracks
    }

    paths = {}

    for split_name, split_tracks in manifests.items():
        if not split_tracks:
            continue

        filepath = output_path / f"{prefix}_{split_name}.jsonl"
        paths[split_name] = str(filepath)

        with open(filepath, 'w', encoding='utf-8') as f:
            for track in split_tracks:
                entry = create_manifest_entry(track)
                f.write(json.dumps(entry.to_dict()) + '\n')

        logger.info(f"Wrote {len(split_tracks)} entries to {filepath}")

    return paths


@dataclass
class ManifestStats:
    """Statistics for manifest generation."""
    total_tracks: int = 0
    train_tracks: int = 0
    val_tracks: int = 0
    nylon_tracks: int = 0
    steel_tracks: int = 0
    duplicates_removed: int = 0
    unique_songs: int = 0
    unique_artists: int = 0

    def compute_from_tracks(self, tracks: List[TrackInfo],
                             duplicates: int = 0):
        """Compute stats from track list."""
        self.total_tracks = len(tracks)
        self.train_tracks = sum(1 for t in tracks if not t.is_validation)
        self.val_tracks = sum(1 for t in tracks if t.is_validation)
        self.nylon_tracks = sum(1 for t in tracks if t.instrument_type == "nylon")
        self.steel_tracks = sum(1 for t in tracks if t.instrument_type == "steel")
        self.duplicates_removed = duplicates

        # Count unique songs and artists
        artists = set()
        songs = set()
        for t in tracks:
            artists.add(t.artist.lower())
            songs.add(f"{t.artist.lower()}|{t.song.lower()}")

        self.unique_artists = len(artists)
        self.unique_songs = len(songs)

    def __str__(self) -> str:
        return f"""Manifest Statistics:
  Total tracks: {self.total_tracks}
  Training: {self.train_tracks}
  Validation: {self.val_tracks}
  Nylon: {self.nylon_tracks}
  Steel: {self.steel_tracks}
  Duplicates removed: {self.duplicates_removed}
  Unique songs: {self.unique_songs}
  Unique artists: {self.unique_artists}"""


if __name__ == "__main__":
    import sys

    # Test metadata loading
    dadagp_path = "/data/andreaguz/DadaGP-v1.1"

    print("Loading DadaGP metadata...")
    metadata = load_dadagp_metadata(dadagp_path)
    print(f"Loaded {len(metadata)} file entries")

    # Count validation vs training
    val_count = sum(1 for v in metadata.values() if v)
    train_count = len(metadata) - val_count
    print(f"Training: {train_count}, Validation: {val_count}")

    # Test extraction
    test_file = "/data/andreaguz/DadaGP-v1.1/D/De mono/De Mono - Znow Jestes Ze Mna.gp4"
    artist, song = extract_artist_song(test_file)
    print(f"\nTest extraction:")
    print(f"  File: {test_file}")
    print(f"  Artist: {artist}")
    print(f"  Song: {song}")
    print(f"  Is validation: {is_validation_file(test_file, metadata)}")

    # Test normalization
    test_names = [
        "Song Name",
        "Song Name (2)",
        "Song Name (Acoustic)",
        "Song Name - Live",
        "SONG name",
    ]
    print("\nNormalization test:")
    for name in test_names:
        print(f"  '{name}' -> '{normalize_song_name(name)}'")

    # Test duplicate detection
    print("\nDuplicate detection test:")
    detector = DuplicateDetector()

    # Create fake tracks
    fake_notes = [TabNote(0, 100, 1, 5, 69, 80), TabNote(100, 100, 2, 7, 71, 80)]
    hash1 = compute_content_hash(fake_notes)

    track1 = TrackInfo(
        track_id="track1",
        source_file="file1.gp4",
        artist="Artist",
        song="Song",
        track_index=0,
        track_name="Guitar",
        instrument_type="steel",
        instrument_program=25,
        midi_path="path1.mid",
        jams_path="path1.jams",
        note_count=2,
        content_hash=hash1
    )

    track2 = TrackInfo(
        track_id="track2",
        source_file="file2.gp4",
        artist="Artist",
        song="Song (2)",  # Duplicate by name
        track_index=0,
        track_name="Guitar",
        instrument_type="steel",
        instrument_program=25,
        midi_path="path2.mid",
        jams_path="path2.jams",
        note_count=2,
        content_hash=""  # Different hash
    )

    dup1 = detector.check_duplicate(track1)
    print(f"  Track1 duplicate: {dup1}")

    dup2 = detector.check_duplicate(track2)
    print(f"  Track2 duplicate of: {dup2.track_id if dup2 else None}")

    print(f"  Stats: {detector.get_stats()}")
