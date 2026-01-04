#!/usr/bin/env python3
"""
DadaGP Pipeline - Main orchestrator for processing DadaGP dataset.

Processes GuitarPro files to create training data for fret_t5.

Usage:
    python dadagp_pipeline.py --input /path/to/DadaGP-v1.1 --output /path/to/output

Options:
    --input         Path to DadaGP dataset directory
    --output        Path to output directory for processed files
    --manifest-dir  Path to write manifest files (default: fret_t5/data)
    --max-files     Maximum number of files to process (for testing)
    --workers       Number of parallel workers (default: 1)
    --verbose       Enable verbose logging
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dadagp_extractor import (
    find_guitarpro_files,
    load_guitarpro_file,
    extract_notes_from_track,
    extract_tempo_changes,
    get_tuning_midi_pitches,
    TabNote,
)
from dadagp_filters import (
    filter_song_tracks,
    FilterConfig,
    FilterStats,
    normalize_notes_to_standard,
    is_valid_guitar_tuning,
)
from dadagp_to_jams import (
    process_track_to_files,
    generate_track_id,
)
from dadagp_manifest import (
    load_dadagp_metadata,
    is_validation_file,
    extract_artist_song,
    compute_content_hash,
    DuplicateDetector,
    TrackInfo,
    write_manifests,
    ManifestStats,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the pipeline."""
    input_dir: str
    output_dir: str
    manifest_dir: str
    max_files: Optional[int] = None
    workers: int = 1
    verbose: bool = False
    filter_config: FilterConfig = field(default_factory=FilterConfig)


@dataclass
class PipelineStats:
    """Statistics from pipeline run."""
    start_time: float = 0
    end_time: float = 0
    files_found: int = 0
    files_processed: int = 0
    files_with_acoustic: int = 0
    files_failed: int = 0
    tracks_extracted: int = 0
    tracks_after_dedup: int = 0
    train_tracks: int = 0
    val_tracks: int = 0
    nylon_tracks: int = 0
    steel_tracks: int = 0
    total_notes: int = 0
    errors: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        duration = self.end_time - self.start_time if self.end_time else 0
        return f"""Pipeline Statistics:
  Duration: {duration:.1f} seconds
  Files found: {self.files_found}
  Files processed: {self.files_processed}
  Files with acoustic tracks: {self.files_with_acoustic} ({100*self.files_with_acoustic/max(1,self.files_processed):.1f}%)
  Files failed: {self.files_failed}
  Tracks extracted: {self.tracks_extracted}
  Tracks after dedup: {self.tracks_after_dedup}
  Train tracks: {self.train_tracks}
  Validation tracks: {self.val_tracks}
  Nylon: {self.nylon_tracks}
  Steel: {self.steel_tracks}
  Total notes: {self.total_notes}
  Errors: {len(self.errors)}"""


def process_single_file(filepath: str,
                        output_dir: str,
                        metadata: Dict[str, bool],
                        filter_config: FilterConfig) -> List[TrackInfo]:
    """
    Process a single GuitarPro file.

    Returns list of TrackInfo for all extracted acoustic tracks.
    """
    tracks = []

    try:
        # Load the file
        song = load_guitarpro_file(filepath)
        if song is None:
            return tracks

        # Filter for acoustic tracks
        results = filter_song_tracks(song, filter_config)
        passed = [r for r in results if r.passed]

        if not passed:
            return tracks

        # Get tempo changes
        tempo_changes = extract_tempo_changes(song)

        # Check if validation file
        is_val = is_validation_file(filepath, metadata)

        # Extract artist and song
        artist, song_name = extract_artist_song(filepath)

        # Process each passed track
        for r in passed:
            track = r.track
            tuning = get_tuning_midi_pitches(track)
            _, tuning_type = is_valid_guitar_tuning(tuning)

            notes = extract_notes_from_track(track, song)
            if not notes:
                continue

            # Generate track ID
            track_id = generate_track_id(filepath, r.track_index, track.name, r.instrument_type)

            # Compute content hash for deduplication
            normalized_notes = normalize_notes_to_standard(notes, tuning, tuning_type)
            content_hash = compute_content_hash(normalized_notes)

            # Process to files
            try:
                midi_path, jams_path = process_track_to_files(
                    notes=notes,
                    tuning=tuning,
                    tuning_type=tuning_type,
                    tempo_changes=tempo_changes,
                    instrument_type=r.instrument_type,
                    output_dir=output_dir,
                    track_id=track_id
                )

                tracks.append(TrackInfo(
                    track_id=track_id,
                    source_file=filepath,
                    artist=artist,
                    song=song_name,
                    track_index=r.track_index,
                    track_name=track.name,
                    instrument_type=r.instrument_type,
                    instrument_program=24 if r.instrument_type == "nylon" else 25,
                    midi_path=midi_path,
                    jams_path=jams_path,
                    note_count=len(notes),
                    is_validation=is_val,
                    content_hash=content_hash
                ))

            except Exception as e:
                logger.warning(f"Failed to write files for {track_id}: {e}")
                continue

    except Exception as e:
        logger.error(f"Failed to process {filepath}: {e}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(traceback.format_exc())

    return tracks


def run_pipeline(config: PipelineConfig) -> PipelineStats:
    """
    Run the full pipeline.

    Returns statistics about the run.
    """
    stats = PipelineStats()
    stats.start_time = time.time()

    # Set up logging
    log_level = logging.DEBUG if config.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logger.info(f"Starting DadaGP pipeline")
    logger.info(f"  Input: {config.input_dir}")
    logger.info(f"  Output: {config.output_dir}")
    logger.info(f"  Manifest dir: {config.manifest_dir}")

    # Create output directories
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.manifest_dir).mkdir(parents=True, exist_ok=True)

    # Load DadaGP metadata for train/val splits
    logger.info("Loading DadaGP metadata...")
    metadata = load_dadagp_metadata(config.input_dir)
    logger.info(f"Loaded {len(metadata)} file entries")

    # Find all GuitarPro files
    logger.info("Finding GuitarPro files...")
    all_files = list(find_guitarpro_files(config.input_dir))
    stats.files_found = len(all_files)
    logger.info(f"Found {stats.files_found} GuitarPro files")

    # Limit files if requested
    if config.max_files:
        all_files = all_files[:config.max_files]
        logger.info(f"Processing first {len(all_files)} files")

    # Process files
    all_tracks: List[TrackInfo] = []
    filter_stats = FilterStats()

    logger.info("Processing files...")
    for i, filepath in enumerate(all_files):
        if (i + 1) % 100 == 0 or i == 0:
            logger.info(f"Processing file {i+1}/{len(all_files)}: {len(all_tracks)} tracks so far")

        try:
            tracks = process_single_file(
                filepath=filepath,
                output_dir=config.output_dir,
                metadata=metadata,
                filter_config=config.filter_config
            )

            if tracks:
                stats.files_with_acoustic += 1
                all_tracks.extend(tracks)
                for t in tracks:
                    stats.total_notes += t.note_count

            stats.files_processed += 1

        except Exception as e:
            stats.files_failed += 1
            stats.errors.append(f"{filepath}: {e}")
            logger.warning(f"Failed to process {filepath}: {e}")

    stats.tracks_extracted = len(all_tracks)
    logger.info(f"Extracted {stats.tracks_extracted} tracks from {stats.files_with_acoustic} files")

    # Deduplicate
    logger.info("Removing duplicates...")
    detector = DuplicateDetector()
    unique_tracks = []

    for track in all_tracks:
        if detector.check_duplicate(track) is None:
            unique_tracks.append(track)

    dup_stats = detector.get_stats()
    logger.info(f"Removed {dup_stats['total_duplicates']} duplicates")
    logger.info(f"  By content hash: {dup_stats['by_content_hash']}")
    logger.info(f"  By normalized name: {dup_stats['by_normalized_name']}")

    stats.tracks_after_dedup = len(unique_tracks)

    # Count by type
    for t in unique_tracks:
        if t.is_validation:
            stats.val_tracks += 1
        else:
            stats.train_tracks += 1

        if t.instrument_type == "nylon":
            stats.nylon_tracks += 1
        else:
            stats.steel_tracks += 1

    # Write manifests
    logger.info("Writing manifests...")
    manifest_paths = write_manifests(
        tracks=unique_tracks,
        output_dir=config.manifest_dir,
        prefix="dadagp_acoustic"
    )

    for split, path in manifest_paths.items():
        logger.info(f"  {split}: {path}")

    # Save processing stats
    stats.end_time = time.time()
    stats_path = Path(config.manifest_dir) / "dadagp_processing_stats.json"
    with open(stats_path, 'w') as f:
        json.dump({
            'duration_seconds': stats.end_time - stats.start_time,
            'files_found': stats.files_found,
            'files_processed': stats.files_processed,
            'files_with_acoustic': stats.files_with_acoustic,
            'files_failed': stats.files_failed,
            'tracks_extracted': stats.tracks_extracted,
            'tracks_after_dedup': stats.tracks_after_dedup,
            'train_tracks': stats.train_tracks,
            'val_tracks': stats.val_tracks,
            'nylon_tracks': stats.nylon_tracks,
            'steel_tracks': stats.steel_tracks,
            'total_notes': stats.total_notes,
            'errors_count': len(stats.errors)
        }, f, indent=2)

    logger.info(f"\nProcessing complete!")
    logger.info(str(stats))

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Process DadaGP GuitarPro files for fret_t5 training"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="/data/andreaguz/DadaGP-v1.1",
        help="Path to DadaGP dataset directory"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="/data/andreaguz/fret_t5_clean/fret_t5/data/dadagp_processed",
        help="Path to output directory for processed files"
    )
    parser.add_argument(
        "--manifest-dir", "-m",
        type=str,
        default="/data/andreaguz/fret_t5_clean/fret_t5/data",
        help="Path to write manifest files"
    )
    parser.add_argument(
        "--max-files", "-n",
        type=int,
        default=None,
        help="Maximum number of files to process (for testing)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--min-notes",
        type=int,
        default=10,
        help="Minimum notes per track"
    )
    parser.add_argument(
        "--max-notes",
        type=int,
        default=2000,
        help="Maximum notes per track"
    )

    args = parser.parse_args()

    # Create config
    filter_config = FilterConfig(
        min_notes=args.min_notes,
        max_notes=args.max_notes
    )

    config = PipelineConfig(
        input_dir=args.input,
        output_dir=args.output,
        manifest_dir=args.manifest_dir,
        max_files=args.max_files,
        workers=args.workers,
        verbose=args.verbose,
        filter_config=filter_config
    )

    # Run pipeline
    stats = run_pipeline(config)

    # Exit with error code if there were failures
    if stats.files_failed > stats.files_processed * 0.1:  # More than 10% failure
        sys.exit(1)


if __name__ == "__main__":
    main()
