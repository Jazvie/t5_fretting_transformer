"""
Unified data loaders for various input formats.

This module provides a clean interface for loading data from:
- General MIDI files (for inference, no ground truth)
- Custom JSONL manifests
- Various tablature formats

Example Usage:
    # Load MIDI notes from a file
    from fret_t5.loaders import load_midi_notes, MidiNote
    
    notes = load_midi_notes("my_song.mid")
    for note in notes:
        print(f"Pitch: {note.pitch}, Start: {note.start:.2f}s, Duration: {note.duration:.2f}s")
    
    # Create inference dataset from MIDI files
    from fret_t5.loaders import MidiInferenceDataset
    
    dataset = MidiInferenceDataset(
        midi_files=["song1.mid", "song2.mid"],
        tokenizer=tokenizer,
    )
    
    # Load custom manifest
    from fret_t5.loaders import load_custom_manifest
    
    entries = load_custom_manifest("my_data.jsonl")
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Iterator

import torch
from torch.utils.data import Dataset

try:
    import pretty_midi
    HAS_PRETTY_MIDI = True
except ImportError:
    HAS_PRETTY_MIDI = False

from .tokenization import MidiTabTokenizerV3, STANDARD_TUNING


__all__ = [
    "DataFormat",
    "MidiNote",
    "load_midi_notes",
    "load_midi_file",
    "MidiInferenceDataset",
    "load_custom_manifest",
    "CustomManifestEntry",
]


class DataFormat(Enum):
    """Supported data formats."""
    SYNTHTAB_JAMS = "synthtab"      # JAMS with note_tab namespace
    GUITARSET_JAMS = "guitarset"    # JAMS with per-string note_midi
    DADAGP = "dadagp"               # Extracted from GuitarPro
    GENERAL_MIDI = "midi"           # Standard MIDI (inference only, no ground truth)
    CUSTOM_JSONL = "custom"         # User-provided JSONL manifest


@dataclass
class MidiNote:
    """Represents a single MIDI note event.
    
    Attributes:
        pitch: MIDI pitch (0-127)
        start: Onset time in seconds
        duration: Duration in seconds
        velocity: Note velocity (0-127), optional
        instrument: Instrument/track index, optional
    """
    pitch: int
    start: float
    duration: float
    velocity: int = 100
    instrument: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format used by inference pipeline."""
        return {
            'pitch': self.pitch,
            'start': self.start,
            'duration': self.duration,
            'velocity': self.velocity,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MidiNote':
        """Create from dictionary."""
        return cls(
            pitch=int(data['pitch']),
            start=float(data.get('start', data.get('onset', 0))),
            duration=float(data['duration']),
            velocity=int(data.get('velocity', 100)),
            instrument=int(data.get('instrument', 0)),
        )


@dataclass
class CustomManifestEntry:
    """Entry from a custom manifest file.
    
    Supports flexible field names for compatibility with various formats.
    """
    midi_path: Optional[Path] = None
    tab_path: Optional[Path] = None
    audio_path: Optional[Path] = None
    notes: Optional[List[Dict]] = None  # Pre-extracted notes
    track_id: str = ""
    split: str = "train"
    metadata: Optional[Dict] = None


def load_midi_notes(
    midi_path: Union[str, Path],
    instrument_filter: Optional[List[int]] = None,
    min_pitch: int = 28,
    max_pitch: int = 96,
    min_duration: float = 0.01,
) -> List[MidiNote]:
    """Load MIDI notes from a file.
    
    Args:
        midi_path: Path to MIDI file
        instrument_filter: List of instrument indices to include (None = all)
        min_pitch: Minimum MIDI pitch to include (default: 28, low guitar range)
        max_pitch: Maximum MIDI pitch to include (default: 96, high guitar range)
        min_duration: Minimum note duration in seconds
        
    Returns:
        List of MidiNote objects sorted by start time
        
    Example:
        >>> notes = load_midi_notes("song.mid")
        >>> print(f"Found {len(notes)} notes")
        >>> for note in notes[:5]:
        ...     print(f"  {note.pitch} at {note.start:.2f}s")
    """
    if not HAS_PRETTY_MIDI:
        raise ImportError("pretty_midi is required for MIDI loading. Install with: pip install pretty_midi")
    
    midi_path = Path(midi_path)
    if not midi_path.exists():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")
    
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    notes: List[MidiNote] = []
    
    for inst_idx, instrument in enumerate(midi.instruments):
        # Skip drums
        if instrument.is_drum:
            continue
            
        # Apply instrument filter
        if instrument_filter is not None and inst_idx not in instrument_filter:
            continue
        
        for note in instrument.notes:
            # Filter by pitch range
            if not (min_pitch <= note.pitch <= max_pitch):
                continue
                
            duration = note.end - note.start
            
            # Filter by duration
            if duration < min_duration:
                continue
            
            notes.append(MidiNote(
                pitch=note.pitch,
                start=note.start,
                duration=duration,
                velocity=note.velocity,
                instrument=inst_idx,
            ))
    
    # Sort by start time, then pitch
    notes.sort(key=lambda n: (n.start, n.pitch))
    
    return notes


def load_midi_file(
    midi_path: Union[str, Path],
    **kwargs
) -> Tuple[List[MidiNote], Dict]:
    """Load MIDI file and return notes with metadata.
    
    Args:
        midi_path: Path to MIDI file
        **kwargs: Arguments passed to load_midi_notes
        
    Returns:
        Tuple of (notes, metadata) where metadata contains:
        - duration: Total file duration in seconds
        - tempo: Estimated tempo in BPM
        - time_signature: Time signature as (numerator, denominator)
        - num_instruments: Number of non-drum instruments
    """
    if not HAS_PRETTY_MIDI:
        raise ImportError("pretty_midi is required for MIDI loading")
    
    midi_path = Path(midi_path)
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    
    notes = load_midi_notes(midi_path, **kwargs)
    
    # Extract metadata
    tempo_times, tempos = midi.get_tempo_changes()
    estimated_tempo = tempos[0] if len(tempos) > 0 else 120.0
    
    time_sig = midi.time_signature_changes
    if time_sig:
        ts = (time_sig[0].numerator, time_sig[0].denominator)
    else:
        ts = (4, 4)
    
    metadata = {
        'duration': midi.get_end_time(),
        'tempo': estimated_tempo,
        'time_signature': ts,
        'num_instruments': len([i for i in midi.instruments if not i.is_drum]),
        'file_path': str(midi_path),
    }
    
    return notes, metadata


def load_custom_manifest(
    manifest_path: Union[str, Path],
    base_dir: Optional[Path] = None,
) -> List[CustomManifestEntry]:
    """Load entries from a custom JSONL manifest.
    
    The manifest should be a JSONL file where each line is a JSON object.
    Supported fields:
    - midi_path: Path to MIDI file
    - tab_path: Path to tablature file (JAMS format)
    - audio_path: Path to audio file
    - notes: Pre-extracted note list
    - track_id / id: Unique identifier
    - split: Data split (train/val/test)
    - Any additional fields stored in metadata
    
    Args:
        manifest_path: Path to JSONL manifest
        base_dir: Base directory for resolving relative paths
        
    Returns:
        List of CustomManifestEntry objects
        
    Example manifest (my_data.jsonl):
        {"midi_path": "songs/song1.mid", "split": "train", "id": "song1"}
        {"midi_path": "songs/song2.mid", "split": "val", "id": "song2"}
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    if base_dir is None:
        base_dir = manifest_path.parent
    
    entries: List[CustomManifestEntry] = []
    
    with manifest_path.open('r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON on line {line_num}: {e}")
                continue
            
            # Extract known fields
            entry = CustomManifestEntry(
                track_id=str(record.get('track_id', record.get('id', f'entry_{line_num}'))),
                split=str(record.get('split', 'train')),
            )
            
            # Handle paths
            if 'midi_path' in record:
                path = Path(record['midi_path'])
                entry.midi_path = path if path.is_absolute() else base_dir / path
            
            if 'tab_path' in record:
                path = Path(record['tab_path'])
                entry.tab_path = path if path.is_absolute() else base_dir / path
            
            if 'audio_path' in record:
                path = Path(record['audio_path'])
                entry.audio_path = path if path.is_absolute() else base_dir / path
            
            # Pre-extracted notes
            if 'notes' in record:
                entry.notes = record['notes']
            
            # Store remaining fields as metadata
            known_fields = {'midi_path', 'tab_path', 'audio_path', 'notes', 
                          'track_id', 'id', 'split'}
            entry.metadata = {k: v for k, v in record.items() if k not in known_fields}
            
            entries.append(entry)
    
    return entries


class MidiInferenceDataset(Dataset):
    """Dataset for inference on MIDI files (no ground truth tablature).
    
    This dataset loads MIDI files and prepares them for tablature inference.
    Unlike training datasets, this doesn't require ground truth annotations.
    
    Example:
        >>> dataset = MidiInferenceDataset(
        ...     midi_files=["song1.mid", "song2.mid"],
        ...     tokenizer=tokenizer,
        ...     capo=0,
        ...     tuning=STANDARD_TUNING,
        ... )
        >>> 
        >>> for batch in DataLoader(dataset, batch_size=1):
        ...     predictions = model.generate(batch['input_ids'])
    """
    
    def __init__(
        self,
        midi_files: List[Union[str, Path]],
        tokenizer: MidiTabTokenizerV3,
        capo: int = 0,
        tuning: Tuple[int, ...] = STANDARD_TUNING,
        max_notes_per_chunk: int = 150,
        overlap_notes: int = 4,
        max_encoder_length: int = 512,
    ):
        """Initialize MIDI inference dataset.
        
        Args:
            midi_files: List of MIDI file paths
            tokenizer: Tokenizer instance
            capo: Capo position for conditioning
            tuning: Tuning for conditioning
            max_notes_per_chunk: Maximum notes per chunk
            overlap_notes: Overlap between chunks for continuity
            max_encoder_length: Maximum encoder sequence length
        """
        self.tokenizer = tokenizer
        self.capo = capo
        self.tuning = tuning
        self.max_notes_per_chunk = max_notes_per_chunk
        self.overlap_notes = overlap_notes
        self.max_encoder_length = max_encoder_length
        
        # Ensure conditioning tokens exist
        tokenizer.ensure_conditioning_tokens(
            capo_values=(capo,),
            tuning_options=(tuning,)
        )
        
        # Load all MIDI files and create chunks
        self.chunks: List[Dict] = []
        self._load_midi_files(midi_files)
    
    def _load_midi_files(self, midi_files: List[Union[str, Path]]) -> None:
        """Load and chunk all MIDI files."""
        for midi_path in midi_files:
            try:
                notes, metadata = load_midi_file(midi_path)
                
                if not notes:
                    print(f"Warning: No notes found in {midi_path}")
                    continue
                
                # Chunk the notes
                chunks = self._chunk_notes(notes, metadata)
                self.chunks.extend(chunks)
                
            except Exception as e:
                print(f"Warning: Failed to load {midi_path}: {e}")
    
    def _chunk_notes(
        self, 
        notes: List[MidiNote], 
        metadata: Dict
    ) -> List[Dict]:
        """Split notes into chunks for processing."""
        chunks = []
        
        start_idx = 0
        chunk_idx = 0
        
        while start_idx < len(notes):
            end_idx = min(start_idx + self.max_notes_per_chunk, len(notes))
            chunk_notes = notes[start_idx:end_idx]
            
            # Create encoder tokens
            encoder_tokens = self._notes_to_encoder_tokens(chunk_notes)
            
            # Add conditioning prefix
            prefix = self.tokenizer.build_conditioning_prefix(self.capo, self.tuning)
            full_tokens = prefix + encoder_tokens
            
            # Truncate if needed
            full_tokens = full_tokens[:self.max_encoder_length]
            
            # Encode
            input_ids = self.tokenizer.encode_encoder_tokens_shared(full_tokens)
            
            chunks.append({
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.ones(len(input_ids), dtype=torch.long),
                'notes': [n.to_dict() for n in chunk_notes],
                'metadata': metadata,
                'chunk_idx': chunk_idx,
                'is_first_chunk': chunk_idx == 0,
            })
            
            # Move to next chunk with overlap
            start_idx += self.max_notes_per_chunk - self.overlap_notes
            chunk_idx += 1
        
        return chunks
    
    def _notes_to_encoder_tokens(self, notes: List[MidiNote]) -> List[str]:
        """Convert notes to encoder tokens."""
        tokens = []
        
        # Sort by start time
        sorted_notes = sorted(notes, key=lambda n: (n.start, n.pitch))
        
        for i, note in enumerate(sorted_notes):
            # Quantize duration to 100ms
            dur_ms = int(round(note.duration * 1000 / 100)) * 100
            dur_ms = max(100, min(5000, dur_ms))  # Clamp to valid range
            
            # Check if this is part of a chord (same onset as next note)
            is_chord = False
            if i < len(sorted_notes) - 1:
                next_note = sorted_notes[i + 1]
                if abs(next_note.start - note.start) < 0.01:  # Within 10ms
                    is_chord = True
            
            # Use 0 time shift for chord notes
            time_shift = 0 if is_chord else dur_ms
            
            tokens.extend([
                f"NOTE_ON<{note.pitch}>",
                f"TIME_SHIFT<{time_shift}>",
                f"NOTE_OFF<{note.pitch}>"
            ])
        
        return tokens
    
    def __len__(self) -> int:
        return len(self.chunks)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chunk = self.chunks[idx]
        return {
            'input_ids': chunk['input_ids'],
            'attention_mask': chunk['attention_mask'],
        }
    
    def get_chunk_metadata(self, idx: int) -> Dict:
        """Get metadata for a specific chunk."""
        return {
            'notes': self.chunks[idx]['notes'],
            'metadata': self.chunks[idx]['metadata'],
            'chunk_idx': self.chunks[idx]['chunk_idx'],
            'is_first_chunk': self.chunks[idx]['is_first_chunk'],
        }


def create_manifest_from_midi_dir(
    midi_dir: Union[str, Path],
    output_path: Union[str, Path],
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    extensions: Tuple[str, ...] = ('.mid', '.midi'),
    seed: int = 42,
) -> Path:
    """Create a JSONL manifest from a directory of MIDI files.
    
    Args:
        midi_dir: Directory containing MIDI files
        output_path: Path for output manifest
        split_ratios: (train, val, test) split ratios
        extensions: File extensions to include
        seed: Random seed for splitting
        
    Returns:
        Path to created manifest
        
    Example:
        >>> create_manifest_from_midi_dir(
        ...     "my_midi_files/",
        ...     "data/my_manifest.jsonl",
        ...     split_ratios=(0.7, 0.15, 0.15)
        ... )
    """
    import random
    
    midi_dir = Path(midi_dir)
    output_path = Path(output_path)
    
    # Find all MIDI files
    midi_files = []
    for ext in extensions:
        midi_files.extend(midi_dir.glob(f"**/*{ext}"))
    
    if not midi_files:
        raise ValueError(f"No MIDI files found in {midi_dir}")
    
    print(f"Found {len(midi_files)} MIDI files")
    
    # Shuffle and split
    random.seed(seed)
    random.shuffle(midi_files)
    
    n = len(midi_files)
    train_end = int(n * split_ratios[0])
    val_end = int(n * (split_ratios[0] + split_ratios[1]))
    
    splits = {
        'train': midi_files[:train_end],
        'val': midi_files[train_end:val_end],
        'test': midi_files[val_end:],
    }
    
    # Write manifest
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open('w', encoding='utf-8') as f:
        for split_name, files in splits.items():
            for midi_file in files:
                entry = {
                    'midi_path': str(midi_file),
                    'track_id': midi_file.stem,
                    'split': split_name,
                }
                f.write(json.dumps(entry) + '\n')
    
    print(f"Created manifest at {output_path}")
    print(f"  Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    
    return output_path
