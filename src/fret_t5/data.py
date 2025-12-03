"""Dataset utilities for SynthTab acoustic guitar training."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import pretty_midi
import torch
from torch.utils.data import Dataset

from .tokenization import (
    MidiTabTokenizerV3,
    TokenizedTrack,
    DEFAULT_CONDITIONING_TUNINGS,
    STANDARD_TUNING,
    NoteMetadata,
)

__all__ = [
    "SynthTabManifestEntry",
    "DataConfig",
    "SynthTabTokenDataset",
    "load_manifest",
    "chunk_tokenized_track",
    "create_song_level_splits",
]


@dataclass
class SynthTabManifestEntry:
    """Represents a single SynthTab track entry."""

    midi_path: Path
    tab_path: Path
    program: int
    split: str
    track_id: str


def load_manifest(path: Path) -> List[SynthTabManifestEntry]:
    """Load a JSONL manifest describing SynthTab tracks."""

    entries: List[SynthTabManifestEntry] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            record = json.loads(line)
            entries.append(
                SynthTabManifestEntry(
                    midi_path=Path(record["midi_path"]),
                    tab_path=Path(record["tab_path"]),
                    program=int(record.get("program", -1)),
                    split=str(record.get("split", "train")),
                    track_id=str(record.get("track_id", record.get("id", ""))),
                )
            )
    return entries


@dataclass
class TempoMap:
    """Represents a tempo map capable of converting ticks to time."""

    ticks_per_beat: float
    tempo_changes: List[Tuple[float, float]]

    def __post_init__(self) -> None:
        if not self.tempo_changes:
            raise ValueError("Tempo map must contain at least one tempo change")

        # Ensure tempo changes are sorted and start at tick 0 for integration.
        self.tempo_changes.sort(key=lambda change: change[0])
        first_tick, first_tempo = self.tempo_changes[0]
        if first_tick > 0:
            self.tempo_changes.insert(0, (0.0, first_tempo))

    def tick_to_time(self, tick: float) -> float:
        """Convert absolute tick position to seconds."""

        if tick <= 0:
            return 0.0

        total_seconds = 0.0
        previous_tick, previous_tempo = self.tempo_changes[0]
        seconds_per_tick = 60.0 / (previous_tempo * self.ticks_per_beat)

        for change_tick, tempo in self.tempo_changes[1:]:
            if tick <= change_tick:
                total_seconds += max(0.0, tick - previous_tick) * seconds_per_tick
                return total_seconds

            total_seconds += max(0.0, change_tick - previous_tick) * seconds_per_tick
            previous_tick = change_tick
            previous_tempo = tempo
            seconds_per_tick = 60.0 / (previous_tempo * self.ticks_per_beat)

        total_seconds += max(0.0, tick - previous_tick) * seconds_per_tick
        return total_seconds

    def duration_ms(self, start_tick: float, duration_ticks: float) -> float:
        """Convert a duration starting at ``start_tick`` into milliseconds."""

        if duration_ticks <= 0:
            return 0.0

        start_time = self.tick_to_time(start_tick)
        end_time = self.tick_to_time(start_tick + duration_ticks)
        return max(0.0, end_time - start_time) * 1000.0


@dataclass
class DataConfig:
    """Configuration controlling dataset chunking and filtering."""

    max_encoder_length: int = 512
    max_decoder_length: int = 512
    overlap_notes: int = 4
    acoustic_programs: Tuple[int, int] = (25, 26)
    train_on_time_shift: bool = True  # Paper-faithful: train on both TAB and TIME_SHIFT
    tab_loss_weight: float = 1.0      # Optional weight boost for TAB tokens
    enable_conditioning: bool = False
    conditioning_capo_values_train: Tuple[int, ...] = tuple(range(8))
    conditioning_capo_values_eval: Tuple[int, ...] = (0,)
    conditioning_tunings_train: Tuple[Tuple[int, ...], ...] = DEFAULT_CONDITIONING_TUNINGS
    conditioning_tunings_eval: Tuple[Tuple[int, ...], ...] = (STANDARD_TUNING,)
    randomize_tuning_per_sequence: bool = True
    augmentation_seed: int = 1337


def _load_tempo_map(tab_path: Path, midi_path: Optional[Path] = None) -> Optional[TempoMap]:
    """Load a tempo map from PrettyMIDI or SynthTab tempo files."""

    midi_candidates: List[Path] = []
    if midi_path is not None:
        midi_candidates.append(midi_path)
        if not midi_path.is_absolute():
            midi_candidates.append(tab_path.parent / midi_path)

    for candidate in midi_candidates:
        if not candidate.exists():
            continue
        try:
            return _tempo_map_from_pretty_midi(candidate)
        except Exception:
            pass # Failed to read tempo from MIDI, try next candidate

    tempo_file = tab_path.parent / "tempo.txt"
    if tempo_file.exists():
        try:
            return _tempo_map_from_tempo_file(tempo_file)
        except Exception:
            pass # Failed to parse tempo map from file, return None

    return None


def _tempo_map_from_pretty_midi(midi_path: Path) -> TempoMap:
    """Create a :class:`TempoMap` from a MIDI file using PrettyMIDI."""

    midi = pretty_midi.PrettyMIDI(str(midi_path))
    tempo_times, tempi = midi.get_tempo_changes()
    tick_positions = [float(midi._time_to_tick(time)) for time in tempo_times]
    tempo_changes = list(zip(tick_positions, (float(t) for t in tempi)))
    return TempoMap(float(midi.resolution), tempo_changes)


def _tempo_map_from_tempo_file(tempo_path: Path) -> TempoMap:
    """Create a tempo map from SynthTab ``tempo.txt`` files."""

    ticks_per_beat: Optional[float] = None
    tempo_changes: List[Tuple[float, float]] = []
    metadata_tempo: Optional[float] = None

    with tempo_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue

            if ":" in line:
                key, value = [part.strip().lower() for part in line.split(":", 1)]
                if key in {"ticks_per_beat", "tpb"}:
                    ticks_per_beat = float(value)
                    continue
                if key in {"tempo", "bpm"}:
                    try:
                        metadata_tempo = float(value)
                    except ValueError:
                        pass
                    continue

            if "=" in line:
                key, value = [part.strip().lower() for part in line.split("=", 1)]
                if key in {"ticks_per_beat", "tpb"}:
                    ticks_per_beat = float(value)
                    continue
                if key in {"tempo", "bpm"}:
                    try:
                        metadata_tempo = float(value)
                    except ValueError:
                        pass
                    continue

            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue

            tick = float(parts[0])
            tempo = float(parts[1])
            if len(parts) >= 3 and ticks_per_beat is None:
                try:
                    ticks_per_beat = float(parts[2])
                except ValueError:
                    pass
            tempo_changes.append((tick, tempo))

    if not tempo_changes and metadata_tempo is not None:
        tempo_changes.append((0.0, metadata_tempo))

    if not tempo_changes:
        raise ValueError("Tempo file did not contain any tempo changes")
    if ticks_per_beat is None:
        raise ValueError("Tempo file is missing ticks_per_beat information")

    return TempoMap(float(ticks_per_beat), tempo_changes)


def _load_jams_events(path: Path, midi_path: Optional[Path] = None) -> List[Dict[str, float]]:
    """Load tablature events directly from JAMS format."""
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    # Skip tempo loading - use fixed conversion for TIME_SHIFT tokenization
    tempo_map = None

    if isinstance(data, dict) and "annotations" in data:
        events = _parse_jams_events_simple(data, tempo_map)
        if events is None:
            return _parse_jams_tablature(data, midi_dir=path.parent, tempo_map=tempo_map)
        return events
    else:
        raise ValueError(f"Expected JAMS format with annotations, got: {type(data)}")


def _parse_jams_events_simple(
    jams_data: Dict,
    tempo_map: Optional[TempoMap],
) -> Optional[List[Dict[str, float]]]:
    """Parse JAMS file to extract clean tablature events."""
    events: List[Dict[str, float]] = []
    missing_strings = False
    notes_found = False

    for annotation in jams_data.get("annotations", []):
        if annotation.get("namespace") != "note_tab":
            continue

        for note in annotation.get("data", []):
            notes_found = True
            # Extract JAMS fields
            time_ticks = float(note["time"])
            duration_ticks = float(note["duration"])
            value = note.get("value", {})
            fret = int(value["fret"])

            string_value = value.get("string")
            if string_value is None:
                missing_strings = True
                break
            string_num = int(string_value)

            # Skip frets above MAX_FRET (24)
            if fret > 24:
                continue

            # Convert ticks to milliseconds using tempo map when available
            duration_ms = _convert_ticks_to_ms(duration_ticks, time_ticks, tempo_map)

            events.append({
                "string": float(string_num),
                "fret": float(fret),
                "duration_ms": duration_ms,
                "time_ticks": time_ticks
            })

        if missing_strings:
            break

    # Sort events by time
    events.sort(key=lambda x: x["time_ticks"])
    if missing_strings and notes_found:
        return None
    return events


def _parse_jams_tablature(
    jams_data: Dict,
    midi_dir: Path = None,
    tempo_map: Optional[TempoMap] = None,
) -> List[Dict[str, float]]:
    """Parse SynthTab JAMS file to extract tablature events.

    SynthTab JAMS files contain note_tab annotations with:
    - time: onset time in ticks
    - duration: note duration in ticks
    - value.fret: fret number
    - value.velocity: note velocity
    - sandbox.string_index: string number (1-6)

    The string information is read directly from sandbox.string_index,
    eliminating the need for MIDI correlation.
    """
    events = []

    for annotation in jams_data.get("annotations", []):
        if annotation.get("namespace") != "note_tab":
            continue

        # Get string number directly from sandbox metadata
        sandbox = annotation.get("sandbox", {})
        string_num = sandbox.get("string_index")

        if string_num is None:
            print(f"Warning: Missing string_index in sandbox for annotation with {len(annotation.get('data', []))} events")
            continue

        for note in annotation.get("data", []):
            # Extract JAMS fields
            time_ticks = float(note["time"])
            duration_ticks = float(note["duration"])
            fret = int(note["value"]["fret"])

            # Convert ticks to milliseconds
            duration_ms = _convert_ticks_to_ms(duration_ticks, time_ticks, tempo_map)

            # Skip frets above MAX_FRET (24)
            if fret > 24:
                continue

            events.append({
                "string": float(string_num),
                "fret": float(fret),
                "duration_ms": duration_ms,
                "time_ticks": time_ticks
            })

    # Sort events by time
    events.sort(key=lambda x: x["time_ticks"])

    return events


def _load_string_midi_notes(midi_dir: Path) -> Dict[int, List[Dict]]:
    """Load notes from individual string MIDI files for correlation."""
    string_notes = {}

    for string_num in range(1, 7):  # Strings 1-6
        string_file = midi_dir / f"string_{string_num}.mid"
        if not string_file.exists():
            continue

        try:
            midi = pretty_midi.PrettyMIDI(str(string_file))
            notes = []

            for instrument in midi.instruments:
                for note in instrument.notes:
                    # Convert to ticks (approximate)
                    time_ticks = note.start * 480 * 2  # Rough conversion
                    duration_ticks = (note.end - note.start) * 480 * 2

                    notes.append({
                        "time": time_ticks,
                        "duration": duration_ticks,
                        "pitch": note.pitch
                    })

            string_notes[string_num] = notes

        except Exception as e:
            print(f"Warning: Could not load {string_file}: {e}")
            continue

    return string_notes


def _determine_string_from_midi(
    time_ticks: float,
    duration_ticks: float,
    fret: int,
    string_notes: Dict[int, List[Dict]]
) -> int:
    """Determine which string a tablature event belongs to by correlating with MIDI.

    Uses both time correlation and pitch validation to assign strings properly.
    """
    # Standard tuning open string pitches
    open_string_pitches = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}

    # Expected MIDI pitch for this fret
    expected_pitches = {}
    for string_num, open_pitch in open_string_pitches.items():
        expected_pitches[string_num] = open_pitch + fret

    # Try to find string with matching time AND pitch
    best_string = None
    best_score = float('inf')

    for string_num, notes in string_notes.items():
        expected_pitch = expected_pitches.get(string_num)
        if expected_pitch is None:
            continue

        for note in notes:
            time_diff = abs(note["time"] - time_ticks)
            duration_diff = abs(note["duration"] - duration_ticks)

            # Check if time is reasonable (within 200 ticks)
            if time_diff > 200:
                continue

            # Check if pitch matches (within 1 semitone for rounding errors)
            pitch_diff = abs(note.get("pitch", 0) - expected_pitch)
            if pitch_diff > 1:
                continue

            # Score based on time and duration matching
            score = time_diff + (duration_diff * 0.5)  # Weight time more than duration

            if score < best_score:
                best_score = score
                best_string = string_num

    # If correlation failed, use pitch-based fallback
    if best_string is None:
        best_string = _determine_string_from_pitch(fret, string_notes)

    return best_string


def _determine_string_from_pitch(fret: int, string_notes: Dict[int, List[Dict]]) -> int:
    """Fallback string assignment based on pitch and available strings."""
    # Standard tuning open string pitches
    open_string_pitches = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}

    # Find strings that can play this fret
    possible_strings = []
    for string_num, open_pitch in open_string_pitches.items():
        if 0 <= fret <= 24:  # Valid fret range
            # Only consider strings that have MIDI data available
            if string_num in string_notes and len(string_notes[string_num]) > 0:
                possible_strings.append(string_num)

    if possible_strings:
        # Prefer higher-numbered strings (lower pitch) for lower frets
        # This follows typical guitar playing patterns
        if fret <= 3:
            return max(possible_strings)  # Prefer lower strings for low frets
        else:
            return min(possible_strings)  # Prefer higher strings for high frets

    # Last resort: return most common string with available data
    string_with_most_notes = max(string_notes.keys(),
                                key=lambda s: len(string_notes.get(s, [])),
                                default=1)
    return string_with_most_notes


def _convert_ticks_to_ms(
    duration_ticks: float,
    time_ticks: float,
    tempo_map: Optional[TempoMap],
) -> float:
    """Convert MIDI ticks to milliseconds using a tempo map if available."""

    if tempo_map is None:
        # Fall back to the historical approximation if tempo data is missing.
        return duration_ticks * 1.041666

    return tempo_map.duration_ms(time_ticks, duration_ticks)


def combine_string_midis(midi_dir: Path) -> pretty_midi.PrettyMIDI:
    """Combine separate string MIDI files into a single PrettyMIDI object.

    SynthTab stores each string in separate MIDI files (string_1.mid, etc.).
    We need to combine them into a single MIDI for the tokenizer.
    """
    combined_midi = pretty_midi.PrettyMIDI()

    # Load each string MIDI file
    strings_found = 0
    for string_num in range(1, 7):  # Strings 1-6
        string_file = midi_dir / f"string_{string_num}.mid"
        if not string_file.exists():
            continue

        try:
            string_midi = pretty_midi.PrettyMIDI(str(string_file))

            # Add all instruments from this string to the combined MIDI
            for instrument in string_midi.instruments:
                # Set the program to match the acoustic guitar type
                # We'll determine this from the directory name
                combined_midi.instruments.append(instrument)

            strings_found += 1

        except Exception as e:
            print(f"Warning: Could not load {string_file}: {e}")
            continue

    # Only return if we found at least 3 strings (reasonable minimum)
    if strings_found < 3:
        raise ValueError(f"Only found {strings_found} valid string MIDI files, need at least 3")

    return combined_midi


def chunk_tokenized_track(
    track: TokenizedTrack,
    data_config: DataConfig,
) -> Iterator[Tuple[List[str], List[str], List[NoteMetadata]]]:
    """Yield encoder/decoder token chunks without splitting note groups."""

    def _group(tokens: Sequence[str], group_lengths: Sequence[int]) -> List[List[str]]:
        groups: List[List[str]] = []
        cursor = 0
        for length in group_lengths:
            groups.append(list(tokens[cursor : cursor + length]))
            cursor += length
        return groups

    encoder_groups = _group(track.encoder_tokens, track.encoder_group_lengths)
    decoder_groups = _group(track.decoder_tokens, track.decoder_group_lengths)
    note_metadata = track.note_metadata
    assert len(encoder_groups) == len(decoder_groups) == len(note_metadata), "Mismatched note groups"

    start = 0
    total_groups = len(encoder_groups)
    while start < total_groups:
        enc_chunk: List[str] = []
        dec_chunk: List[str] = []
        groups_used = 0
        chunk_metadata: List[NoteMetadata] = []

        decoder_budget = max(data_config.max_decoder_length - 1, 0)

        for group_idx in range(start, total_groups):
            enc_group = encoder_groups[group_idx]
            dec_group = decoder_groups[group_idx]
            if (
                len(enc_chunk) + len(enc_group) > data_config.max_encoder_length
                or (
                    decoder_budget > 0
                    and len(dec_chunk) + len(dec_group) > decoder_budget
                )
                or (
                    decoder_budget == 0
                    and len(dec_chunk) + len(dec_group) >= data_config.max_decoder_length
                )
            ):
                break
            enc_chunk.extend(enc_group)
            dec_chunk.extend(dec_group)
            chunk_metadata.append(note_metadata[group_idx])
            groups_used += 1

        if groups_used == 0:
            # Single group exceeds max length; fall back to truncation.
            enc_chunk = encoder_groups[start][: data_config.max_encoder_length]
            if data_config.max_decoder_length <= 0:
                dec_chunk = []
            elif decoder_budget > 0:
                dec_chunk = decoder_groups[start][:decoder_budget]
            else:
                dec_chunk = []
            groups_used = 1

        if decoder_budget > 0:
            dec_output = list(dec_chunk[:decoder_budget])
        elif data_config.max_decoder_length > 0:
            # No budget for content tokens, reserve EOS only.
            dec_output = []
        else:
            dec_output = list(dec_chunk)

        if data_config.max_decoder_length > 0:
            dec_output.append("<eos>")

        yield enc_chunk, dec_output, chunk_metadata

        if groups_used <= data_config.overlap_notes:
            start += 1
        else:
            start += groups_used - data_config.overlap_notes


class SynthTabTokenDataset(Dataset):
    """Token-level dataset for SynthTab acoustic guitar tracks."""

    def __init__(
        self,
        tokenizer: MidiTabTokenizerV3,
        manifests: Sequence[Path],
        data_config: DataConfig,
        split: str,
        preload: bool = True,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.data_config = data_config
        self.split = split
        self.examples: List[Dict[str, torch.Tensor]] = []
        self._rng = random.Random(self.data_config.augmentation_seed)
        if preload:
            self._preload_examples(manifests)

    def _preload_examples(self, manifests: Sequence[Path]) -> None:
        split_name = self.split.lower()
        is_training_split = split_name in {"train", "all"}

        if self.data_config.enable_conditioning:
            capo_values = (
                self.data_config.conditioning_capo_values_train
                if is_training_split
                else self.data_config.conditioning_capo_values_eval
            )
            if not capo_values:
                capo_values = (0,)

            tuning_options = (
                self.data_config.conditioning_tunings_train
                if is_training_split
                else self.data_config.conditioning_tunings_eval
            )
            if not tuning_options:
                tuning_options = (STANDARD_TUNING,)

            self.tokenizer.ensure_conditioning_tokens(capo_values, tuning_options)

            sample_prefix = self.tokenizer.build_conditioning_prefix(capo_values[0], tuning_options[0])
            reserved_encoder_length = max(1, self.data_config.max_encoder_length - len(sample_prefix))
            chunk_config = replace(self.data_config, max_encoder_length=reserved_encoder_length)
        else:
            capo_values = (0,)
            tuning_options = (STANDARD_TUNING,)
            chunk_config = self.data_config

        for manifest in manifests:
            for entry in load_manifest(manifest):
                if entry.split != self.split:
                    continue
                if entry.program not in self.data_config.acoustic_programs:
                    continue

                try:
                    # Load JAMS events directly
                    jams_events = _load_jams_events(entry.tab_path, entry.midi_path)

                    # Tokenize directly from JAMS (perfect alignment guaranteed)
                    tokenised = self.tokenizer.tokenize_track_from_jams(jams_events)

                except Exception as e:
                    print(f"Warning: Failed to process {entry.track_id}: {e}")
                    continue
                chunks = list(chunk_tokenized_track(tokenised, chunk_config))

                if not self.data_config.enable_conditioning:
                    for enc_tokens, dec_tokens, _ in chunks:
                        self._append_example(enc_tokens, dec_tokens)
                    continue

                for enc_tokens, dec_tokens, note_metadata in chunks:
                    for capo in capo_values:
                        if is_training_split and self.data_config.randomize_tuning_per_sequence:
                            tuning_choices = [self._rng.choice(tuning_options)]
                        else:
                            tuning_choices = list(tuning_options)

                        for tuning in tuning_choices:
                            conditioned_encoder = self._apply_conditioning_to_encoder_tokens(
                                enc_tokens,
                                note_metadata,
                                capo,
                                tuning,
                            )
                            prefix_tokens = self.tokenizer.build_conditioning_prefix(capo, tuning)
                            final_encoder_tokens = prefix_tokens + conditioned_encoder

                            self._append_example(final_encoder_tokens, list(dec_tokens))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]

    def _append_example(self, encoder_tokens: Sequence[str], decoder_tokens: Sequence[str]) -> None:
        input_ids = self.tokenizer.encode_encoder_tokens_shared(encoder_tokens)
        labels = self.tokenizer.encode_decoder_tokens_shared(decoder_tokens)

        input_ids = input_ids[:self.data_config.max_encoder_length]
        labels = labels[:self.data_config.max_decoder_length]

        attention_mask = [1] * len(input_ids)

        pad_id = self.tokenizer.shared_token_to_id.get("<pad>", 0)
        while len(input_ids) < self.data_config.max_encoder_length:
            input_ids.append(pad_id)
            attention_mask.append(0)

        # Loss mask
        if self.data_config.train_on_time_shift:
            loss_values = [
                self.data_config.tab_loss_weight if token.startswith("TAB<")
                else 1.0 if token.startswith("TIME_SHIFT<")
                else 1.0 if token == "<eos>"
                else 0.0
                for token in decoder_tokens
            ]
        else:
            loss_values = [
                1.0 if token.startswith("TAB<") or token == "<eos>" else 0.0
                for token in decoder_tokens
            ]

        while len(labels) < self.data_config.max_decoder_length:
            labels.append(-100)
            loss_values.append(0.0)

        self.examples.append({
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'loss_mask': torch.tensor(loss_values, dtype=torch.float),
        })

    @staticmethod
    def _apply_conditioning_to_encoder_tokens(
        encoder_tokens: Sequence[str],
        metadata: Sequence[NoteMetadata],
        capo: int,
        tuning: Sequence[int],
    ) -> List[str]:
        conditioned_tokens = list(encoder_tokens)
        if not metadata:
            return conditioned_tokens

        tuning_values = [int(value) for value in tuning]
        pointer = 0

        for note in metadata:
            while pointer < len(conditioned_tokens) and not conditioned_tokens[pointer].startswith("NOTE_ON<"):
                pointer += 1

            if pointer >= len(conditioned_tokens):
                break

            base_pitch = tuning_values[note.string - 1] + note.fret + int(capo)
            base_pitch = max(0, min(127, base_pitch))
            conditioned_tokens[pointer] = f"NOTE_ON<{base_pitch}>"

            off_idx = pointer + 1
            while off_idx < len(conditioned_tokens) and not conditioned_tokens[off_idx].startswith("NOTE_OFF<"):
                off_idx += 1

            if off_idx < len(conditioned_tokens):
                conditioned_tokens[off_idx] = f"NOTE_OFF<{base_pitch}>"
                pointer = off_idx + 1
            else:
                pointer += 1

        return conditioned_tokens


def create_song_level_splits(
    tokenizer: MidiTabTokenizerV3,
    manifests: Sequence[Path],
    data_config: DataConfig,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Tuple[SynthTabTokenDataset, SynthTabTokenDataset, SynthTabTokenDataset]:
    """Create train/val/test splits at the song level to prevent data leakage.

    Parameters
    ----------
    tokenizer: MidiTabTokenizerV3
        Tokenizer for encoding
    manifests: Sequence[Path]
        List of manifest files to load
    data_config: DataConfig
        Configuration for data processing
    train_ratio: float
        Fraction of songs for training (default: 0.8)
    val_ratio: float
        Fraction of songs for validation (default: 0.1)
    test_ratio: float
        Fraction of songs for testing (default: 0.1)

    Returns
    -------
    Tuple of (train_dataset, val_dataset, test_dataset)
    """
    import random
    from collections import defaultdict

    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

    # Load all entries and group by song
    all_entries = []
    for manifest in manifests:
        all_entries.extend(load_manifest(manifest))

    # Group entries by track_id (song)
    songs_to_entries = defaultdict(list)
    for entry in all_entries:
        # Extract song name from track_id (remove suffixes like file extensions)
        song_name = entry.track_id.split("__")[0]  # Remove instrument/track suffixes
        songs_to_entries[song_name].append(entry)

    print(f"Found {len(songs_to_entries)} unique songs with {len(all_entries)} total entries")

    # Split songs (not entries) into train/val/test
    song_names = list(songs_to_entries.keys())
    random.shuffle(song_names)

    n_songs = len(song_names)
    train_end = int(n_songs * train_ratio)
    val_end = int(n_songs * (train_ratio + val_ratio))

    train_songs = song_names[:train_end]
    val_songs = song_names[train_end:val_end]
    test_songs = song_names[val_end:]

    print(f"Song splits: {len(train_songs)} train, {len(val_songs)} val, {len(test_songs)} test")

    # Create temporary manifest files for each split
    import tempfile
    import os

    def _create_split_dataset(song_list: List[str], split_name: str) -> SynthTabTokenDataset:
        """Create dataset for a specific split."""
        split_entries = []
        for song in song_list:
            split_entries.extend(songs_to_entries[song])

        print(f"{split_name} split: {len(split_entries)} entries from {len(song_list)} songs")

        # Create temporary manifest
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for entry in split_entries:
                record = {
                    "track_id": entry.track_id,
                    "midi_path": str(entry.midi_path),
                    "tab_path": str(entry.tab_path),
                    "program": entry.program,
                    "split": "all"  # Use "all" since we're handling splits manually
                }
                f.write(json.dumps(record) + '\n')
            temp_manifest = Path(f.name)

        # Create dataset
        dataset = SynthTabTokenDataset(
            tokenizer=tokenizer,
            manifests=[temp_manifest],
            data_config=data_config,
            split="all",
            preload=True
        )

        # Clean up temporary file
        os.unlink(temp_manifest)

        return dataset

    # Create datasets for each split
    train_dataset = _create_split_dataset(train_songs, "Train")
    val_dataset = _create_split_dataset(val_songs, "Validation")
    test_dataset = _create_split_dataset(test_songs, "Test")

    return train_dataset, val_dataset, test_dataset
