# Fret-T5

A transformer-based model for converting MIDI to guitar tablature with preserved timing. Based on the Fretting-Transformer paper (Hamberger et al., 2024).

## Installation

```bash
# Clone the repository
git clone https://github.com/Jazvie/t5_fretting_transformer.git
cd t5_fretting_transformer

# Install dependencies (choose one)
uv sync              # using uv (recommended)
pip install -e .     # using pip
```

## Training

### Supported Datasets

| Dataset | Description | Source |
|---------|-------------|--------|
| **SynthTab** | Synthetic guitar tablature with conditioning support | Pre-processed manifests |
| **GuitarSet** | Real guitar recordings with professional annotations | [GuitarSet](https://guitarset.weebly.com/) |
| **DadaGP** | Large-scale Guitar Pro tablature collection | [DadaGP](https://github.com/dada-bots/dadaGP) |
| **Custom** | Your own JAMS-format data | User-provided manifests |

### Training Commands

```bash
# SynthTab with conditioning (capo 0-7, multiple tunings)
python train.py --dataset synthtab --output-dir checkpoints/synthtab

# SynthTab without conditioning (standard tuning, capo 0 only)
python train.py --dataset synthtab --no-conditioning --output-dir checkpoints/synthtab_baseline

# GuitarSet training (requires --guitarset-dir)
python train.py --dataset guitarset \
    --guitarset-dir /path/to/GuitarSet/annotation \
    --output-dir checkpoints/guitarset

# GuitarSet finetuning from pretrained checkpoint
python train.py --dataset guitarset \
    --guitarset-dir /path/to/GuitarSet/annotation \
    --pretrained-checkpoint checkpoints/synthtab/best_model.pt \
    --output-dir checkpoints/guitarset_finetuned

# GuitarSet with LoRA adapters
python train.py --dataset guitarset \
    --guitarset-dir /path/to/GuitarSet/annotation \
    --pretrained-checkpoint checkpoints/synthtab/best_model.pt \
    --use-lora \
    --output-dir checkpoints/guitarset_lora

# DadaGP training
python train.py --dataset dadagp --output-dir checkpoints/dadagp

# Custom dataset with your own manifests
python train.py --dataset custom \
    --train-manifest data/my_train.jsonl \
    --val-manifest data/my_val.jsonl \
    --output-dir checkpoints/custom
```

### Key Training Flags

- `--conditioning/--no-conditioning`: Toggle capo and tuning augmentation
- `--guitarset-dir`: Path to GuitarSet annotation directory (required for guitarset dataset)
- `--pretrained-checkpoint`: Initialize from a pretrained model
- `--use-lora`: Use LoRA adapters for efficient finetuning
- `--guitarset-inverted-strings`: Use inverted string numbering (String 1 = Low E)
- `--use-t5-pretrained --model-name t5-small`: Start from HuggingFace T5 weights

## Inference

### Quick Start

```python
from fret_t5 import FretT5Inference

# Load model
inference = FretT5Inference(
    checkpoint_path="checkpoints/best_model.pt",
    tokenizer_path="universal_tokenizer"
)

# MIDI notes from your audio-to-MIDI system
midi_notes = [
    {'pitch': 60, 'start': 0.0, 'duration': 0.523},
    {'pitch': 64, 'start': 0.555, 'duration': 0.312},
    {'pitch': 67, 'start': 0.555, 'duration': 0.312},  # chord with above
]

# Get tablature with original timing preserved
tab_events = inference.predict_with_timing(midi_notes)

for event in tab_events:
    print(f"String {event.string}, Fret {event.fret}")
    print(f"  onset: {event.onset_sec:.3f}s, duration: {event.duration_sec:.3f}s")
```

### Output Format

Each `TabEvent` contains:
- `string`: Guitar string (1-6, where 1 is high E)
- `fret`: Fret number (0-24)
- `onset_sec`: Original onset time from MIDI (continuous, not quantized)
- `duration_sec`: Original duration from MIDI (continuous, not quantized)
- `midi_pitch`: MIDI pitch produced by this position

For JSON-serializable output:
```python
tab_events = inference.predict_with_timing(midi_notes, return_dict=True)
# Returns list of dicts with same fields
```

### Chord Handling

Notes within 10ms of each other are detected as chords. All chord notes receive the same onset time in the output.

## Evaluation

```bash
# Evaluate model on a dataset
python evaluate.py checkpoints/best_model.pt --dataset synthtab --split val --num-samples 50

# Evaluate on GuitarSet (requires --guitarset-dir)
python evaluate.py checkpoints/best_model.pt \
    --dataset guitarset \
    --guitarset-dir /path/to/GuitarSet/annotation \
    --split val

# Post-processing evaluation
python postprocess_best_model.py checkpoints/best_model.pt \
    --dataset synthtab --split val --num_pieces 50
```

## Postprocessing

The postprocessing algorithm (Section 3.5 of the paper) corrects pitch errors in model output:

1. **Pitch Correction**: If predicted pitch differs from input by <=5 semitones, finds alternative fingering that matches input pitch
2. **Fingering Selection**: Uses `fret_stretch` metric to choose fingering closest to model's prediction
3. **Playability Constraint**: Ensures chord shapes have a fret span <= 5 (configurable)
4. **Timing Reconstruction**: Restores original continuous timing from `TimingContext`

```python
from fret_t5.postprocess import postprocess_with_timing

tab_events = postprocess_with_timing(
    encoder_tokens=tokens,
    decoder_tokens=model_output,
    timing_context=timing_ctx,
    pitch_window=5,      # correct pitches within +/-5 semitones
    alignment_window=5,  # alignment window for sequence matching
)
```

## API Reference

### FretT5Inference

```python
class FretT5Inference:
    def __init__(self, checkpoint_path: str, tokenizer_path: str = "universal_tokenizer")
    
    def predict_with_timing(
        self,
        midi_notes: List[Dict],      # [{'pitch': int, 'start': float, 'duration': float}, ...]
        capo: int = 0,
        tuning: tuple = STANDARD_TUNING,
        pitch_window: int = 5,
        alignment_window: int = 5,
        forced_tokens: Optional[Dict[int, int]] = None,
        return_dict: bool = False,
        max_fret_span: int = 5,       # Maximum fret span for playable chords
        enforce_playability: bool = True,  # Enforce playable chord shapes
    ) -> List[TabEvent] | List[Dict]
    
    def predict(
        self,
        midi_notes: List[Dict],
        capo: int = 0,
        tuning: tuple = STANDARD_TUNING,
    ) -> List[str]  # Raw decoder tokens
    
    def predict_raw(
        self,
        midi_notes: List[Dict],
        capo: int = 0,
        tuning: tuple = STANDARD_TUNING,
    ) -> Tuple[List[str], List[str], TimingContext]
```

### Tuning Constants

```python
from fret_t5 import STANDARD_TUNING

STANDARD_TUNING = (64, 59, 55, 50, 45, 40)  # E4, B3, G3, D3, A2, E2
```

## How Timing Preservation Works

The model internally uses quantized TIME_SHIFT tokens (100ms steps), but the pipeline preserves original continuous timing:

1. **Tokenization**: `midi_notes_to_encoder_tokens_with_timing()` creates quantized tokens for the model while storing original timing in a `TimingContext`
2. **Inference**: Model generates tablature using quantized tokens
3. **Postprocessing**: `postprocess_with_timing()` reconstructs original timing from `TimingContext`

This means your output tablature has the exact timestamps from your input MIDI, suitable for sheet music or synchronized playback.

## Project Structure

```
t5_fretting_transformer/
├── src/fret_t5/
│   ├── inference.py      # FretT5Inference class
│   ├── postprocess.py    # Timing preservation & pitch correction
│   ├── tokenization.py   # MidiTabTokenizerV3
│   ├── training.py       # Model creation & training utilities
│   └── constraints.py    # Constrained decoding
├── universal_tokenizer/  # Pre-built tokenizer
├── train.py              # Unified training entrypoint
├── train_guitarset.py    # GuitarSet-specific training
├── evaluate.py           # Model evaluation
├── postprocess_best_model.py  # Post-processing evaluation
└── scripts/              # Data preparation utilities
```
