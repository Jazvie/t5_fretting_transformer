# Fret T5

A transformer-based model for converting MIDI to guitar tablature. Based on the Fretting Transformer paper (Hamberger et al., 2024).

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   uv sync
   ```

## Data Setup

For training, the model expects data in JAMS format with MIDI note annotations.

1. Organize your JAMS files in a directory (e.g., `data/guitarset/`).
2. Ensure files contain `note_midi` annotations or `pitch_contour` (which will be converted).
3. The loader expects standard tuning or will attempt to detect it.

## Training

Use `train_guitarset.py` to train the model. Training from scratch is not advised for guitarset, as the dataset is too small for this architecture.

### Basic Usage

Train from scratch:
```bash
python train_guitarset.py --output-dir checkpoints/baseline --guitarset-dir /path/to/jams
```

### Advanced Options

- **Conditioning**: Enable capo/tuning conditioning (recommended for robustness).
  ```bash
  python train_guitarset.py --enable-conditioning --output-dir checkpoints/conditioned
  ```

- **Finetuning**: Finetune from a pretrained checkpoint.
  ```bash
  python train_guitarset.py \
      --pretrained-checkpoint /path/to/checkpoint.pt \
      --learning-rate 5e-5 \
      --output-dir checkpoints/finetuned
  ```

- **LoRA**: Use Low-Rank Adaptation for efficient finetuning.
  ```bash
  python train_guitarset.py \
      --pretrained-checkpoint /path/to/checkpoint.pt \
      --use-lora \
      --output-dir checkpoints/lora
  ```

## SynthTab Training

To train on the SynthTab dataset:

- **With Conditioning** (Recommended): Trains with capo (0-7) and multiple tunings.
  ```bash
  python retrain_synthtab_corrected.py --output-dir checkpoints/synthtab_cond
  ```

- **Without Conditioning**: Trains with standard tuning and capo 0 only.
  ```bash
  python retrain_synthtab_no_conditioning.py --output-dir checkpoints/synthtab_baseline
  ```

## Inference

To run inference using a trained model:

```python
import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, "src")

from fret_t5 import MidiTabTokenizerV3, create_model, ModelConfig, STANDARD_TUNING

# 1. Load Tokenizer
tokenizer = MidiTabTokenizerV3.load("universal_tokenizer")

# 2. Load Model
# If using LoRA or specific config, adjust ModelConfig accordingly
config = ModelConfig(use_pretrained=False, d_model=128, num_layers=3) # Match your training config
model = create_model(tokenizer, config)

# Load weights
checkpoint = torch.load("checkpoints/best_model.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"], strict=False)
model.eval()

# 3. Prepare Input (Encoder Tokens)
# Input should be a sequence of "NOTE_ON<pitch>", "TIME_SHIFT<ms>", "NOTE_OFF<pitch>"
encoder_tokens = [...] 

# --- For Conditioned Models ---
# Prepend conditioning tokens (e.g., Standard Tuning, Capo 0)
capo = 0
tuning = STANDARD_TUNING
prefix = tokenizer.build_conditioning_prefix(capo, tuning)
encoder_tokens = prefix + encoder_tokens
# ------------------------------

input_ids = tokenizer.encode_encoder_tokens_shared(encoder_tokens)
inputs = torch.tensor([input_ids])

# 4. Generate
outputs = model.generate(inputs, max_length=512)
decoded = tokenizer.decode(outputs[0])
print(decoded)
```
