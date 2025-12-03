"""Evaluation metrics for SynthTab acoustic training."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Union

import numpy as np

from .tokenization import MidiTabTokenizerV3

__all__ = ["compute_tab_accuracy", "tab_to_midi_pitch"]


def tab_to_midi_pitch(string: int, fret: int) -> int:
    """Convert guitar tablature to MIDI pitch using standard tuning.

    Parameters
    ----------
    string : int
        Guitar string number (1-6, where 1 is high E, 6 is low E)
    fret : int
        Fret number (0-24)

    Returns
    -------
    int
        MIDI pitch number
    """
    # Standard tuning MIDI notes for open strings (1=high E, 6=low E)
    open_strings = {
        1: 64,  # High E (E4)
        2: 59,  # B      (B3)
        3: 55,  # G      (G3)
        4: 50,  # D      (D3)
        5: 45,  # A      (A2)
        6: 40   # Low E  (E2)
    }

    if string not in open_strings:
        raise ValueError(f"Invalid string number: {string} (must be 1-6)")
    if not (0 <= fret <= 24):
        raise ValueError(f"Invalid fret number: {fret} (must be 0-24)")

    return open_strings[string] + fret


def _extract_predictions_and_labels(
    pred: Union[Mapping[str, Any], Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Return predictions and labels from Trainer outputs.

    ``transformers`` can pass either a mapping with ``predictions``/``label_ids``
    entries or an ``EvalPrediction``-like object exposing the same attributes.
    This helper normalises both representations and always returns numpy
    arrays, which simplifies downstream processing.
    """

    if hasattr(pred, "predictions") and hasattr(pred, "label_ids"):
        predictions = getattr(pred, "predictions")
        labels = getattr(pred, "label_ids")
    elif isinstance(pred, Mapping):
        predictions = pred["predictions"]
        labels = pred["label_ids"]
    else:
        raise TypeError(
            "pred must be a mapping or expose 'predictions'/'label_ids' attributes"
        )

    if not isinstance(predictions, (np.ndarray, tuple)):
        predictions = np.asarray(predictions)

    labels = np.asarray(labels)
    return predictions, labels


def compute_tab_accuracy(
    pred: Union[Dict[str, np.ndarray], Any], tokenizer: MidiTabTokenizerV3
) -> Dict[str, float]:
    """Compute comprehensive accuracy metrics for v3 encoding.

    Parameters
    ----------
    pred:
        Output provided by ``Seq2SeqTrainer``. Can be either a mapping with
        ``predictions``/``label_ids`` entries or an ``EvalPrediction``-like
        object exposing the same attributes.
    tokenizer:
        Tokeniser used to determine which tokens correspond to TAB and TIME_SHIFT events.

    Returns
    -------
    Dictionary containing:
    - tab_accuracy: Accuracy on TAB tokens only (exact string/fret match)
    - pitch_accuracy: Accuracy on MIDI pitch for TAB tokens (correct note, any fingering)
    - time_shift_accuracy: Accuracy on TIME_SHIFT tokens only
    - overall_accuracy: Accuracy on all non-masked tokens
    - sequence_validity: Fraction of sequences following v3 pattern
    """

    predictions, labels = _extract_predictions_and_labels(pred)
    if isinstance(predictions, tuple):  # Generation can return (logits, ...)
        predictions = predictions[0]

    # Handle both logits and predicted ids
    if predictions.ndim > 2:  # Logits [batch, seq, vocab]
        predicted_ids = predictions.argmax(axis=-1)
    else:  # Already predicted ids [batch, seq] or [batch*seq]
        predicted_ids = predictions

    if predicted_ids.ndim == 1:
        predicted_ids = predicted_ids.reshape(1, -1)
        labels = labels.reshape(1, -1)

    pad_id = tokenizer.shared_token_to_id.get("<pad>", 0)
    eos_id = tokenizer.shared_token_to_id.get("<eos>", 1)
    decoder_start_id = tokenizer.shared_token_to_id.get("<pad>", pad_id)

    valid_mask = labels != -100
    labels_clean = labels.copy()
    labels_clean[~valid_mask] = pad_id

    trimmed_mask = valid_mask.copy()
    for batch_idx in range(trimmed_mask.shape[0]):
        indices = np.where(trimmed_mask[batch_idx])[0]
        for idx in indices:
            if labels_clean[batch_idx, idx] == eos_id:
                trimmed_mask[batch_idx, indices[indices > idx]] = False
                break

    valid_mask = trimmed_mask
    total_valid = valid_mask.sum()

    if total_valid == 0:
        return {
            "tab_accuracy": 0.0,
            "pitch_accuracy": 0.0,
            "time_shift_accuracy": 0.0,
            "overall_accuracy": 0.0,
            "sequence_validity": 0.0
        }

    # Align predictions with labels to account for decoder start tokens (e.g. T5 <pad>)
    predicted_ids = _align_predictions_with_labels(
        predicted_ids,
        labels_clean,
        valid_mask,
        decoder_start_id,
        pad_id,
    )

    # Get token strings for classification
    predicted_tokens = []
    label_tokens = []
    sequence_validity_scores = []

    for batch_idx in range(predicted_ids.shape[0]):
        sequence_tokens: list[str] = []
        indices = np.where(valid_mask[batch_idx])[0]
        for idx in indices:
            pred_token = tokenizer.shared_id_to_token.get(int(predicted_ids[batch_idx, idx]), "<unk>")
            label_token = tokenizer.shared_id_to_token.get(int(labels_clean[batch_idx, idx]), "<unk>")
            predicted_tokens.append(pred_token)
            label_tokens.append(label_token)
            sequence_tokens.append(pred_token)
        sequence_validity_scores.append(_compute_sequence_validity(_trim_tokens(sequence_tokens)))

    # Classify tokens and compute pitch accuracy
    tab_correct = 0
    tab_total = 0
    pitch_correct = 0
    pitch_total = 0
    time_shift_correct = 0
    time_shift_total = 0

    for pred_tok, label_tok in zip(predicted_tokens, label_tokens):
        if label_tok.startswith("TAB<"):
            tab_total += 1
            if pred_tok == label_tok:
                tab_correct += 1

            # Pitch accuracy: check if MIDI pitch matches
            # Parse ground truth TAB token
            label_parts = label_tok[4:-1].split(',')  # Remove "TAB<" and ">"
            if len(label_parts) == 2:
                label_string, label_fret = int(label_parts[0]), int(label_parts[1])
                label_pitch = tab_to_midi_pitch(label_string, label_fret)

            # Parse predicted TAB token
            if pred_tok.startswith("TAB<") and pred_tok.endswith(">"):
                pred_parts = pred_tok[4:-1].split(',')
                if len(pred_parts) == 2:
                    pred_string, pred_fret = int(pred_parts[0]), int(pred_parts[1])
                    pred_pitch = tab_to_midi_pitch(pred_string, pred_fret)

                    pitch_total += 1
                    if pred_pitch == label_pitch:
                        pitch_correct += 1

        elif label_tok.startswith("TIME_SHIFT<"):
            time_shift_total += 1
            if pred_tok == label_tok:
                time_shift_correct += 1

    # Compute accuracies
    tab_accuracy = float(tab_correct / tab_total) if tab_total > 0 else 0.0
    pitch_accuracy = float(pitch_correct / pitch_total) if pitch_total > 0 else 0.0
    time_shift_accuracy = float(time_shift_correct / time_shift_total) if time_shift_total > 0 else 0.0

    # Overall accuracy
    correct = (predicted_ids == labels_clean) & valid_mask
    overall_accuracy = float(correct.sum() / total_valid)

    # Sequence validity (v3 pattern compliance)
    if sequence_validity_scores:
        sequence_validity = float(np.mean(sequence_validity_scores))
    else:
        sequence_validity = 0.0

    return {
        "tab_accuracy": tab_accuracy,
        "pitch_accuracy": pitch_accuracy,
        "time_shift_accuracy": time_shift_accuracy,
        "overall_accuracy": overall_accuracy,
        "sequence_validity": sequence_validity
    }


def _align_predictions_with_labels(
    predicted_ids: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray,
    decoder_start_id: int,
    pad_id: int,
) -> np.ndarray:
    """Shift predictions to account for decoder start tokens preceding outputs.

    Seq2Seq models such as T5 prepend ``decoder_start_token_id`` (``<pad>`` for
    SynthTab) to every generated sequence. When comparing predictions against
    labels this extra token introduces a one-step offset, leading to the
    near-zero accuracies observed during evaluation. The trainer removes the
    start token from ``labels`` internally, so we mirror that here by aligning
    predictions with the first valid label position per sequence.

    Parameters
    ----------
    predicted_ids:
        Array of predicted token ids with shape ``[batch, seq_len]``.
    labels:
        Array of label token ids with shape ``[batch, seq_len]`` where masked
        positions have already been filled with ``pad_id``.
    valid_mask:
        Boolean mask indicating positions that should be considered for metric
        computation (i.e. ``labels != -100`` before padding & trimming).
    decoder_start_id:
        Token id used to start decoder generation (typically ``<pad>``).
    pad_id:
        Token id representing padding in the shared vocabulary.

    Returns
    -------
    np.ndarray
        Adjusted prediction array with decoder start tokens aligned to the
        target sequence.
    """

    if predicted_ids.size == 0:
        return predicted_ids

    aligned = predicted_ids.copy()

    for batch_idx in range(aligned.shape[0]):
        valid_positions = np.where(valid_mask[batch_idx])[0]
        if valid_positions.size == 0:
            continue

        first_valid = valid_positions[0]
        label_first_token = labels[batch_idx, first_valid]

        # Only align if prediction starts with decoder_start_id
        if aligned[batch_idx, first_valid] != decoder_start_id:
            continue

        # Skip if label also starts with decoder_start_id (edge case)
        if label_first_token == decoder_start_id:
            continue

        # Shift the entire suffix left by one position so that the decoder
        # start token is dropped and the remaining tokens line up with the
        # target sequence.
        aligned_row = aligned[batch_idx]
        if first_valid < aligned_row.shape[0] - 1:
            aligned_row[first_valid:-1] = aligned_row[first_valid + 1 :]
        aligned_row[-1] = pad_id

    return aligned


def _trim_tokens(tokens: list[str]) -> list[str]:
    """Trim a token sequence at the first ``<eos>`` token (inclusive)."""

    trimmed: list[str] = []
    for token in tokens:
        trimmed.append(token)
        if token == "<eos>":
            break
    return trimmed


def _compute_sequence_validity(tokens: list[str]) -> float:
    """Check what fraction of predicted tokens follow v3 TAB ↔ TIME_SHIFT pattern."""
    if len(tokens) == 0:
        return 0.0

    valid_transitions = 0
    total_transitions = 0
    expect_tab = True  # v3 starts with TAB

    for token in tokens:
        if token in ["<pad>", "<eos>", "<unk>"]:
            continue

        is_tab = token.startswith("TAB<")
        is_time_shift = token.startswith("TIME_SHIFT<")

        if not (is_tab or is_time_shift):
            continue  # Skip special tokens

        total_transitions += 1

        if expect_tab and is_tab:
            valid_transitions += 1
            expect_tab = False  # Next should be TIME_SHIFT
        elif not expect_tab and is_time_shift:
            valid_transitions += 1
            expect_tab = True   # Next should be TAB
        # else: invalid transition

    return float(valid_transitions / total_transitions) if total_transitions > 0 else 0.0
