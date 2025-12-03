"""Custom data collators for SynthTab Seq2Seq training."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import DataCollatorForSeq2Seq
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from transformers.utils import PaddingStrategy

__all__ = ["TabSeq2SeqCollator"]


class TabSeq2SeqCollator(DataCollatorForSeq2Seq):
    """Extension of :class:`DataCollatorForSeq2Seq` that keeps ``loss_mask`` tensors."""

    def __call__(self, features: List[Dict[str, Any]], return_tensors: Optional[str] = None) -> Dict[str, Any]:  # type: ignore[override]
        loss_masks: List[Optional[torch.Tensor]] = [f.pop("loss_mask", None) for f in features]

        if return_tensors is None:
            return_tensors = self.return_tensors

        label_name = "label" if "label" in features[0] else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0] else None

        if labels is not None and all(label is None for label in labels):
            labels = None

        non_labels_features = [{k: v for k, v in feature.items() if k != label_name} for feature in features]

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            non_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        no_padding = self.padding is False or self.padding == PaddingStrategy.DO_NOT_PAD
        padding_side = self.tokenizer.padding_side

        def _to_numpy_array(label: Any) -> np.ndarray:
            if isinstance(label, np.ndarray):
                array = label
            elif isinstance(label, torch.Tensor):
                array = label.cpu().numpy()
            else:
                array = np.asarray(label)
            return array.astype(np.int64, copy=False)

        if labels is not None:
            if no_padding:
                if isinstance(features[0][label_name], list):
                    batch["labels"] = list(labels)
                else:
                    batch["labels"] = [_to_numpy_array(label) for label in labels]
            else:
                max_padding = self.padding == PaddingStrategy.MAX_LENGTH and self.max_length is not None
                max_label_length = max(len(l) for l in labels) if not max_padding else self.max_length
                if self.pad_to_multiple_of is not None:
                    max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                    )

                if isinstance(features[0][label_name], list):
                    batch["labels"] = [
                        label + [self.label_pad_token_id] * (max_label_length - len(label))
                        if padding_side == "right"
                        else [self.label_pad_token_id] * (max_label_length - len(label)) + label
                        for label in labels
                    ]
                else:
                    batch["labels"] = []
                    for label in labels:
                        label_array = _to_numpy_array(label)
                        pad_width = max_label_length - len(label)
                        pad_values = (
                            np.full(pad_width, self.label_pad_token_id, dtype=np.int64)
                            if pad_width > 0
                            else np.empty(0, dtype=np.int64)
                        )
                        if padding_side == "right":
                            padded = np.concatenate([label_array, pad_values])
                        else:
                            padded = np.concatenate([pad_values, label_array])
                        batch["labels"].append(padded)

        if batch.get("labels", None) is not None:
            if return_tensors == "pt":
                labels_field = batch["labels"]
                if isinstance(labels_field, list) and labels_field and isinstance(labels_field[0], np.ndarray):
                    stacked = np.stack(labels_field, axis=0)
                    batch["labels"] = torch.from_numpy(stacked).to(torch.int64)
                else:
                    batch["labels"] = torch.tensor(labels_field, dtype=torch.int64)
            elif return_tensors == "tf":
                import tensorflow as tf

                batch["labels"] = tf.constant(batch["labels"], dtype=tf.int64)
            else:
                batch["labels"] = np.array(batch["labels"], dtype=np.int64)
        else:
            batch["labels"] = None

        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=batch["labels"])
            batch["decoder_input_ids"] = decoder_input_ids

        if any(mask is not None for mask in loss_masks):
            if batch["labels"] is not None:
                max_len = batch["labels"].shape[1]
            else:
                max_len = max((mask.size(0) for mask in loss_masks if mask is not None), default=0)

            dtype = next((mask.dtype for mask in loss_masks if mask is not None), torch.float)
            padded_masks: List[torch.Tensor] = []

            for mask in loss_masks:
                if mask is None:
                    padded_masks.append(torch.zeros(max_len, dtype=dtype))
                    continue

                mask_tensor = torch.as_tensor(mask, dtype=dtype)
                trimmed = mask_tensor[-max_len:] if mask_tensor.numel() > max_len else mask_tensor
                padded = torch.zeros(max_len, dtype=dtype)
                length = trimmed.numel()

                if padding_side == "right":
                    padded[:length] = trimmed
                else:
                    padded[-length:] = trimmed

                padded_masks.append(padded)

            batch["loss_mask"] = torch.stack(padded_masks, dim=0)

        return batch
