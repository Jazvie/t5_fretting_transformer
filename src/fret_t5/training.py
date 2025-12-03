"""Training utilities for the SynthTab Fretting-Transformer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

from torch import nn
import torch.nn.functional as F
from transformers import (
    EarlyStoppingCallback,
    GenerationConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5Config,
    T5ForConditionalGeneration,
)

from .collators import TabSeq2SeqCollator
from .constraints import build_v3_constraint_processor
from .data import SynthTabTokenDataset
from .hf_tokenizer import build_hf_tokenizer
from .metrics import compute_tab_accuracy
from .tokenization import MidiTabTokenizerV3

__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "create_model",
    "TabSeq2SeqTrainer",
    "train_model",
]


@dataclass
class ModelConfig:
    """Configuration for constructing the sequence-to-sequence model."""

    use_pretrained: bool = True
    model_name_or_path: str = "t5-small"
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    tiny_dims: Dict[str, float] = field(
        default_factory=lambda: {
            "d_model": 128,
            "d_ff": 512,
            "num_layers": 4,
            "num_heads": 4,
            "dropout_rate": 0.1,
            "layer_norm_epsilon": 1e-6,
            "relative_attention_num_buckets": 32,
            "feed_forward_proj": "relu",
        }
    )
    d_model: Optional[int] = None
    d_ff: Optional[int] = None
    num_layers: Optional[int] = None
    num_heads: Optional[int] = None
    dropout_rate: Optional[float] = None
    layer_norm_epsilon: Optional[float] = None
    relative_attention_num_buckets: Optional[int] = None
    feed_forward_proj: Optional[str] = None

    def __post_init__(self) -> None:
        overrides = {
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
            "layer_norm_epsilon": self.layer_norm_epsilon,
            "relative_attention_num_buckets": self.relative_attention_num_buckets,
            "feed_forward_proj": self.feed_forward_proj,
        }
        for key, value in overrides.items():
            if value is not None:
                self.tiny_dims[key] = value


@dataclass
class TrainingConfig:
    """Hyper-parameters for training the SynthTab model."""

    output_dir: str = "checkpoints"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 16
    eval_batch_size: int = 16
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 4000
    num_train_epochs: float = 30
    logging_steps: int = 100
    save_total_limit: int = 2
    label_smoothing_factor: float = 0.1
    gradient_clip: float = 1.0
    bf16: bool = True
    fp16: bool = False
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    generation_max_length: int = 128
    generation_num_beams: int = 4
    predict_with_generate: bool = True
    gradient_checkpointing: bool = True
    lr_scheduler_type: str = "linear"
    group_by_length: bool = True
    use_constrained_generation: bool = True
    eval_with_constraints: bool = True
    early_stopping_patience: int = 3
    pad_to_multiple_of: int = 8
    use_adafactor: bool = True
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "tab_accuracy"
    greater_is_better: bool = True
    seed: int = 42
    report_separate_accuracies: bool = False
    eval_delay: float = 0  # Number of epochs to wait before first evaluation
    def __post_init__(self) -> None:
        # Normalize evaluation_strategy to the valid Hugging Face choices.
        es = self.evaluation_strategy
        if isinstance(es, bool):
            es = "epoch" if es else "no"
        if es is None:
            es = "no"
        es = str(es).lower().strip()
        if es not in {"no", "steps", "epoch"}:
            raise ValueError(f"Invalid evaluation_strategy: {es!r}")
        self.evaluation_strategy = es

        # ``load_best_model_at_end`` requires matching save/eval strategies.
        if self.load_best_model_at_end and self.save_strategy != self.evaluation_strategy:
            self.save_strategy = self.evaluation_strategy


def create_model(
    tokenizer: MidiTabTokenizerV3,
    model_config: ModelConfig,
    hf_tokenizer=None,
) -> T5ForConditionalGeneration:
    """Instantiate a ``T5ForConditionalGeneration`` model."""

    if hf_tokenizer is None:
        hf_tokenizer = build_hf_tokenizer(tokenizer.shared_token_to_id)

    if model_config.use_pretrained:
        model = T5ForConditionalGeneration.from_pretrained(model_config.model_name_or_path)
        model.resize_token_embeddings(len(hf_tokenizer))
    else:
        dims = model_config.tiny_dims
        config = T5Config(
            vocab_size=len(hf_tokenizer),
            d_model=int(dims.get("d_model", 128)),
            d_ff=int(dims.get("d_ff", 512)),
            num_layers=int(dims.get("num_layers", 4)),
            num_heads=int(dims.get("num_heads", 4)),
            dropout_rate=float(dims.get("dropout_rate", 0.1)),
            layer_norm_epsilon=float(dims.get("layer_norm_epsilon", 1e-6)),
            relative_attention_num_buckets=int(dims.get("relative_attention_num_buckets", 32)),
            feed_forward_proj=str(dims.get("feed_forward_proj", "relu")),
            pad_token_id=hf_tokenizer.pad_token_id,
            eos_token_id=hf_tokenizer.eos_token_id,
            decoder_start_token_id=hf_tokenizer.pad_token_id,
        )
        model = T5ForConditionalGeneration(config)

    model.config.pad_token_id = hf_tokenizer.pad_token_id
    model.config.eos_token_id = hf_tokenizer.eos_token_id
    model.config.decoder_start_token_id = hf_tokenizer.pad_token_id

    if model_config.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=model_config.lora_r,
            lora_alpha=model_config.lora_alpha,
            lora_dropout=model_config.lora_dropout,
            target_modules=["q", "k", "v", "o", "wi", "wo"],
        )
        model = get_peft_model(model, peft_config)

    return model


class TabSeq2SeqTrainer(Seq2SeqTrainer):
    """Custom trainer that supports ``loss_mask`` weighting."""

    constraint_processor_factory: Optional[Callable[[], object]] = None
    eval_with_constraints: bool = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # type: ignore[override]
        loss_mask = inputs.pop("loss_mask", None)
        if loss_mask is None:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

        labels = inputs.get("labels")
        if labels is None:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

        outputs = model(**inputs)
        logits = outputs.get("logits") if isinstance(outputs, dict) else outputs.logits
        if logits is None:
            raise ValueError("Model outputs must include logits for loss masking.")

        logits = logits.to(model.device)
        labels = labels.to(model.device)
        loss_mask = loss_mask.to(model.device)

        label_mask = (labels != -100).to(loss_mask.dtype)
        effective_mask = loss_mask * label_mask

        if hasattr(self, "label_smoother") and self.label_smoother is not None:
            epsilon = getattr(self.label_smoother, "epsilon", 0.0)
            log_probs = F.log_softmax(logits, dim=-1)
            gather_indices = labels.masked_fill(labels == -100, 0).unsqueeze(-1)
            nll_loss = -log_probs.gather(dim=-1, index=gather_indices).squeeze(-1)
            nll_loss = nll_loss * label_mask
            smooth_loss = -log_probs.mean(dim=-1)
            per_token_loss = (1 - epsilon) * nll_loss + epsilon * smooth_loss
            per_token_loss = per_token_loss * label_mask
        else:
            vocab_size = logits.size(-1)
            loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
            per_token_loss = loss_fct(logits.view(-1, vocab_size), labels.view(-1))
            per_token_loss = per_token_loss.view_as(labels)

        weighted_loss = per_token_loss * effective_mask
        denom = effective_mask.sum()
        loss = weighted_loss.sum() / denom if denom.item() > 0 else weighted_loss.sum()

        return (loss, outputs) if return_outputs else loss

    def _get_logits_processor(self, *args, **kwargs):  # type: ignore[override]
        processors = super()._get_logits_processor(*args, **kwargs)
        if self.eval_with_constraints and self.constraint_processor_factory is not None:
            processors.append(self.constraint_processor_factory())
        return processors


def train_model(
    tokenizer: MidiTabTokenizerV3,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    train_dataset: SynthTabTokenDataset,
    eval_dataset: Optional[SynthTabTokenDataset] = None,
    resume_from_checkpoint: Optional[str] = None,
    model: Optional[nn.Module] = None,
) -> TabSeq2SeqTrainer:
    """Create and run a :class:`TabSeq2SeqTrainer` instance.

    Parameters
    ----------
    tokenizer : MidiTabTokenizerV3
        The tokenizer to use for encoding/decoding.
    model_config : ModelConfig
        Configuration for model architecture.
    training_config : TrainingConfig
        Training hyperparameters.
    train_dataset : SynthTabTokenDataset
        Training dataset.
    eval_dataset : Optional[SynthTabTokenDataset]
        Validation dataset (optional).
    resume_from_checkpoint : Optional[str]
        Path to HuggingFace checkpoint directory to resume from.
    model : Optional[nn.Module]
        Pre-existing model to use (e.g., for finetuning with loaded weights).
        If None, a new model will be created from model_config.
    """

    hf_tokenizer = build_hf_tokenizer(tokenizer.shared_token_to_id)
    if model is None:
        model = create_model(tokenizer, model_config, hf_tokenizer=hf_tokenizer)

    generation_config = GenerationConfig(
        max_length=training_config.generation_max_length,
        num_beams=training_config.generation_num_beams,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.pad_token_id,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    model.generation_config = generation_config

    if training_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    collator_kwargs: Dict[str, object] = {
        "tokenizer": hf_tokenizer,
        "model": model,
        "label_pad_token_id": -100,
    }
    if training_config.pad_to_multiple_of:
        collator_kwargs["pad_to_multiple_of"] = training_config.pad_to_multiple_of
    data_collator = TabSeq2SeqCollator(**collator_kwargs)

    args = Seq2SeqTrainingArguments(
        output_dir=training_config.output_dir,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        optim="adafactor" if training_config.use_adafactor else "adamw_torch",
        per_device_train_batch_size=training_config.batch_size,
        per_device_eval_batch_size=training_config.eval_batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        warmup_steps=training_config.warmup_steps,
        num_train_epochs=training_config.num_train_epochs,
        logging_steps=training_config.logging_steps,
        save_total_limit=training_config.save_total_limit,
        label_smoothing_factor=training_config.label_smoothing_factor,
        max_grad_norm=training_config.gradient_clip,
        bf16=training_config.bf16,
        fp16=training_config.fp16,
        eval_strategy=training_config.evaluation_strategy,
        eval_delay=training_config.eval_delay,
        save_strategy=training_config.save_strategy,
        generation_max_length=training_config.generation_max_length,
        generation_num_beams=training_config.generation_num_beams,
        predict_with_generate=training_config.predict_with_generate,
        gradient_checkpointing=training_config.gradient_checkpointing,
        lr_scheduler_type=training_config.lr_scheduler_type,
        group_by_length=training_config.group_by_length,
        load_best_model_at_end=training_config.load_best_model_at_end,
        metric_for_best_model=training_config.metric_for_best_model,
        greater_is_better=training_config.greater_is_better,
        seed=training_config.seed,
    )

    def _compute_metrics(eval_pred):
        return compute_tab_accuracy(eval_pred, tokenizer)

    trainer = TabSeq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=_compute_metrics if eval_dataset is not None else None,
        tokenizer=hf_tokenizer,
    )

    if training_config.use_constrained_generation:
        trainer.constraint_processor_factory = lambda: build_v3_constraint_processor(tokenizer)
        trainer.eval_with_constraints = training_config.eval_with_constraints

    if training_config.early_stopping_patience > 0 and eval_dataset is not None:
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=training_config.early_stopping_patience))

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    return trainer
