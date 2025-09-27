import argparse
import json
import logging
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed as transformers_set_seed,
)
from transformers.utils import logging as hf_logging

DEFAULT_CSV_PATH = Path(r"D:\sentiment classifier\data\after training dataset.csv")
DEFAULT_OUT_DIR = Path(r"D:\sentiment classifier\models\xlmr_after_v2_weighted_compat")
DEFAULT_LOG_DIR = Path(r"D:\sentiment classifier\logs\xlmr_after_v2_weighted_compat")
LABELS: Sequence[str] = ("negative", "neutral", "positive")
LOG_CHOICES = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")


class SentimentDataset(torch.utils.data.Dataset):
    """Lightweight Dataset wrapping tokenized inputs and label ids."""

    def __init__(self, encodings: dict[str, Sequence[int]], labels: Sequence[int]):
        self.encodings = {k: torch.tensor(v) for k, v in encodings.items()}
        self.labels = torch.tensor(labels)

    def __len__(self) -> int:  # type: ignore[override]
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        item = {key: tensor[idx] for key, tensor in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


class WeightedTrainer(Trainer):
    """Trainer that applies class weighting when computing the loss."""

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(  # type: ignore[override]
        self,
        model,
        inputs,
        return_outputs: bool = False,
        **kwargs,
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        labels = labels.to(logits.device)
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        inputs["labels"] = labels
        return (loss, outputs) if return_outputs else loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune XLM-R sentiment classifier.")
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH, help="Input CSV with text,label columns.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Directory to save fine-tuned model.")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR, help="Directory for Trainer logs.")
    parser.add_argument("--model", default="xlm-roberta-base", help="Base model checkpoint.")
    parser.add_argument("--max-length", type=int, default=256, help="Maximum sequence length for tokenization.")
    parser.add_argument("--test-size", type=float, default=0.15, help="Validation split size (0-1).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--train-batch", type=int, default=16, help="Per-device train batch size.")
    parser.add_argument("--eval-batch", type=int, default=32, help="Per-device eval batch size.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="AdamW learning rate.")
    parser.add_argument("--logging-steps", type=int, default=50, help="Interval (in steps) for train logging.")
    parser.add_argument("--save-steps", type=int, default=500, help="Interval (in steps) for checkpoints.")
    parser.add_argument("--eval-steps", type=int, default=500, help="Interval (in steps) for eval runs.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=LOG_CHOICES,
        help="Python logging level for this script.",
    )
    parser.add_argument(
        "--hf-log-level",
        choices=LOG_CHOICES,
        help="Transformers logging level (defaults to --log-level).",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Disable progress bars emitted by Hugging Face Trainer.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Shortcut: set script logging to WARNING, transformers to ERROR, and disable tqdm.",
    )
    return parser.parse_args()


def resolve_level(level_name: str) -> int:
    return getattr(logging, level_name.upper())


def configure_logging(level_name: str, hf_level_name: str | None) -> None:
    logging.basicConfig(
        level=resolve_level(level_name),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    hf_level = resolve_level(hf_level_name or level_name)
    hf_logging.set_verbosity(hf_level)
    hf_logging.enable_default_handler()
    hf_logging.enable_explicit_format()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    transformers_set_seed(seed)


def prepare_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"text", "label"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing columns in CSV: {sorted(missing)}")
    df = df.copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    unknown = set(df["label"]) - set(LABELS)
    if unknown:
        raise ValueError(f"Unknown labels present: {sorted(unknown)}")
    return df


def compute_class_weights(df: pd.DataFrame) -> torch.Tensor:
    counts = df["label"].value_counts().reindex(LABELS, fill_value=0).astype(float).values
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    weights = inv / inv.sum() * len(LABELS)
    return torch.tensor(weights, dtype=torch.float32)


def split_dataframe(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx_train, idx_val = next(splitter.split(df.index, df["label"]))
    return df.iloc[idx_train].reset_index(drop=True), df.iloc[idx_val].reset_index(drop=True)


def encode_dataframe(
    tokenizer: AutoTokenizer,
    df: pd.DataFrame,
    label2id: dict[str, int],
    max_length: int,
) -> SentimentDataset:
    encodings = tokenizer(
        df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    label_ids = [label2id[label] for label in df["label"]]
    return SentimentDataset(encodings, label_ids)


def compute_metrics(prediction: EvalPrediction) -> dict[str, float]:
    logits = prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    labels = prediction.label_ids
    accuracy = float((preds == labels).mean())
    return {
        "accuracy": accuracy,
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
    }


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()

    script_log_level = "WARNING" if args.quiet else args.log_level
    hf_log_level = "ERROR" if args.quiet else (args.hf_log_level or script_log_level)
    configure_logging(script_log_level, hf_log_level)

    disable_tqdm = args.disable_tqdm or args.quiet

    csv_path = args.csv_path.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    log_dir = args.log_dir.expanduser().resolve()

    logging.info("Loading data from %s", csv_path)
    df = prepare_dataframe(csv_path)

    label2id = {label: idx for idx, label in enumerate(LABELS)}
    id2label = {idx: label for label, idx in label2id.items()}

    class_weights = compute_class_weights(df)

    train_df, val_df = split_dataframe(df, args.test_size, args.seed)

    set_global_seed(args.seed)

    logging.info("Loading tokenizer %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    logging.info("Tokenizing datasets (train=%d, val=%d)", len(train_df), len(val_df))
    train_dataset = encode_dataframe(tokenizer, train_df, label2id, args.max_length)
    val_dataset = encode_dataframe(tokenizer, val_df, label2id, args.max_length)

    logging.info("Loading model %s", args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    ensure_directory(out_dir)
    ensure_directory(log_dir)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=args.train_batch,
        per_device_eval_batch_size=args.eval_batch,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_dir=str(log_dir),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        seed=args.seed,
        report_to=[],
        disable_tqdm=disable_tqdm,
        log_level=script_log_level.lower(),
        log_level_replica="warning",
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    logging.info("Starting training")
    trainer.train()
    logging.info("Evaluating model")
    metrics = trainer.evaluate()
    logging.info("Validation metrics: %s", metrics)

    logging.info("Saving model to %s", out_dir)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    mapping_path = out_dir / "label_mapping.json"
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)
    logging.info("Wrote label mapping to %s", mapping_path)


if __name__ == "__main__":
    main()
