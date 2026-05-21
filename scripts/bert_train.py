"""
BERT training script for ontological security narrative classification.

This script fine-tunes BERT on a fixed training set and saves one model
directory per run. The default number of runs is 5 because fine-tuning involves
random initialization and mini-batch shuffling.

Expected input columns:
- Text
- Label
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)


DEFAULT_TEXT_COL = "Text"
DEFAULT_LABEL_COL = "Label"
DEFAULT_MODEL_NAME = "bert-base-uncased"
DEFAULT_N_RUNS = 5
DEFAULT_RANDOM_STATE = 42
DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_LEN = 128
DEFAULT_LR = 2e-5
DEFAULT_WARMUP_RATIO = 0.0
DEFAULT_LABELS = ["Denial", "Insult", "Pride", "Shame"]


class TextDataset(Dataset):
    """Dataset for BERT fine-tuning."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer: BertTokenizer, max_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune BERT for ontological security narrative classification."
    )
    parser.add_argument(
        "--train-path",
        required=True,
        help="Path to the training Excel file.",
    )
    parser.add_argument(
        "--model-output-dir",
        default="models/bert",
        help="Directory where fine-tuned BERT runs will be saved.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Base Hugging Face model name.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=DEFAULT_N_RUNS,
        help="Number of training runs. Default is 5.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Base random seed. Each run uses random_state + run_id - 1.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Training batch size.",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=DEFAULT_MAX_LEN,
        help="Maximum token length.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LR,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=DEFAULT_WARMUP_RATIO,
        help="Warmup ratio for the linear learning-rate scheduler.",
    )
    parser.add_argument(
        "--text-col",
        default=DEFAULT_TEXT_COL,
        help="Name of the text column.",
    )
    parser.add_argument(
        "--label-col",
        default=DEFAULT_LABEL_COL,
        help="Name of the label column.",
    )
    parser.add_argument(
        "--labels",
        default=",".join(DEFAULT_LABELS),
        help=(
            "Comma-separated label order. Default: "
            + ",".join(DEFAULT_LABELS)
        ),
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_labels(labels_arg: str) -> List[str]:
    labels = [label.strip() for label in labels_arg.split(",") if label.strip()]
    if not labels:
        raise ValueError("--labels must contain at least one label.")
    if len(labels) != len(set(labels)):
        raise ValueError("--labels contains duplicate labels.")
    return labels


def load_training_data(path: str | os.PathLike, text_col: str, label_col: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    required_cols = {text_col, label_col}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(
            f"Missing required column(s): {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )
    return df.dropna(subset=[text_col, label_col]).copy()


def save_label_map(output_dir: Path, labels: List[str]) -> Dict[str, object]:
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    label_info = {
        "labels": labels,
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()},
    }

    with (output_dir / "label_map.json").open("w", encoding="utf-8") as f:
        json.dump(label_info, f, ensure_ascii=False, indent=2)

    return label_info


def main() -> None:
    args = parse_args()

    if args.n_runs < 1:
        raise ValueError("--n-runs must be at least 1.")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.max_len < 1:
        raise ValueError("--max-len must be at least 1.")

    labels_order = parse_labels(args.labels)
    label2id = {label: i for i, label in enumerate(labels_order)}
    id2label = {i: label for label, i in label2id.items()}

    model_output_dir = Path(args.model_output_dir)
    model_output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_df = load_training_data(args.train_path, args.text_col, args.label_col)
    texts = train_df[args.text_col].astype(str).tolist()
    labels_raw = train_df[args.label_col].astype(str).str.strip().tolist()

    unknown_labels = sorted(set(labels_raw).difference(labels_order))
    if unknown_labels:
        raise ValueError(
            f"Training data contains labels not listed in --labels: {unknown_labels}. "
            "Please update the label order or fix the data."
        )

    labels_encoded = [label2id[label] for label in labels_raw]

    print(f"Labels: {labels_order}")
    print(f"Train distribution: {Counter(labels_raw)}")
    print(f"Train size: {len(train_df)}")

    label_info = save_label_map(model_output_dir, labels_order)

    train_logs = []

    for run_id in range(1, args.n_runs + 1):
        run_seed = args.random_state + run_id - 1
        set_seed(run_seed)

        print(f"\n===== Training Run {run_id}/{args.n_runs} | seed={run_seed} =====")

        run_dir = model_output_dir / f"run_{run_id:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        tokenizer = BertTokenizer.from_pretrained(args.model_name)
        train_dataset = TextDataset(texts, labels_encoded, tokenizer, args.max_len)

        generator = torch.Generator()
        generator.manual_seed(run_seed)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
        )

        model = BertForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=len(labels_order),
            id2label=id2label,
            label2id=label2id,
        ).to(device)

        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        total_steps = len(train_loader) * args.epochs
        warmup_steps = int(total_steps * args.warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        model.train()
        for epoch in range(1, args.epochs + 1):
            epoch_losses = []

            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                y = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=y,
                )
                loss = outputs.loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                epoch_losses.append(float(loss.item()))

            avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            print(f"Run {run_id} | Epoch {epoch}/{args.epochs} | Loss={avg_loss:.6f}")

            train_logs.append(
                {
                    "run": run_id,
                    "seed": run_seed,
                    "epoch": epoch,
                    "loss": avg_loss,
                }
            )

        model.save_pretrained(run_dir)
        tokenizer.save_pretrained(run_dir)

        run_config = {
            "run": run_id,
            "seed": run_seed,
            "train_path": str(args.train_path),
            "model_name": args.model_name,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_len": args.max_len,
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "labels": labels_order,
            "train_distribution": dict(Counter(labels_raw)),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        with (run_dir / "training_config.json").open("w", encoding="utf-8") as f:
            json.dump(run_config, f, ensure_ascii=False, indent=2)

        with (run_dir / "label_map.json").open("w", encoding="utf-8") as f:
            json.dump(label_info, f, ensure_ascii=False, indent=2)

        print(f"[OK] Model saved to: {run_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = model_output_dir / f"bert_train_log_{timestamp}.xlsx"
    pd.DataFrame(train_logs).to_excel(log_path, index=False)

    print(f"\n[OK] Training log saved to: {log_path}")
    print("BERT training finished.")


if __name__ == "__main__":
    main()
