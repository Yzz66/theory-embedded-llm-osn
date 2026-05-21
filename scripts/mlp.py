"""
MLP baseline for ontological security narrative classification.

This script trains a TF-IDF + MLPClassifier model on a fixed training set and
evaluates it on a fixed test set. Because MLP uses random initialization, the
default number of runs is 5.

Expected input columns:
- Text
- Label
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


DEFAULT_TEXT_COL = "Text"
DEFAULT_LABEL_COL = "Label"
DEFAULT_N_RUNS = 5
DEFAULT_RANDOM_STATE = 42
DEFAULT_LABEL_ORDER = ["Denial", "Insult", "Pride", "Shame"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TF-IDF + MLP train-test evaluation."
    )
    parser.add_argument(
        "--train-path",
        required=True,
        help="Path to the training Excel file.",
    )
    parser.add_argument(
        "--test-path",
        required=True,
        help="Path to the test Excel file.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/mlp",
        help="Directory for result files.",
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
        "--n-runs",
        type=int,
        default=DEFAULT_N_RUNS,
        help="Number of repeated train-test runs. Default is 5 for MLP.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Base random seed. Each run uses random_state + run_id - 1.",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=1,
        help="Minimum document frequency for TF-IDF.",
    )
    parser.add_argument(
        "--max-df",
        type=float,
        default=0.95,
        help="Maximum document frequency for TF-IDF.",
    )
    parser.add_argument(
        "--hidden-layer-sizes",
        default="100,200",
        help="Comma-separated hidden layer sizes, e.g., '100,200'.",
    )
    parser.add_argument(
        "--activation",
        default="relu",
        help="Activation function for MLPClassifier.",
    )
    parser.add_argument(
        "--solver",
        default="adam",
        help="Solver for MLPClassifier.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1e-4,
        help="L2 regularization term for MLPClassifier.",
    )
    parser.add_argument(
        "--batch-size",
        default="auto",
        help="Batch size for MLPClassifier.",
    )
    parser.add_argument(
        "--learning-rate",
        default="adaptive",
        help="Learning rate schedule for MLPClassifier.",
    )
    parser.add_argument(
        "--learning-rate-init",
        type=float,
        default=0.001,
        help="Initial learning rate for MLPClassifier.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=20,
        help="Maximum number of MLP training iterations.",
    )
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        help="Enable early stopping. Default is disabled to match the original setup.",
    )
    return parser.parse_args()


def parse_hidden_layer_sizes(value: str) -> Tuple[int, ...]:
    try:
        sizes = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    except ValueError as exc:
        raise ValueError(
            "--hidden-layer-sizes must be a comma-separated list of integers, "
            "for example: 100,200"
        ) from exc

    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("--hidden-layer-sizes must contain positive integers.")

    return sizes


def load_dataset(path: str | os.PathLike, text_col: str, label_col: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    required_cols = {text_col, label_col}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(
            f"Missing required column(s): {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )
    return df.dropna(subset=[text_col, label_col]).copy()


def resolve_label_order(y_train: List[str], y_test: List[str]) -> List[str]:
    observed = set(y_train) | set(y_test)
    labels = [label for label in DEFAULT_LABEL_ORDER if label in observed]
    labels.extend(sorted(observed.difference(labels)))
    return labels


def build_pipeline(args: argparse.Namespace, seed: int, hidden_layer_sizes: Tuple[int, ...]) -> Pipeline:
    """Build a TF-IDF + MLPClassifier pipeline."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(2, 3),
                    min_df=args.min_df,
                    max_df=args.max_df,
                ),
            ),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation=args.activation,
                    solver=args.solver,
                    alpha=args.alpha,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    learning_rate_init=args.learning_rate_init,
                    max_iter=args.max_iter,
                    early_stopping=args.early_stopping,
                    random_state=seed,
                ),
            ),
        ]
    )


def compute_metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> dict:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    kappa = cohen_kappa_score(y_true, y_pred, labels=labels)

    return {
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": macro_f1,
        "kappa": kappa,
    }


def format_confusion_matrix(cm: np.ndarray, labels: Iterable[str]) -> str:
    labels = list(labels)
    col_width = max(10, max(len(str(label)) for label in labels) + 2)
    header = " " * col_width + "".join(f"{label:>{col_width}}" for label in labels)
    rows = [header]

    for label, row in zip(labels, cm):
        rows.append(f"{label:<{col_width}}" + "".join(f"{int(v):>{col_width}}" for v in row))

    return "\n".join(rows)


def mean_std(values: Iterable[float]) -> Tuple[float, float]:
    values = np.array(list(values), dtype=float)
    if len(values) <= 1:
        return float(np.mean(values)), 0.0
    return float(np.mean(values)), float(np.std(values, ddof=1))


def write_text_report(
    path: Path,
    args: argparse.Namespace,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    labels: List[str],
    y_train: List[str],
    y_test: List[str],
    metrics_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    confusion_matrix_sum: np.ndarray,
    reports: dict[int, str],
    hidden_layer_sizes: Tuple[int, ...],
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("MLP Baseline: Train-Test Evaluation\n")
        f.write("=" * 70 + "\n\n")

        f.write("[1] Dataset Information\n")
        f.write("-" * 70 + "\n")
        f.write(f"Train path: {args.train_path}\n")
        f.write(f"Test path: {args.test_path}\n")
        f.write(f"Train size: {len(train_df)}\n")
        f.write(f"Test size: {len(test_df)}\n")
        f.write(f"Labels: {', '.join(labels)}\n")
        f.write(f"Train label distribution: {dict(Counter(y_train))}\n")
        f.write(f"Test label distribution: {dict(Counter(y_test))}\n\n")

        f.write("[2] Model Settings\n")
        f.write("-" * 70 + "\n")
        f.write("Vectorizer: TfidfVectorizer\n")
        f.write("Classifier: MLPClassifier\n")
        f.write("ngram_range: (2, 3)\n")
        f.write(f"min_df: {args.min_df}\n")
        f.write(f"max_df: {args.max_df}\n")
        f.write(f"hidden_layer_sizes: {hidden_layer_sizes}\n")
        f.write(f"activation: {args.activation}\n")
        f.write(f"solver: {args.solver}\n")
        f.write(f"alpha: {args.alpha}\n")
        f.write(f"batch_size: {args.batch_size}\n")
        f.write(f"learning_rate: {args.learning_rate}\n")
        f.write(f"learning_rate_init: {args.learning_rate_init}\n")
        f.write(f"max_iter: {args.max_iter}\n")
        f.write(f"early_stopping: {args.early_stopping}\n")
        f.write(f"N_RUNS: {args.n_runs}\n")
        f.write(f"Base RANDOM_STATE: {args.random_state}\n\n")

        f.write("[3] Per-Run Metrics\n")
        f.write("-" * 70 + "\n")
        f.write(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
        f.write("\n\n")

        f.write(f"[4] Summary Across {args.n_runs} Run(s): Mean ± Std\n")
        f.write("-" * 70 + "\n")
        for _, row in summary_df.iterrows():
            f.write(f"{row['metric']}: {row['mean']:.6f} ± {row['std']:.6f}\n")
        f.write("\n")

        f.write("[5] Confusion Matrix\n")
        f.write("-" * 70 + "\n")
        f.write("Rows = Ground Truth, Columns = Prediction\n")
        if args.n_runs == 1:
            f.write("Matrix = Run 1\n\n")
        else:
            f.write("Matrix = Sum over all runs\n\n")
        f.write(format_confusion_matrix(confusion_matrix_sum, labels))
        f.write("\n\n")

        f.write("[6] Classification Report by Run\n")
        f.write("-" * 70 + "\n")
        for run_id, report in reports.items():
            f.write(f"\nRun {run_id}\n")
            f.write("~" * 40 + "\n")
            f.write(report)
            f.write("\n")


def main() -> None:
    args = parse_args()

    if args.n_runs < 1:
        raise ValueError("--n-runs must be at least 1.")
    if args.max_iter < 1:
        raise ValueError("--max-iter must be at least 1.")

    hidden_layer_sizes = parse_hidden_layer_sizes(args.hidden_layer_sizes)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_dataset(args.train_path, args.text_col, args.label_col)
    test_df = load_dataset(args.test_path, args.text_col, args.label_col)

    x_train = train_df[args.text_col].astype(str).tolist()
    y_train = train_df[args.label_col].astype(str).str.strip().tolist()
    x_test = test_df[args.text_col].astype(str).tolist()
    y_test = test_df[args.label_col].astype(str).str.strip().tolist()

    labels = resolve_label_order(y_train, y_test)

    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    y_train_encoded = label_encoder.transform(y_train)

    print(f"Train path: {args.train_path}")
    print(f"Test path: {args.test_path}")
    print(f"Detected labels: {labels}")
    print(f"Train label distribution: {Counter(y_train)}")
    print(f"Test label distribution: {Counter(y_test)}")

    metrics_records = []
    confusion_matrices = []
    reports = {}
    pred_df = test_df.copy()

    for run_id in range(1, args.n_runs + 1):
        seed = args.random_state + run_id - 1
        print(f"\n===== Run {run_id}/{args.n_runs} | random_state={seed} =====")

        pipeline = build_pipeline(args, seed, hidden_layer_sizes)
        pipeline.fit(x_train, y_train_encoded)

        y_pred_encoded = pipeline.predict(x_test)
        y_pred = label_encoder.inverse_transform(y_pred_encoded).tolist()

        metrics = compute_metrics(y_test, y_pred, labels)
        metrics["run"] = run_id
        metrics["random_state"] = seed
        metrics_records.append(metrics)

        cm = confusion_matrix(y_test, y_pred, labels=labels)
        confusion_matrices.append(cm)

        report = classification_report(
            y_test,
            y_pred,
            labels=labels,
            digits=4,
            zero_division=0,
        )
        reports[run_id] = report

        pred_df[f"Pred_Label_Run{run_id}"] = y_pred
        pred_df[f"Correct_Run{run_id}"] = (
            pred_df[args.label_col].astype(str).str.strip()
            == pred_df[f"Pred_Label_Run{run_id}"].astype(str).str.strip()
        ).astype(int)
        pred_df[f"Error_Type_Run{run_id}"] = np.where(
            pred_df[f"Correct_Run{run_id}"] == 1,
            "",
            pred_df[args.label_col].astype(str).str.strip()
            + " -> "
            + pred_df[f"Pred_Label_Run{run_id}"].astype(str).str.strip(),
        )

        print(
            f"Accuracy={metrics['accuracy']:.4f}, "
            f"Macro-F1={metrics['macro_f1']:.4f}, "
            f"Kappa={metrics['kappa']:.4f}"
        )

    metrics_df = pd.DataFrame(metrics_records)
    metric_cols = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "kappa"]

    summary_df = pd.DataFrame(
        [
            {
                "metric": metric,
                "mean": mean_std(metrics_df[metric])[0],
                "std": mean_std(metrics_df[metric])[1],
            }
            for metric in metric_cols
        ]
    )

    confusion_matrix_sum = np.sum(confusion_matrices, axis=0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    txt_path = output_dir / f"mlp_train_test_results_{timestamp}.txt"
    write_text_report(
        txt_path,
        args,
        train_df,
        test_df,
        labels,
        y_train,
        y_test,
        metrics_df[["run", "random_state"] + metric_cols],
        summary_df,
        confusion_matrix_sum,
        reports,
        hidden_layer_sizes,
    )
    print(f"[OK] Text report saved to: {txt_path}")

    pred_xlsx_path = output_dir / f"mlp_train_test_predictions_{timestamp}.xlsx"
    pred_df.to_excel(pred_xlsx_path, index=False)

    metrics_xlsx_path = output_dir / f"mlp_train_test_metrics_{timestamp}.xlsx"
    with pd.ExcelWriter(metrics_xlsx_path, engine="openpyxl") as writer:
        metrics_df[["run", "random_state"] + metric_cols].to_excel(
            writer, sheet_name="per_run_metrics", index=False
        )
        summary_df.to_excel(writer, sheet_name="summary_mean_std", index=False)
        pd.DataFrame(confusion_matrix_sum, index=labels, columns=labels).to_excel(
            writer, sheet_name="confusion_matrix_sum"
        )

    print(f"[OK] Prediction Excel saved to: {pred_xlsx_path}")
    print(f"[OK] Metrics Excel saved to: {metrics_xlsx_path}")


if __name__ == "__main__":
    main()
