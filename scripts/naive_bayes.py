"""
Multinomial Naive Bayes baseline for ontological security narrative classification.

This script trains a TF-IDF + Multinomial Naive Bayes classifier on a fixed
training set and evaluates it on a fixed test set. Because this baseline is
deterministic under the current configuration, the default number of runs is 1.

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
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


DEFAULT_TEXT_COL = "Text"
DEFAULT_LABEL_COL = "Label"
DEFAULT_N_RUNS = 1
DEFAULT_RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TF-IDF + Multinomial Naive Bayes train-test evaluation."
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
        default="outputs/naive_bayes",
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
        help="Number of repeated train-test runs. Default is 1 for deterministic baselines.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Base random seed recorded in the output files.",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=3,
        help="Minimum document frequency for TF-IDF.",
    )
    parser.add_argument(
        "--max-df",
        type=float,
        default=0.95,
        help="Maximum document frequency for TF-IDF.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Laplace smoothing parameter for MultinomialNB.",
    )
    return parser.parse_args()


def load_dataset(path: str | os.PathLike, text_col: str, label_col: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    required_cols = {text_col, label_col}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(
            f"Missing required column(s): {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )
    return df


def build_pipeline(min_df: int, max_df: float, alpha: float) -> Pipeline:
    """Build a TF-IDF + Multinomial Naive Bayes pipeline."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=min_df,
                    max_df=max_df,
                    sublinear_tf=True,
                ),
            ),
            ("clf", MultinomialNB(alpha=alpha)),
        ]
    )


def evaluate_once(y_true: List[str], y_pred: List[str], labels: List[str]) -> dict:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    kappa = cohen_kappa_score(y_true, y_pred, labels=labels)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        digits=4,
        zero_division=0,
    )

    return {
        "Accuracy": accuracy,
        "Macro_Precision": precision,
        "Macro_Recall": recall,
        "Macro_F1": macro_f1,
        "Kappa": kappa,
        "Confusion_Matrix": cm,
        "Classification_Report": report,
    }


def format_confusion_matrix(cm: np.ndarray, labels: Iterable[str]) -> str:
    labels = list(labels)
    col_width = max(10, max(len(label) for label in labels) + 2)
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
    final_cm: np.ndarray,
    final_report: str,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("Multinomial Naive Bayes Baseline: Train-Test Evaluation\n")
        f.write("=" * 70 + "\n\n")

        f.write("[1] Dataset Information\n")
        f.write("-" * 70 + "\n")
        f.write(f"Train path: {args.train_path}\n")
        f.write(f"Test path: {args.test_path}\n")
        f.write(f"Train size: {len(train_df)}\n")
        f.write(f"Test size: {len(test_df)}\n")
        f.write(f"Labels: {', '.join(labels)}\n")
        f.write(f"Train distribution: {dict(Counter(y_train))}\n")
        f.write(f"Test distribution: {dict(Counter(y_test))}\n\n")

        f.write("[2] Model Settings\n")
        f.write("-" * 70 + "\n")
        f.write("Vectorizer: TfidfVectorizer\n")
        f.write("Classifier: MultinomialNB\n")
        f.write("ngram_range: (1, 2)\n")
        f.write(f"min_df: {args.min_df}\n")
        f.write(f"max_df: {args.max_df}\n")
        f.write("sublinear_tf: True\n")
        f.write(f"alpha: {args.alpha}\n")
        f.write(f"N_RUNS: {args.n_runs}\n")
        f.write(f"RANDOM_STATE: {args.random_state}\n\n")

        f.write("[3] Per-Run Metrics\n")
        f.write("-" * 70 + "\n")
        f.write(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
        f.write("\n\n")

        f.write("[4] Summary: Mean ± Std\n")
        f.write("-" * 70 + "\n")
        for _, row in summary_df.iterrows():
            f.write(f"{row['Metric']}: {row['Mean']:.6f} ± {row['Std']:.6f}\n")
        f.write("\n")

        f.write("[5] Confusion Matrix from Final Run\n")
        f.write("-" * 70 + "\n")
        f.write("Rows = Ground Truth, Columns = Prediction\n")
        f.write("Labels: " + ", ".join(labels) + "\n\n")
        f.write(format_confusion_matrix(final_cm, labels))
        f.write("\n\n")

        f.write("[6] Classification Report from Final Run\n")
        f.write("-" * 70 + "\n")
        f.write(final_report)


def main() -> None:
    args = parse_args()

    if args.n_runs < 1:
        raise ValueError("--n-runs must be at least 1.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_dataset(args.train_path, args.text_col, args.label_col)
    test_df = load_dataset(args.test_path, args.text_col, args.label_col)

    x_train = train_df[args.text_col].astype(str).tolist()
    y_train = train_df[args.label_col].astype(str).str.strip().tolist()
    x_test = test_df[args.text_col].astype(str).tolist()
    y_test = test_df[args.label_col].astype(str).str.strip().tolist()

    labels = sorted(set(y_train) | set(y_test))

    print(f"Detected labels: {labels}")
    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")
    print(f"Train distribution: {Counter(y_train)}")
    print(f"Test distribution: {Counter(y_test)}")

    run_metrics = []
    run_predictions = []

    for run_id in range(1, args.n_runs + 1):
        print(f"\n===== Run {run_id}/{args.n_runs} =====")

        pipeline = build_pipeline(
            min_df=args.min_df,
            max_df=args.max_df,
            alpha=args.alpha,
        )
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test).tolist()

        metrics = evaluate_once(y_test, y_pred, labels)
        run_metrics.append(
            {
                "Run": run_id,
                "Random_State": args.random_state + run_id - 1,
                "Accuracy": metrics["Accuracy"],
                "Macro_Precision": metrics["Macro_Precision"],
                "Macro_Recall": metrics["Macro_Recall"],
                "Macro_F1": metrics["Macro_F1"],
                "Kappa": metrics["Kappa"],
            }
        )
        run_predictions.append(y_pred)

        print(
            f"Run {run_id} | "
            f"Accuracy={metrics['Accuracy']:.4f}, "
            f"Macro-F1={metrics['Macro_F1']:.4f}, "
            f"Kappa={metrics['Kappa']:.4f}"
        )

    final_pred = run_predictions[-1]
    final_metrics = evaluate_once(y_test, final_pred, labels)
    final_cm = final_metrics["Confusion_Matrix"]
    final_report = final_metrics["Classification_Report"]

    metrics_df = pd.DataFrame(run_metrics)

    summary_rows = []
    for metric in ["Accuracy", "Macro_Precision", "Macro_Recall", "Macro_F1", "Kappa"]:
        mean, std = mean_std(metrics_df[metric])
        summary_rows.append({"Metric": metric, "Mean": mean, "Std": std})
    summary_df = pd.DataFrame(summary_rows)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    txt_path = output_dir / f"naive_bayes_train_test_results_{timestamp}.txt"
    write_text_report(
        txt_path,
        args,
        train_df,
        test_df,
        labels,
        y_train,
        y_test,
        metrics_df,
        summary_df,
        final_cm,
        final_report,
    )
    print(f"[OK] Text report saved to: {txt_path}")

    pred_df = test_df.copy()
    pred_df["Pred_Label"] = final_pred
    pred_df["Correct"] = (
        pred_df[args.label_col].astype(str).str.strip()
        == pred_df["Pred_Label"].astype(str).str.strip()
    ).astype(int)
    pred_df["Error_Type"] = np.where(
        pred_df["Correct"] == 1,
        "",
        pred_df[args.label_col].astype(str).str.strip()
        + " -> "
        + pred_df["Pred_Label"].astype(str).str.strip(),
    )

    xlsx_path = output_dir / f"naive_bayes_train_test_predictions_{timestamp}.xlsx"
    pred_df.to_excel(xlsx_path, index=False)
    print(f"[OK] Prediction Excel saved to: {xlsx_path}")

    metrics_path = output_dir / f"naive_bayes_train_test_metrics_{timestamp}.xlsx"
    with pd.ExcelWriter(metrics_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        metrics_df.to_excel(writer, sheet_name="run_metrics", index=False)
        pd.DataFrame(
            final_cm,
            index=[f"True_{label}" for label in labels],
            columns=[f"Pred_{label}" for label in labels],
        ).to_excel(writer, sheet_name="confusion_matrix_final")

    print(f"[OK] Metrics Excel saved to: {metrics_path}")


if __name__ == "__main__":
    main()
