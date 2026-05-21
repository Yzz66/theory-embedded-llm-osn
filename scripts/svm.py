"""
Linear SVM baseline for ontological security narrative classification.

This script trains a TF-IDF + LinearSVC classifier on a fixed training set and
evaluates it on a fixed test set. Under the current configuration, LinearSVC is
effectively deterministic, so the default number of runs is 1.

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
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


DEFAULT_TEXT_COL = "Text"
DEFAULT_LABEL_COL = "Label"
DEFAULT_N_RUNS = 1
DEFAULT_RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TF-IDF + Linear SVM train-test evaluation."
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
        default="outputs/svm",
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
        help="Base random seed for LinearSVC.",
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
        "--c",
        type=float,
        default=1.0,
        help="Regularization parameter C for LinearSVC.",
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


def build_pipeline(seed: int, min_df: int, max_df: float, c_value: float) -> Pipeline:
    """Build a TF-IDF + LinearSVC pipeline."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=min_df,
                    max_df=max_df,
                ),
            ),
            (
                "clf",
                LinearSVC(
                    C=c_value,
                    random_state=seed,
                ),
            ),
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
    final_metrics: dict,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("Linear SVM Baseline: Train-Test Evaluation\n")
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
        f.write("Classifier: LinearSVC\n")
        f.write("ngram_range: (1, 2)\n")
        f.write(f"min_df: {args.min_df}\n")
        f.write(f"max_df: {args.max_df}\n")
        f.write(f"C: {args.c}\n")
        f.write(f"N_RUNS: {args.n_runs}\n")
        f.write(f"RANDOM_STATE: {args.random_state}\n\n")

        f.write("[3] Repeated Runs: Mean ± Std\n")
        f.write("-" * 70 + "\n")
        for _, row in summary_df.iterrows():
            f.write(f"{row['Metric']}: {row['Mean']:.6f} ± {row['Std']:.6f}\n")
        f.write("\n")

        f.write("[4] Per-Run Metrics\n")
        f.write("-" * 70 + "\n")
        f.write(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
        f.write("\n\n")

        f.write("[5] Final Run Metrics\n")
        f.write("-" * 70 + "\n")
        f.write(f"Run: {final_metrics['Run']}\n")
        f.write(f"Seed: {final_metrics['Seed']}\n")
        f.write(f"Cohen's Kappa: {final_metrics['Kappa']:.6f}\n")
        f.write(f"Accuracy: {final_metrics['Accuracy']:.6f}\n")
        f.write(f"Macro Precision: {final_metrics['Macro_Precision']:.6f}\n")
        f.write(f"Macro Recall: {final_metrics['Macro_Recall']:.6f}\n")
        f.write(f"Macro F1: {final_metrics['Macro_F1']:.6f}\n\n")

        f.write("[6] Confusion Matrix from Final Run\n")
        f.write("-" * 70 + "\n")
        f.write("Rows = Ground Truth, Columns = Prediction\n")
        f.write("Labels: " + ", ".join(labels) + "\n\n")
        f.write(format_confusion_matrix(final_cm, labels))
        f.write("\n\n")

        f.write("[7] Classification Report from Final Run\n")
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

    all_metrics = []
    all_predictions = []
    all_confusion_matrices = []
    all_reports = []

    for run_id in range(1, args.n_runs + 1):
        seed = args.random_state + run_id - 1
        print(f"\n===== Run {run_id}/{args.n_runs} | seed={seed} =====")

        pipeline = build_pipeline(
            seed=seed,
            min_df=args.min_df,
            max_df=args.max_df,
            c_value=args.c,
        )
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test).tolist()

        metrics = evaluate_once(y_test, y_pred, labels)
        metrics_row = {
            "Run": run_id,
            "Seed": seed,
            "Accuracy": metrics["Accuracy"],
            "Macro_Precision": metrics["Macro_Precision"],
            "Macro_Recall": metrics["Macro_Recall"],
            "Macro_F1": metrics["Macro_F1"],
            "Kappa": metrics["Kappa"],
        }

        all_predictions.append(y_pred)
        all_confusion_matrices.append(metrics["Confusion_Matrix"])
        all_reports.append(metrics["Classification_Report"])
        all_metrics.append(metrics_row)

        print(
            f"Run {run_id:02d} | "
            f"Accuracy={metrics['Accuracy']:.4f}, "
            f"Macro-F1={metrics['Macro_F1']:.4f}, "
            f"Kappa={metrics['Kappa']:.4f}"
        )

    metrics_df = pd.DataFrame(all_metrics)
    metric_cols = ["Accuracy", "Macro_Precision", "Macro_Recall", "Macro_F1", "Kappa"]

    summary_df = pd.DataFrame(
        [
            {
                "Metric": metric,
                "Mean": mean_std(metrics_df[metric])[0],
                "Std": mean_std(metrics_df[metric])[1],
            }
            for metric in metric_cols
        ]
    )

    final_pred = all_predictions[-1]
    final_cm = all_confusion_matrices[-1]
    final_report = all_reports[-1]
    final_metrics = all_metrics[-1]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    txt_path = output_dir / f"svm_train_test_results_{timestamp}.txt"
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
        final_metrics,
    )
    print(f"[OK] Text report saved to: {txt_path}")

    pred_df = test_df.copy()
    for run_id, pred in enumerate(all_predictions, start=1):
        pred_df[f"Pred_Label_Run_{run_id}"] = pred
        pred_df[f"Correct_Run_{run_id}"] = (
            pred_df[args.label_col].astype(str).str.strip()
            == pred_df[f"Pred_Label_Run_{run_id}"].astype(str).str.strip()
        ).astype(int)
        pred_df[f"Error_Type_Run_{run_id}"] = np.where(
            pred_df[f"Correct_Run_{run_id}"] == 1,
            "",
            pred_df[args.label_col].astype(str).str.strip()
            + " -> "
            + pred_df[f"Pred_Label_Run_{run_id}"].astype(str).str.strip(),
        )

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

    xlsx_path = output_dir / f"svm_train_test_predictions_{timestamp}.xlsx"
    pred_df.to_excel(xlsx_path, index=False)
    print(f"[OK] Prediction Excel saved to: {xlsx_path}")

    metrics_path = output_dir / f"svm_train_test_metrics_{timestamp}.xlsx"
    with pd.ExcelWriter(metrics_path, engine="openpyxl") as writer:
        metrics_df.to_excel(writer, sheet_name="per_run_metrics", index=False)
        summary_df.to_excel(writer, sheet_name="summary_mean_std", index=False)
        pd.DataFrame(
            final_cm,
            index=[f"True_{label}" for label in labels],
            columns=[f"Pred_{label}" for label in labels],
        ).to_excel(writer, sheet_name="confusion_matrix_final")

    print(f"[OK] Metrics Excel saved to: {metrics_path}")


if __name__ == "__main__":
    main()
