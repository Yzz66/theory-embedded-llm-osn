"""
NRC Emotion Lexicon baseline for ontological security narrative classification.

This script evaluates a dictionary-based baseline on a test set. It maps NRC
emotion categories to four ontological security narrative categories and reports
accuracy, macro precision/recall/F1, and Cohen's kappa over repeated runs.

Repeated runs are useful because the baseline randomly resolves ties and cases
with no lexicon matches.
"""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

DEFAULT_CLASSES = ["Denial", "Insult", "Pride", "Shame"]
DEFAULT_N_RUNS = 1
DEFAULT_RANDOM_STATE = 42

NRC_TO_OSN_MAP = {
    "Pride": ["positive", "joy", "trust"],
    "Shame": ["negative", "sadness", "fear"],
    "Insult": ["anger", "disgust"],
    "Denial": ["anticipation", "surprise"],
}

TOKEN_PATTERN = re.compile(r"\b[a-z]+\b", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the NRC Emotion Lexicon baseline on a test set."
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to the test Excel file containing text and label columns.",
    )
    parser.add_argument(
        "--lexicon-path",
        required=True,
        help="Path to NRC-Emotion-Lexicon-Wordlevel-v0.92.txt.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/nrc_emotion_lexicon",
        help="Directory where result files will be saved.",
    )
    parser.add_argument(
        "--text-col",
        default="Text",
        help="Name of the text column in the test file.",
    )
    parser.add_argument(
        "--label-col",
        default="Label",
        help="Name of the gold-label column in the test file.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=DEFAULT_N_RUNS,
        help="Number of repeated evaluation runs. Defaults to 5.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Base random seed used for tie-breaking.",
    )
    parser.add_argument(
        "--class-order",
        nargs="+",
        default=DEFAULT_CLASSES,
        help="Class order used for metrics and confusion matrices.",
    )
    return parser.parse_args()


def validate_input_columns(df: pd.DataFrame, required_columns: Sequence[str]) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Available columns: {list(df.columns)}"
        )


def load_nrc_lexicon(lexicon_path: str) -> Dict[str, List[str]]:
    lexicon = pd.read_csv(
        lexicon_path,
        sep="\t",
        names=["word", "emotion", "assoc"],
    )
    lexicon = lexicon[lexicon["assoc"] == 1]
    return lexicon.groupby("word")["emotion"].apply(list).to_dict()


def build_nrc_to_four_map() -> Dict[str, str]:
    mapping = {}
    for label, emotions in NRC_TO_OSN_MAP.items():
        for emotion in emotions:
            mapping[emotion] = label
    return mapping


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(str(text).lower())


def predict_label(
    text: str,
    rng: np.random.RandomState,
    word_to_emotions: Dict[str, List[str]],
    nrc_to_four: Dict[str, str],
    class_labels: Sequence[str],
) -> Tuple[str, str, str]:
    """Predict one label using dictionary counts.

    Returns:
        predicted label, decision rule, and a serialized count dictionary.
    """
    counts: Counter[str] = Counter()

    for token in tokenize(text):
        if token not in word_to_emotions:
            continue
        for emotion in word_to_emotions[token]:
            if emotion in nrc_to_four:
                counts[nrc_to_four[emotion]] += 1

    if not counts:
        return rng.choice(class_labels), "no_match", ""

    max_count = max(counts.values())
    winners = [label for label, value in counts.items() if value == max_count]

    if len(winners) > 1:
        return rng.choice(winners), "tie", str(dict(counts))

    return winners[0], "unique", str(dict(counts))


def evaluate_once(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    class_labels: Sequence[str],
) -> Dict[str, object]:
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=class_labels,
        average="macro",
        zero_division=0,
    )
    kappa = cohen_kappa_score(y_true, y_pred, labels=class_labels)
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=class_labels,
        digits=4,
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "macro_precision": precision_macro,
        "macro_recall": recall_macro,
        "macro_f1": f1_macro,
        "kappa": kappa,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def format_confusion_matrix(cm: np.ndarray, labels: Sequence[str]) -> str:
    width = 10
    header = " " * width + "".join(f"{label:>{width}}" for label in labels)
    rows = [header]
    for label, values in zip(labels, cm):
        rows.append(f"{label:<{width}}" + "".join(f"{int(x):>{width}}" for x in values))
    return "\n".join(rows)


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.array(values, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return mean, std


def build_summary_dataframe(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "kappa"]:
        mean, std = mean_std(metrics_df[metric])
        rows.append(
            {
                "metric": metric,
                "mean": mean,
                "std": std,
                "mean_std": f"{mean:.6f} ± {std:.6f}",
            }
        )
    return pd.DataFrame(rows)


def save_text_report(
    path: Path,
    settings: Dict[str, object],
    metrics_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    confusion_matrices: Dict[int, np.ndarray],
    reports: Dict[int, str],
    class_labels: Sequence[str],
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("NRC Emotion Lexicon: Test-Only Evaluation\n")
        f.write("=" * 70 + "\n\n")

        f.write("[1] Settings\n")
        f.write("-" * 70 + "\n")
        for key, value in settings.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        f.write("[2] Per-Run Metrics\n")
        f.write("-" * 70 + "\n")
        f.write(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
        f.write("\n\n")

        f.write("[3] Mean ± Std Across Runs\n")
        f.write("-" * 70 + "\n")
        for _, row in summary_df.iterrows():
            f.write(f"{row['metric']}: {row['mean_std']}\n")
        f.write("\n")

        for run_id in sorted(confusion_matrices):
            f.write(f"[4.{run_id}] Confusion Matrix - Run {run_id}\n")
            f.write("-" * 70 + "\n")
            f.write(format_confusion_matrix(confusion_matrices[run_id], class_labels))
            f.write("\n\n")

            f.write(f"Classification Report - Run {run_id}\n")
            f.write("-" * 70 + "\n")
            f.write(reports[run_id])
            f.write("\n\n")


def save_excel_report(
    path: Path,
    summary_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    confusion_matrices: Dict[int, np.ndarray],
    class_labels: Sequence[str],
) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary_Mean_Std", index=False)
        metrics_df.to_excel(writer, sheet_name="Metrics_Per_Run", index=False)
        predictions_df.to_excel(writer, sheet_name="Predictions_All_Runs", index=False)

        for run_id, cm in confusion_matrices.items():
            cm_df = pd.DataFrame(
                cm,
                index=[f"True_{label}" for label in class_labels],
                columns=[f"Pred_{label}" for label in class_labels],
            )
            cm_df.to_excel(writer, sheet_name=f"CM_Run_{run_id}")


def run_evaluation(args: argparse.Namespace) -> None:
    data_path = Path(args.data_path)
    lexicon_path = Path(args.lexicon_path)
    output_dir = Path(args.output_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not lexicon_path.exists():
        raise FileNotFoundError(f"Lexicon file not found: {lexicon_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(data_path)
    validate_input_columns(df, [args.text_col, args.label_col])

    texts = df[args.text_col].astype(str).tolist()
    labels = df[args.label_col].astype(str).tolist()
    class_labels = list(args.class_order)

    print(f"[INFO] Loaded test data: {len(df)} samples")
    print(df[args.label_col].value_counts().reindex(class_labels, fill_value=0))

    word_to_emotions = load_nrc_lexicon(str(lexicon_path))
    nrc_to_four = build_nrc_to_four_map()

    all_metrics = []
    all_predictions = []
    confusion_matrices = {}
    reports = {}

    for run_id in range(1, args.n_runs + 1):
        seed = args.random_state + run_id - 1
        rng = np.random.RandomState(seed)

        pred_labels = []
        pred_rules = []
        score_details = []

        for text in texts:
            pred, rule, detail = predict_label(
                text=text,
                rng=rng,
                word_to_emotions=word_to_emotions,
                nrc_to_four=nrc_to_four,
                class_labels=class_labels,
            )
            pred_labels.append(pred)
            pred_rules.append(rule)
            score_details.append(detail)

        metrics = evaluate_once(labels, pred_labels, class_labels)

        all_metrics.append(
            {
                "run": run_id,
                "random_state": seed,
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "kappa": metrics["kappa"],
            }
        )
        confusion_matrices[run_id] = metrics["confusion_matrix"]
        reports[run_id] = metrics["classification_report"]

        out_df = df.copy()
        out_df["Run"] = run_id
        out_df["Random_State"] = seed
        out_df["Pred_Label"] = pred_labels
        out_df["Pred_Rule"] = pred_rules
        out_df["NRC_Counts"] = score_details
        out_df["Correct"] = (out_df[args.label_col] == out_df["Pred_Label"]).astype(int)
        out_df["Error_Type"] = np.where(
            out_df["Correct"] == 1,
            "",
            out_df[args.label_col] + " -> " + out_df["Pred_Label"],
        )
        all_predictions.append(out_df)

        print(
            f"Run {run_id:02d} | "
            f"Acc={metrics['accuracy']:.4f}, "
            f"Macro-F1={metrics['macro_f1']:.4f}, "
            f"Kappa={metrics['kappa']:.4f}"
        )

    metrics_df = pd.DataFrame(all_metrics)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    summary_df = build_summary_dataframe(metrics_df)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = output_dir / f"nrc_emotion_lexicon_{args.n_runs}runs_results_{timestamp}.txt"
    xlsx_path = output_dir / f"nrc_emotion_lexicon_{args.n_runs}runs_predictions_{timestamp}.xlsx"

    settings = {
        "data_path": data_path,
        "lexicon_path": lexicon_path,
        "n_test": len(df),
        "n_runs": args.n_runs,
        "random_state": args.random_state,
        "classes": class_labels,
        "mapping": NRC_TO_OSN_MAP,
    }

    save_text_report(
        path=txt_path,
        settings=settings,
        metrics_df=metrics_df,
        summary_df=summary_df,
        confusion_matrices=confusion_matrices,
        reports=reports,
        class_labels=class_labels,
    )
    save_excel_report(
        path=xlsx_path,
        summary_df=summary_df,
        metrics_df=metrics_df,
        predictions_df=predictions_df,
        confusion_matrices=confusion_matrices,
        class_labels=class_labels,
    )

    print(f"[OK] Text report saved to: {txt_path}")
    print(f"[OK] Excel report saved to: {xlsx_path}")


def main() -> None:
    args = parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
