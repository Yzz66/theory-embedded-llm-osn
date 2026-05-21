"""
Baseline prompting with confidence scores.

This public version is designed for reproducible GitHub releases. It supports
three paper-level model presets: Llama-3.1-8B, Llama-3.1-70B, and Qwen3-32B.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
import threading
import time
from pathlib import Path
from typing import Optional

try:
    import krippendorff
except ImportError:
    krippendorff = None
import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)

_LAST_REQUEST_TIME = 0.0
_REQUEST_LOCK = threading.Lock()

EVAL_LABELS = ["Pride", "Shame", "Denial", "Insult", "Unknown"]
EVAL_LABELS_4 = ["Pride", "Shame", "Denial", "Insult"]

DEFAULT_N_RUNS = 5
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0

MODEL_PRESETS = {
    "llama-3.1-8b": {
        "model": "meta/llama-3.1-8b-instruct",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key_env": "NVIDIA_API_KEY",
        "disable_thinking": False,
    },
    "llama-3.1-70b": {
        "model": "meta/llama-3.1-70b-instruct",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key_env": "NVIDIA_API_KEY",
        "disable_thinking": False,
    },
    "qwen3-32b": {
        "model": "qwen3-32b",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "disable_thinking": True,
    },
}


def safe_model_name(model_name: str) -> str:
    """Convert a model name into a safe string for output file names."""
    return (
        model_name.replace("/", "_")
        .replace("\\\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def clean_text(text: object) -> str:
    """Normalize text before inserting it into a prompt."""
    if pd.isna(text):
        return ""
    cleaned = str(text).replace("{", "(").replace("}", ")")
    cleaned = (
        cleaned.replace("\\u200b", "")
        .replace("\\u200e", "")
        .replace("\\u200f", "")
        .strip()
    )
    return cleaned


def normalize_label(label: object) -> str:
    """Map empty or invalid labels to Unknown for evaluation."""
    value = str(label).strip()
    if value in {"None", "Empty", "Error", "nan", "NaN", ""}:
        return "Unknown"
    return value


def parse_rank_values(raw_values: Optional[str]) -> Optional[list[int]]:
    """Parse comma-separated rank values, e.g., '1,2,3'."""
    if not raw_values:
        return None
    return [int(item.strip()) for item in raw_values.split(",") if item.strip()]


def parse_rank_range(raw_range: Optional[str]) -> Optional[tuple[int, int]]:
    """Parse an inclusive rank range, e.g., '1,200'."""
    if not raw_range:
        return None
    parts = [item.strip() for item in raw_range.split(",") if item.strip()]
    if len(parts) != 2:
        raise ValueError("--rank-range must use the format 'low,high', for example '1,200'.")
    low, high = int(parts[0]), int(parts[1])
    if low > high:
        raise ValueError("--rank-range requires low <= high.")
    return low, high


def resolve_model_config(args: argparse.Namespace) -> dict:
    """Resolve model, provider endpoint, API key variable, and thinking mode."""
    if args.model_preset == "custom":
        preset = {
            "model": None,
            "base_url": os.getenv("OPENAI_BASE_URL"),
            "api_key_env": "OPENAI_API_KEY",
            "disable_thinking": False,
        }
    else:
        preset = MODEL_PRESETS[args.model_preset].copy()

    model = args.model or preset["model"] or os.getenv("OPENAI_MODEL")
    if not model:
        raise ValueError("A model name is required. Use --model-preset or --model.")

    base_url = args.base_url or preset["base_url"] or os.getenv("OPENAI_BASE_URL")
    api_key_env = args.api_key_env or preset["api_key_env"]
    disable_thinking = preset["disable_thinking"] if args.disable_thinking is None else args.disable_thinking

    api_key = os.getenv(api_key_env) or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            f"No API key found. Set {api_key_env} or OPENAI_API_KEY before running this script."
        )

    return {
        "model": model,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "api_key": api_key,
        "disable_thinking": disable_thinking,
        "preset_name": args.model_preset,
    }


def build_client(model_config: dict) -> OpenAI:
    """Create an OpenAI-compatible client for the selected provider."""
    if model_config.get("base_url"):
        return OpenAI(api_key=model_config["api_key"], base_url=model_config["base_url"])
    return OpenAI(api_key=model_config["api_key"])


def chat_completion(
    client: OpenAI,
    model_name: str,
    prompt: str,
    temperature: float,
    top_p: float,
    disable_thinking: bool,
):
    """Call an OpenAI-compatible chat completion endpoint."""
    kwargs = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
    }
    if disable_thinking:
        kwargs["extra_body"] = {"enable_thinking": False}
    return client.chat.completions.create(**kwargs)


def parse_label_and_confidence(raw_output: str) -> tuple[str, Optional[float]]:
    """Extract the narrative label and confidence score from model output."""
    label = "Unknown"
    confidence = None

    patterns = [
        r"Form\s*:\s*(Pride|Shame|Denial|Insult|None)",
        r"Primary\s*:\s*(Pride|Shame|Denial|Insult|None)\s*\(\s*(?:confidence\s*:\s*)?(\d+)%\s*\)",
        r"Primary\s*:\s*(Pride|Shame|Denial|Insult|None)",
        r"(?:Form|Primary)?\s*:?\s*<?\s*(Pride|Shame|Insult|Denial|None)\s*>?",
    ]

    for pattern in patterns:
        match = re.search(pattern, raw_output, re.IGNORECASE)
        if match:
            label = match.group(1).capitalize()
            if len(match.groups()) >= 2 and match.group(2):
                confidence = int(match.group(2)) / 100.0
            break

    if label == "Unknown":
        head_match = re.match(r"\s*(Pride|Shame|Denial|Insult)\b", raw_output, re.IGNORECASE)
        if head_match:
            label = head_match.group(1).capitalize()

    if confidence is None:
        confidence_match = re.search(r"confidence\s*:\s*(\d+)\s*%", raw_output, re.IGNORECASE)
        if confidence_match:
            confidence = int(confidence_match.group(1)) / 100.0

    if label == "None":
        label = "Unknown"
    return label, confidence


def validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Ensure that the input data contains the required columns."""
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")


def filter_by_rank(
    df: pd.DataFrame,
    use_rank_filter: bool,
    rank_values: Optional[list[int]],
    rank_range: Optional[tuple[int, int]],
) -> pd.DataFrame:
    """Apply optional rank filtering."""
    if not use_rank_filter:
        return df.copy()

    validate_columns(df, ["Rank"])
    if rank_values is not None:
        return df[df["Rank"].isin(rank_values)].copy()
    if rank_range is not None:
        low, high = rank_range
        return df[(df["Rank"] >= low) & (df["Rank"] <= high)].copy()
    raise ValueError("Rank filtering is enabled, but no rank values or rank range were provided.")


def compute_metrics(y_true: list[object], y_pred: list[object]) -> dict:
    """Compute diagnostic and main evaluation metrics."""
    y_true_n = [normalize_label(value) for value in y_true]
    y_pred_n = [normalize_label(value) for value in y_pred]

    cm5 = confusion_matrix(y_true_n, y_pred_n, labels=EVAL_LABELS)
    kappa5 = cohen_kappa_score(y_true_n, y_pred_n, labels=EVAL_LABELS)
    acc5 = float(np.mean([actual == pred for actual, pred in zip(y_true_n, y_pred_n)]))

    try:
        if krippendorff is None:
            raise ImportError("krippendorff is not installed")
        alpha5 = krippendorff.alpha(
            reliability_data=[y_true_n, y_pred_n],
            level_of_measurement="nominal",
        )
    except Exception:
        alpha5 = float("nan")

    covered_mask = [pred in EVAL_LABELS_4 for pred in y_pred_n]
    y_true_covered = [actual for actual, keep in zip(y_true_n, covered_mask) if keep]
    y_pred_covered = [pred for pred, keep in zip(y_pred_n, covered_mask) if keep]
    coverage = len(y_pred_covered) / max(1, len(y_pred_n))

    if y_pred_covered:
        kappa4 = cohen_kappa_score(y_true_covered, y_pred_covered, labels=EVAL_LABELS_4)
        acc4 = float(np.mean([actual == pred for actual, pred in zip(y_true_covered, y_pred_covered)]))
    else:
        kappa4, acc4 = float("nan"), float("nan")

    return {
        "cm5": cm5,
        "kappa5": kappa5,
        "alpha5": alpha5,
        "acc5": acc5,
        "coverage": coverage,
        "kappa4_cov": kappa4,
        "acc4_cov": acc4,
        "n": len(y_true_n),
        "n_cov": len(y_pred_covered),
    }


def mean_std(metrics: list[dict], key: str) -> tuple[float, float]:
    """Return mean and sample standard deviation for a metric across runs."""
    values = np.array([item[key] for item in metrics], dtype=float)
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values, ddof=1)) if len(values) > 1 else float("nan")
    return mean, std


def write_evaluation_report(
    eval_path: Path,
    method_name: str,
    input_file: Path,
    model_config: dict,
    n_runs: int,
    temperature: float,
    y_true: list[object],
    y_pred_by_run: list[list[object]],
) -> None:
    """Write a multi-run evaluation report."""
    run_metrics = []
    cm5_sum = np.zeros((len(EVAL_LABELS), len(EVAL_LABELS)), dtype=int)
    y_true_all = []
    y_pred_all = []

    with open(eval_path, "w", encoding="utf-8") as file:
        file.write(f"Evaluation Summary ({method_name} – Confidence – Multi-run)\n")
        file.write(f"Input file: {input_file}\n")
        file.write(f"Model preset: {model_config['preset_name']}\n")
        file.write(f"Model: {model_config['model']}\n")
        file.write(f"Base URL: {model_config.get('base_url') or 'default'}\n")
        file.write(f"Temperature: {temperature}\n")
        file.write(f"Rows evaluated: {len(y_true)}\n")
        file.write(f"N_RUNS: {n_runs}\n\n")

        for run_id, y_pred in enumerate(y_pred_by_run, start=1):
            metrics = compute_metrics(y_true, y_pred)
            run_metrics.append(metrics)
            cm5_sum += metrics["cm5"]
            y_true_all.extend(y_true)
            y_pred_all.extend(y_pred)

            file.write("=" * 64 + "\n")
            file.write(f"Run {run_id}\n")
            file.write("=" * 64 + "\n")
            file.write(
                f"kappa5={metrics['kappa5']:.4f}, "
                f"alpha5={metrics['alpha5']:.4f}, "
                f"acc5={metrics['acc5']:.4f}, "
                f"coverage={metrics['coverage']:.4f} ({metrics['n_cov']}/{metrics['n']}), "
                f"kappa4_cov={metrics['kappa4_cov']:.4f}, "
                f"acc4_cov={metrics['acc4_cov']:.4f}\n"
            )

        file.write("\n" + "=" * 64 + "\n")
        file.write("Final Prediction: Mean ± Std over runs\n")
        file.write("=" * 64 + "\n")
        for key in ["kappa5", "alpha5", "acc5", "coverage", "kappa4_cov", "acc4_cov"]:
            mean, std = mean_std(run_metrics, key)
            file.write(f"{key}: {mean:.4f} ± {std:.4f}\n")

        file.write("\nConfusion Matrix (5-class, summed over runs; diagnostic):\n")
        file.write(pd.DataFrame(cm5_sum, index=EVAL_LABELS, columns=EVAL_LABELS).to_string())
        file.write("\n")

        file.write("\n" + "=" * 64 + "\n")
        file.write("Classification Report (5-class, aggregated over runs)\n")
        file.write("=" * 64 + "\n")
        file.write(classification_report(y_true_all, y_pred_all, labels=EVAL_LABELS, zero_division=0))

        covered_mask = [pred in EVAL_LABELS_4 for pred in y_pred_all]
        y_true_covered = [actual for actual, keep in zip(y_true_all, covered_mask) if keep]
        y_pred_covered = [pred for pred, keep in zip(y_pred_all, covered_mask) if keep]
        if y_pred_covered:
            file.write("\n" + "=" * 64 + "\n")
            file.write("Classification Report (4-class, covered, aggregated over runs)\n")
            file.write("=" * 64 + "\n")
            file.write(classification_report(y_true_covered, y_pred_covered, labels=EVAL_LABELS_4, zero_division=0))


def add_common_arguments(parser: argparse.ArgumentParser, default_output_dir: str, default_workers: int, default_interval: float) -> None:
    """Add common command-line arguments."""
    parser.add_argument("--input-file", required=True, type=Path, help="Path to the input Excel file.")
    parser.add_argument("--output-dir", default=Path(default_output_dir), type=Path, help="Directory for outputs.")
    parser.add_argument(
        "--model-preset",
        choices=[*MODEL_PRESETS.keys(), "custom"],
        default="llama-3.1-70b",
        help="Model/provider preset. Use 'custom' with --model and optionally --base-url.",
    )
    parser.add_argument("--model", default=None, help="Override the model name from the preset.")
    parser.add_argument("--base-url", default=None, help="Override the OpenAI-compatible base URL.")
    parser.add_argument("--api-key-env", default=None, help="Environment variable containing the API key.")
    parser.add_argument(
        "--disable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to pass extra_body={'enable_thinking': False}. Default follows the selected preset.",
    )
    parser.add_argument("--n-runs", default=DEFAULT_N_RUNS, type=int, help="Number of repeated runs.")
    parser.add_argument("--temperature", default=DEFAULT_TEMPERATURE, type=float, help="Sampling temperature.")
    parser.add_argument("--top-p", default=DEFAULT_TOP_P, type=float, help="Top-p sampling parameter.")
    parser.add_argument("--max-workers", default=default_workers, type=int, help="Number of parallel workers.")
    parser.add_argument("--request-interval", default=default_interval, type=float, help="Minimum interval between API requests.")
    parser.add_argument("--rank-values", default=None, help="Optional comma-separated rank values, e.g., '1,2,3'.")
    parser.add_argument("--rank-range", default="1,200", help="Optional inclusive rank range, e.g., '1,200'.")
    parser.add_argument("--no-rank-filter", action="store_true", help="Disable rank filtering.")
    parser.add_argument("--retries", default=3, type=int, help="Number of retries for each API request.")


METHOD_NAME = "Baseline prompting"
OUTPUT_PREFIX = "baseline_prompting"
DEFAULT_OUTPUT_DIR = "outputs/baseline_prompting"
DEFAULT_MAX_WORKERS = 1
DEFAULT_REQUEST_INTERVAL = 10.0

PROMPT_TEMPLATE = """
[Role]
Political narrative analyst.

[Task]
Assign the primary ontological security narrative(Pride, Shame, Denial, Insult, or None).

[Theoretical Framing]
1. Conduct the analysis strictly from the narrator’s perspective.
2. Ontological category cues:
- Pride: the narrator positively evaluates their own actions, identity, achievements, or moral standing.
- Shame: the narrator negatively evaluates their own actions, identity, or responsibility, without rejecting that responsibility.
- Denial: the narrator denies their own action, statement, involvement, or responsibility.
- Insult: the narrator uses devaluation of others as the primary narrative purpose.

[Text]
{CORPUS}

[Output instruction]

- Output only one primary narrative.
- Do NOT output any explanation, analysis, reasoning steps, or intermediate thoughts.
- Provide a confidence score from 0 to 100.
- Use the following format:

Primary: <type> (confidence: <score>%)

""".strip()


def classify_text(text: object, client: OpenAI, model_config: dict, temperature: float, top_p: float, request_interval: float, retries: int) -> tuple[str, Optional[float], str]:
    """Classify one text with baseline prompting."""
    cleaned = clean_text(text)
    if cleaned == "":
        return "Empty", None, "Empty Content"
    prompt = PROMPT_TEMPLATE.replace("{CORPUS}", cleaned)

    for attempt in range(1, retries + 1):
        try:
            global _LAST_REQUEST_TIME
            with _REQUEST_LOCK:
                now = time.time()
                wait_time = request_interval - (now - _LAST_REQUEST_TIME)
                if wait_time > 0:
                    time.sleep(wait_time)
                _LAST_REQUEST_TIME = time.time()

            response = chat_completion(
                client=client,
                model_name=model_config["model"],
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                disable_thinking=model_config["disable_thinking"],
            )
            raw = response.choices[0].message.content.strip()
            label, confidence = parse_label_and_confidence(raw)
            return label, confidence, raw
        except Exception as exc:
            print(f"Request failed on attempt {attempt}/{retries}: {exc}")
            time.sleep(2)
    return "Error", None, "API Error"


def process_all(texts: list[object], client: OpenAI, model_config: dict, temperature: float, top_p: float, request_interval: float, max_workers: int, retries: int) -> list[tuple[str, Optional[float], str]]:
    """Classify all texts, optionally in parallel."""
    results: list[Optional[tuple[str, Optional[float], str]]] = [None] * len(texts)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(classify_text, text, client, model_config, temperature, top_p, request_interval, retries): idx
            for idx, text in enumerate(texts)
        }
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            results[futures[future]] = future.result()
            print(f"Completed: {completed}/{len(texts)}")
    return [item for item in results if item is not None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline prompting.")
    add_common_arguments(parser, DEFAULT_OUTPUT_DIR, DEFAULT_MAX_WORKERS, DEFAULT_REQUEST_INTERVAL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_config = resolve_model_config(args)
    client = build_client(model_config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(args.input_file)
    validate_columns(df, ["Text", "Label"])
    df_run = filter_by_rank(
        df,
        use_rank_filter=not args.no_rank_filter,
        rank_values=parse_rank_values(args.rank_values),
        rank_range=parse_rank_range(args.rank_range),
    )

    texts = df_run["Text"].tolist()
    y_true = df_run["Label"].tolist()
    y_pred_by_run: list[list[object]] = []

    for run_id in range(1, args.n_runs + 1):
        print(f"\n{METHOD_NAME} | {model_config['preset_name']} | Run {run_id}/{args.n_runs}")
        preds = process_all(texts, client, model_config, args.temperature, args.top_p, args.request_interval, args.max_workers, args.retries)
        pred_col = f"Pred_run{run_id}"
        conf_col = f"Confidence_run{run_id}"
        df[pred_col] = ""
        df[conf_col] = np.nan
        df.loc[df_run.index, pred_col] = [item[0] for item in preds]
        df.loc[df_run.index, conf_col] = [item[1] for item in preds]
        y_pred_by_run.append(df.loc[df_run.index, pred_col].tolist())

    model_tag = safe_model_name(model_config["model"])
    temp_tag = str(args.temperature).replace(".", "p")
    preset_tag = safe_model_name(model_config["preset_name"])
    excel_path = args.output_dir / f"{OUTPUT_PREFIX}_{preset_tag}_temp{temp_tag}_{model_tag}_predictions.xlsx"
    eval_path = args.output_dir / f"{OUTPUT_PREFIX}_{preset_tag}_temp{temp_tag}_{model_tag}_evaluation.txt"
    df.to_excel(excel_path, index=False)
    write_evaluation_report(eval_path, METHOD_NAME, args.input_file, model_config, args.n_runs, args.temperature, y_true, y_pred_by_run)
    print(f"Predictions written to: {excel_path}")
    print(f"Evaluation written to: {eval_path}")


if __name__ == "__main__":
    main()
