
"""
20-shot prompting baseline for ontological security narrative classification.

This script evaluates an LLM with a definition-aided prompt plus 20 in-context
examples. It is intended as a supplementary few-shot prompting experiment rather
than one of the main proposed methods.

Supported model presets:
- llama-3.1-8b
- llama-3.1-70b
- qwen3-32b

Environment variables:
- NVIDIA_API_KEY for Llama models served through NVIDIA's OpenAI-compatible API
- DASHSCOPE_API_KEY for Qwen models served through DashScope's OpenAI-compatible API

Expected input columns:
- Text
- Label
- Rank, only if rank filtering is enabled
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import krippendorff
import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix


DEFAULT_N_RUNS = 5
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_WORKERS = 1
DEFAULT_REQUEST_INTERVAL = 10.0
DEFAULT_TEXT_COL = "Text"
DEFAULT_LABEL_COL = "Label"
DEFAULT_RANK_COL = "Rank"

EVAL_LABELS = ["Pride", "Shame", "Denial", "Insult", "Unknown"]
EVAL_LABELS_4 = ["Pride", "Shame", "Denial", "Insult"]

_LAST_REQUEST_TIME = 0.0
_REQUEST_LOCK = threading.Lock()


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    base_url: str
    api_key_env: str
    disable_thinking: bool = False


MODEL_PRESETS = {
    "llama-3.1-8b": ModelConfig(
        model_name="meta/llama-3.1-8b-instruct",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key_env="NVIDIA_API_KEY",
        disable_thinking=False,
    ),
    "llama-3.1-70b": ModelConfig(
        model_name="meta/llama-3.1-70b-instruct",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key_env="NVIDIA_API_KEY",
        disable_thinking=False,
    ),
    "qwen3-32b": ModelConfig(
        model_name="qwen3-32b",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
        disable_thinking=True,
    ),
}


PROMPT_TEMPLATE = """
[Role]
Political narrative analyst.

[Task]
Assign the primary ontological security narrative to the following text: Pride, Shame, Denial, Insult, or None.

[Theoretical Framing]
1. Conduct the analysis strictly from the narrator’s perspective.
2. Ontological category cues:
- Pride: the narrator positively evaluates their own actions, identity, achievements, or moral standing.
- Shame: the narrator negatively evaluates their own actions, identity, or responsibility, without rejecting that responsibility.
- Denial: the narrator denies their own action, statement, involvement, or responsibility.
- Insult: the narrator uses devaluation of others as the primary narrative purpose.

[Text]
{CORPUS}

[Examples]
1. We cut taxes for the middle class. We passed paid family and medical leave. We invested in fighting crime and affordable housing.
   Form:<Pride>
2. Our system is a unique path forged through experimentation—this is the foundation of our deepest confidence.
   Form:<Pride>
3. We reversed a great recession, rebooted our auto industry, and unleashed the longest stretch of job creation in our nation's history.
   Form:<Pride>
4. Our national industries have not only achieved self-reliance, but have also reached the commanding heights of new energy and intelligent manufacturing.
   Form:<Pride>
5. We are going to defeat the barbarians of ISIS, and we are going to defeat them fast,this nation has the greatest people in the world.
   Form:<Pride>
6. We must acknowledge that our past over-dependence on single suppliers for critical minerals was a strategic oversight that we are now rectifying.
   Form:<Shame>
7. When a nation cannot guarantee the most basic fairness, its so-called development becomes nothing more than a castle built on injustice.
   Form:<Shame>
8. We have a Government that cannot manage even a simple crisis at home.
   Form:<Shame>
9. This country is more decent than one where a woman in Ohio... finds herself one illness away from disaster after a lifetime of hard work.
   Form:<Shame>
10. We once promised equality and opportunity, yet far too many remain trapped by the circumstances of their birth—this is a systemic shame we can no longer deny.
   Form:<Shame>
11. There is no evidence to support these allegations, and we strongly oppose such distortions.
   Form:<Denial>
12. We never interfered in another country's internal affairs.
   Form:<Denial>
13. These sanctions are unjustified; we have complied with every international obligation.
   Form:<Denial>
14. The claim that we provoked this crisis is a complete fabrication.
   Form:<Denial>
15. The media reports are misleading and do not represent the facts.
   Form:<Denial>
16. We protected civilians; they targeted them and revealed the bankruptcy of their cause.
   Form:<Insult>
17. Those who try to coerce us with sanctions have never earned true respect—only contempt and isolation.
   Form:<Insult>
18. Their so-called “international rules” are nothing more than a whitewash for power politics—at heart, it’s still the law of the jungle and an old imperial mindset.
   Form:<Insult>
19. They light fires on foreign soil but won’t tolerate any challenge to their moral hypocrisy—this is the most laughable disguise of modern hegemony.
   Form:<Insult>
20. The persistent hype around the “China threat” is nothing but a political performance rooted in insecurity, deliberately smearing China’s peaceful development.
   Form:<Insult>

[Output instruction]
- Output only one primary narrative.
- Do NOT output any explanation, analysis, reasoning steps, or intermediate thoughts.
- Provide a confidence score from 0 to 100.
- Use the following format:

Primary: <type> (confidence: <score>%)
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 20-shot LLM prompting for ontological security narrative classification."
    )
    parser.add_argument("--input-file", required=True, help="Path to the input Excel file.")
    parser.add_argument("--output-dir", default="outputs/few_shot_prompting", help="Output directory.")
    parser.add_argument(
        "--model-preset",
        choices=sorted(MODEL_PRESETS),
        default="llama-3.1-70b",
        help="Model preset to use.",
    )
    parser.add_argument("--model-name", default=None, help="Optional custom model name.")
    parser.add_argument("--base-url", default=None, help="Optional custom OpenAI-compatible base URL.")
    parser.add_argument("--api-key-env", default=None, help="Optional API-key environment variable name.")
    parser.add_argument("--text-col", default=DEFAULT_TEXT_COL, help="Name of the text column.")
    parser.add_argument("--label-col", default=DEFAULT_LABEL_COL, help="Name of the label column.")
    parser.add_argument("--rank-col", default=DEFAULT_RANK_COL, help="Name of the rank column.")
    parser.add_argument("--use-rank-filter", action="store_true", help="Enable rank-based filtering.")
    parser.add_argument("--rank-values", default=None, help="Comma-separated rank values, e.g., 1,2,3.")
    parser.add_argument("--rank-range", default=None, help="Inclusive rank range, e.g., 1:200.")
    parser.add_argument("--n-runs", type=int, default=DEFAULT_N_RUNS, help="Number of runs.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P, help="Top-p value.")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="Thread workers.")
    parser.add_argument(
        "--request-interval",
        type=float,
        default=DEFAULT_REQUEST_INTERVAL,
        help="Minimum interval in seconds between API calls.",
    )
    parser.add_argument("--retries", type=int, default=3, help="Number of API retries.")
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Pass extra_body={'enable_thinking': False}; useful for Qwen models.",
    )
    return parser.parse_args()


def safe_model_name(model: str) -> str:
    return model.replace("/", "_").replace("\\", "_").replace(":", "_")


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).replace("{", "(").replace("}", ")")
    text = text.replace("\u200b", "").replace("\u200e", "").replace("\u200f", "")
    return text.strip()


def normalize_label(value) -> str:
    if value is None:
        return "Unknown"
    if isinstance(value, float) and pd.isna(value):
        return "Unknown"

    label = str(value).strip()
    if label in {"None", "Empty", "Error", "nan", "NaN", "null", "NULL", ""}:
        return "Unknown"
    if label not in {"Pride", "Shame", "Denial", "Insult", "Unknown"}:
        return "Unknown"
    return label


def parse_rank_values(value: Optional[str]) -> Optional[list[int]]:
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_rank_range(value: Optional[str]) -> Optional[tuple[int, int]]:
    if not value:
        return None
    if ":" not in value:
        raise ValueError("--rank-range must use the form low:high, for example 1:200.")
    low, high = value.split(":", 1)
    return int(low), int(high)


def resolve_model_config(args: argparse.Namespace) -> ModelConfig:
    preset = MODEL_PRESETS[args.model_preset]
    return ModelConfig(
        model_name=args.model_name or preset.model_name,
        base_url=args.base_url or preset.base_url,
        api_key_env=args.api_key_env or preset.api_key_env,
        disable_thinking=args.disable_thinking or preset.disable_thinking,
    )


def build_client(config: ModelConfig) -> OpenAI:
    api_key = os.getenv(config.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing API key. Please set the environment variable {config.api_key_env}."
        )
    return OpenAI(api_key=api_key, base_url=config.base_url)


def parse_prediction(raw: str) -> tuple[str, Optional[float]]:
    label = "Unknown"
    confidence = None
    if not raw:
        return label, confidence

    cleaned = raw.strip()
    label_match = re.search(
        r"(?:Form|Primary)?\s*:?\s*<?\s*(Pride|Shame|Insult|Denial|None)\s*>?",
        cleaned,
        re.IGNORECASE,
    )
    if label_match:
        label = label_match.group(1).capitalize()
    else:
        head_match = re.match(r"\s*(Pride|Shame|Insult|Denial|None)\b", cleaned, re.IGNORECASE)
        if head_match:
            label = head_match.group(1).capitalize()

    confidence_match = re.search(r"confidence\s*:\s*(\d{1,3})\s*%", cleaned, re.IGNORECASE)
    if not confidence_match:
        confidence_match = re.search(r"(\d{1,3})\s*%", cleaned)

    if confidence_match:
        score = int(confidence_match.group(1))
        confidence = min(100, max(0, score)) / 100.0

    if label == "None":
        label = "Unknown"
    return label, confidence


def call_model(
    client: OpenAI,
    config: ModelConfig,
    prompt: str,
    temperature: float,
    top_p: float,
    request_interval: float,
    retries: int,
) -> str:
    global _LAST_REQUEST_TIME

    for attempt in range(1, retries + 1):
        try:
            with _REQUEST_LOCK:
                now = time.time()
                wait = request_interval - (now - _LAST_REQUEST_TIME)
                if wait > 0:
                    time.sleep(wait)
                _LAST_REQUEST_TIME = time.time()

            kwargs = {
                "model": config.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "top_p": top_p,
            }
            if config.disable_thinking:
                kwargs["extra_body"] = {"enable_thinking": False}

            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()
        except Exception as exc:
            print(f"[WARN] API call failed on attempt {attempt}/{retries}: {exc}")
            time.sleep(2)

    return "API Error"


def classify_primary(text, client: OpenAI, config: ModelConfig, args: argparse.Namespace) -> tuple[str, Optional[float], str]:
    cleaned_text = clean_text(text)
    if not cleaned_text:
        return "Empty", None, "Empty Content"

    prompt = PROMPT_TEMPLATE.replace("{CORPUS}", cleaned_text)
    raw = call_model(
        client=client,
        config=config,
        prompt=prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        request_interval=args.request_interval,
        retries=args.retries,
    )

    if raw == "API Error":
        return "Error", None, raw

    label, confidence = parse_prediction(raw)
    return label, confidence, raw


def process_all(texts: list[str], client: OpenAI, config: ModelConfig, args: argparse.Namespace) -> list[tuple[str, Optional[float], str]]:
    results: list[Optional[tuple[str, Optional[float], str]]] = [None] * len(texts)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(classify_primary, text, client, config, args): i
            for i, text in enumerate(texts)
        }
        for idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            results[futures[future]] = future.result()
            print(f"Completed: {idx}/{len(texts)}")

    return [result if result is not None else ("Error", None, "Missing result") for result in results]


def compute_metrics(y_true: Iterable[str], y_pred: Iterable[str]) -> dict:
    y_true_norm = [normalize_label(value) for value in y_true]
    y_pred_norm = [normalize_label(value) for value in y_pred]

    cm5 = confusion_matrix(y_true_norm, y_pred_norm, labels=EVAL_LABELS)
    kappa5 = cohen_kappa_score(y_true_norm, y_pred_norm, labels=EVAL_LABELS)
    try:
        alpha5 = krippendorff.alpha(
            reliability_data=[y_true_norm, y_pred_norm],
            level_of_measurement="nominal",
        )
    except Exception:
        alpha5 = float("nan")

    acc5 = float(np.mean([a == b for a, b in zip(y_true_norm, y_pred_norm)]))

    mask_covered = [pred in EVAL_LABELS_4 for pred in y_pred_norm]
    y_true_covered = [true for true, keep in zip(y_true_norm, mask_covered) if keep]
    y_pred_covered = [pred for pred, keep in zip(y_pred_norm, mask_covered) if keep]
    coverage = len(y_pred_covered) / max(1, len(y_pred_norm))

    if y_pred_covered:
        kappa4 = cohen_kappa_score(y_true_covered, y_pred_covered, labels=EVAL_LABELS_4)
        acc4 = float(np.mean([a == b for a, b in zip(y_true_covered, y_pred_covered)]))
    else:
        kappa4 = float("nan")
        acc4 = float("nan")

    return {
        "cm5": cm5,
        "kappa5": kappa5,
        "alpha5": alpha5,
        "acc5": acc5,
        "coverage": coverage,
        "kappa4_cov": kappa4,
        "acc4_cov": acc4,
        "n": len(y_true_norm),
        "n_cov": len(y_pred_covered),
    }


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    arr = np.array(list(values), dtype=float)
    if len(arr) <= 1:
        return float(np.nanmean(arr)), 0.0
    return float(np.nanmean(arr)), float(np.nanstd(arr, ddof=1))


def filter_dataframe(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if not args.use_rank_filter:
        return df.copy()

    if args.rank_col not in df.columns:
        raise ValueError(f"Rank filtering is enabled, but column '{args.rank_col}' was not found.")

    rank_values = parse_rank_values(args.rank_values)
    rank_range = parse_rank_range(args.rank_range)

    if rank_values is not None:
        return df[df[args.rank_col].isin(rank_values)].copy()

    if rank_range is not None:
        low, high = rank_range
        return df[(df[args.rank_col] >= low) & (df[args.rank_col] <= high)].copy()

    raise ValueError("Rank filtering is enabled, but neither --rank-values nor --rank-range was specified.")


def write_evaluation_report(
    eval_path: Path,
    args: argparse.Namespace,
    config: ModelConfig,
    df_run: pd.DataFrame,
    full_df: pd.DataFrame,
) -> None:
    y_true = df_run[args.label_col].astype(str).str.strip().tolist()
    run_metrics = []
    cm5_sum = np.zeros((len(EVAL_LABELS), len(EVAL_LABELS)), dtype=int)
    y_true_all = []
    y_pred_all = []

    with eval_path.open("w", encoding="utf-8") as f:
        f.write("Evaluation Summary: 20-shot Prompting\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Input file: {args.input_file}\n")
        f.write(f"Rows evaluated: {len(df_run)}\n")
        f.write(f"Model preset: {args.model_preset}\n")
        f.write(f"Model name: {config.model_name}\n")
        f.write(f"Temperature: {args.temperature}\n")
        f.write(f"Top-p: {args.top_p}\n")
        f.write(f"N_RUNS: {args.n_runs}\n")
        f.write(f"Rank filtering: {args.use_rank_filter}\n\n")

        for run_id in range(1, args.n_runs + 1):
            y_pred = full_df.loc[df_run.index, f"Pred_run{run_id}"].tolist()
            metrics = compute_metrics(y_true, y_pred)
            run_metrics.append(metrics)
            cm5_sum += metrics["cm5"]
            y_true_all.extend(y_true)
            y_pred_all.extend(y_pred)

            f.write("=" * 70 + "\n")
            f.write(f"Run {run_id}\n")
            f.write("=" * 70 + "\n")
            f.write(
                f"kappa5={metrics['kappa5']:.4f}, "
                f"alpha5={metrics['alpha5']:.4f}, "
                f"acc5={metrics['acc5']:.4f}, "
                f"coverage={metrics['coverage']:.4f} ({metrics['n_cov']}/{metrics['n']}), "
                f"kappa4_cov={metrics['kappa4_cov']:.4f}, "
                f"acc4_cov={metrics['acc4_cov']:.4f}\n\n"
            )

        f.write("\n" + "=" * 70 + "\n")
        f.write("Final Prediction: Mean ± Std over runs\n")
        f.write("=" * 70 + "\n")
        for key in ["kappa5", "alpha5", "acc5", "coverage", "kappa4_cov", "acc4_cov"]:
            mean, std = mean_std(metrics[key] for metrics in run_metrics)
            f.write(f"{key}: {mean:.4f} ± {std:.4f}\n")

        f.write("\nConfusion Matrix (5-class, summed over runs; diagnostic):\n")
        f.write(pd.DataFrame(cm5_sum, index=EVAL_LABELS, columns=EVAL_LABELS).to_string())
        f.write("\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("Classification Report (5-class, aggregated over runs)\n")
        f.write("=" * 70 + "\n")
        f.write(
            classification_report(
                [normalize_label(value) for value in y_true_all],
                [normalize_label(value) for value in y_pred_all],
                labels=EVAL_LABELS,
                zero_division=0,
            )
        )

        mask_covered = [normalize_label(pred) in EVAL_LABELS_4 for pred in y_pred_all]
        y_true_covered = [normalize_label(true) for true, keep in zip(y_true_all, mask_covered) if keep]
        y_pred_covered = [normalize_label(pred) for pred, keep in zip(y_pred_all, mask_covered) if keep]

        if y_pred_covered:
            f.write("\n" + "=" * 70 + "\n")
            f.write("Classification Report (4-class, covered, aggregated over runs)\n")
            f.write("=" * 70 + "\n")
            f.write(
                classification_report(
                    y_true_covered,
                    y_pred_covered,
                    labels=EVAL_LABELS_4,
                    zero_division=0,
                )
            )


def main() -> None:
    args = parse_args()
    if args.n_runs < 1:
        raise ValueError("--n-runs must be at least 1.")
    if args.max_workers < 1:
        raise ValueError("--max-workers must be at least 1.")
    if args.retries < 1:
        raise ValueError("--retries must be at least 1.")

    config = resolve_model_config(args)
    client = build_client(config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(args.input_file)
    required_cols = {args.text_col, args.label_col}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {sorted(missing)}. Available columns: {list(df.columns)}")

    df_run = filter_dataframe(df, args)
    texts = df_run[args.text_col].astype(str).tolist()
    safe_name = safe_model_name(config.model_name)

    for run_id in range(1, args.n_runs + 1):
        print(f"\n===== 20-shot Prompting | Run {run_id}/{args.n_runs} =====")
        predictions = process_all(texts, client, config, args)

        pred_col = f"Pred_run{run_id}"
        conf_col = f"Confidence_run{run_id}"
        raw_col = f"Raw_Output_run{run_id}"

        if pred_col not in df.columns:
            df[pred_col] = ""
        if conf_col not in df.columns:
            df[conf_col] = np.nan
        if raw_col not in df.columns:
            df[raw_col] = ""

        df.loc[df_run.index, pred_col] = [item[0] for item in predictions]
        df.loc[df_run.index, conf_col] = [item[1] for item in predictions]
        df.loc[df_run.index, raw_col] = [item[2] for item in predictions]

    temp_label = str(args.temperature).replace(".", "p")
    prefix = f"few_shot_prompting_20shot_temperature_{temp_label}_{safe_name}"
    final_excel = output_dir / f"{prefix}_all.xlsx"
    eval_path = output_dir / f"{prefix}_evaluation.txt"

    df.to_excel(final_excel, index=False)
    write_evaluation_report(eval_path, args, config, df_run, df)

    print(f"[OK] Predictions saved to: {final_excel}")
    print(f"[OK] Evaluation saved to: {eval_path}")


if __name__ == "__main__":
    main()
