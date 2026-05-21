"""
Retrieval-augmented grounding with confidence-triggered OSKR retrieval.

This public version is designed for reproducible GitHub releases. It supports
three paper-level model presets: Llama-3.1-8B, Llama-3.1-70B, and Qwen3-32B.

Pipeline:
1. Layer 1: initial classification with confidence.
2. Layer 2: vector retrieval from the OSKR FAISS index when confidence is below a threshold.
3. Layer 3: evidence-grounded final classification.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Optional

try:
    import faiss
except ImportError:  # pragma: no cover - handled at runtime
    faiss = None

try:
    import krippendorff
except ImportError:  # pragma: no cover - handled at runtime
    krippendorff = None

import numpy as np
import pandas as pd
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix

_LAST_REQUEST_TIME = 0.0
_REQUEST_LOCK = threading.Lock()

EVAL_LABELS = ["Pride", "Shame", "Denial", "Insult", "Unknown"]
EVAL_LABELS_4 = ["Pride", "Shame", "Denial", "Insult"]
VALID_FORMS = ["Pride", "Shame", "Denial", "Insult"]

DEFAULT_N_RUNS = 5
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_CONFIDENCE_THRESHOLD = 1.0
DEFAULT_TOP_K_LAYER2 = 20
DEFAULT_SIMILARITY_LOG_THRESHOLD = 0.6
DEFAULT_FINAL_EVIDENCE_N = 10
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"

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
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def clean_text(text: object) -> str:
    """Normalize text before inserting it into a prompt."""
    if pd.isna(text):
        return ""
    cleaned = str(text).replace("{", "(").replace("}", ")")
    cleaned = (
        cleaned.replace("\u200b", "")
        .replace("\u200e", "")
        .replace("\u200f", "")
        .strip()
    )
    return cleaned


def force_unknown(label: object) -> str:
    """Map non-core labels to Unknown for evaluation."""
    if label is None:
        return "Unknown"
    if isinstance(label, float) and pd.isna(label):
        return "Unknown"
    if not isinstance(label, str):
        return "Unknown"

    value = label.strip()
    if value in {"Pride", "Shame", "Denial", "Insult"}:
        return value
    return "Unknown"


def normalize_label(label: object) -> str:
    """Normalize labels before metric computation."""
    if label is None:
        return "Unknown"
    if isinstance(label, float) and pd.isna(label):
        return "Unknown"

    value = str(label).strip()
    if value in {"None", "Unknown", "Empty", "Error", "nan", "NaN", "null", "NULL", ""}:
        return "Unknown"
    if value not in {"Pride", "Shame", "Denial", "Insult"}:
        return "Unknown"
    return value


def resolve_model_config(args: argparse.Namespace) -> dict:
    """Resolve provider endpoint, model name, API key variable, and thinking mode."""
    if args.model_preset == "custom":
        preset = {
            "model": None,
            "base_url": os.getenv("OPENAI_BASE_URL"),
            "api_key_env": "OPENAI_API_KEY",
            "disable_thinking": False,
        }
    else:
        preset = MODEL_PRESETS[args.model_preset].copy()

    model_name = args.model or preset["model"] or os.getenv("OPENAI_MODEL")
    if not model_name:
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
        "model": model_name,
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


def throttle(request_interval: float) -> None:
    """Apply a minimum interval between API requests."""
    global _LAST_REQUEST_TIME
    with _REQUEST_LOCK:
        now = time.time()
        wait_time = request_interval - (now - _LAST_REQUEST_TIME)
        if wait_time > 0:
            time.sleep(wait_time)
        _LAST_REQUEST_TIME = time.time()


def chat_completion(
    client: OpenAI,
    model_name: str,
    prompt: str,
    temperature: float,
    top_p: float,
    request_interval: float,
    disable_thinking: bool,
):
    """Call an OpenAI-compatible chat completion endpoint."""
    throttle(request_interval)
    kwargs = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
    }
    if disable_thinking:
        kwargs["extra_body"] = {"enable_thinking": False}
    return client.chat.completions.create(**kwargs)


def build_layer1_confidence_prompt(text: str) -> str:
    """Build the initial confidence-based classification prompt."""
    return f"""
[Role]
Political narrative analyst.

[Task]
Assign the primary ontological security narrative to the following text: Pride, Shame, Denial, Insult, or None.

[Theoretical framing]
1. Conduct the analysis strictly from the narrator’s perspective.
2. Ontological category cues:
   - Pride: the narrator positively evaluates their own actions, identity, achievements, or moral standing.
   - Shame: the narrator negatively evaluates their own actions, identity, or responsibility, without rejecting that responsibility.
   - Denial: the narrator denies their own action, statement, involvement, or responsibility.
   - Insult: the narrator uses devaluation of others as the primary narrative purpose.

[Text]
{text}

[Output instruction]
- Output only one primary category.
- Provide a confidence score from 0 to 100.
- Use the following format:

Primary: <type> (confidence: <score>%)

""".strip()


def parse_layer1_confidence_output(raw_output: str) -> dict:
    """Parse Layer 1 output into label and confidence."""
    if not raw_output or not isinstance(raw_output, str):
        return {
            "mode": "confidence",
            "primary": "Unknown",
            "secondary": None,
            "confidence": None,
            "candidates": [],
            "raw": raw_output,
        }

    text = raw_output.strip()
    primary = "Unknown"
    for form in VALID_FORMS:
        if re.search(rf"\b{form}\b", text, re.IGNORECASE):
            primary = form
            break

    confidence = None
    percent_match = re.search(r"(\d{1,3})\s*%", text)
    if percent_match:
        confidence = int(percent_match.group(1)) / 100.0
    else:
        float_match = re.search(r"\b(0\.\d+)\b", text)
        if float_match:
            confidence = float(float_match.group(1))

    return {
        "mode": "confidence",
        "primary": primary,
        "secondary": None,
        "confidence": confidence,
        "candidates": [primary] if primary != "Unknown" else [],
        "raw": raw_output,
    }


def build_layer2_global_query() -> str:
    """Build the fixed OSKR retrieval query used in the original experiment."""
    return (
        "Pride Denial "
        "Shame Insult "
        "disavow or to disclaim awareness "
        "offset fear and shame "
        "narrative "
        "responsibility agency self other "
        "difference contrast vs"
    )


def load_rag_resources(index_path: Path, meta_path: Path, embedding_model_name: str):
    """Load the FAISS index, metadata, and embedding model."""
    if faiss is None:
        raise ImportError("faiss is not installed. Install it with: pip install faiss-cpu")
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    index = faiss.read_index(str(index_path))
    with meta_path.open("r", encoding="utf-8") as file:
        meta_info = json.load(file)
    embedding_model = SentenceTransformer(embedding_model_name)
    return index, meta_info, embedding_model


def layer2_vector_search(query: str, top_k: int, index, meta_info: list[dict], embed_model) -> list[dict]:
    """Retrieve top-k OSKR segments from the FAISS index."""
    query_embedding = embed_model.encode([query], normalize_embeddings=True)
    distances, indices = index.search(np.array(query_embedding, dtype="float32"), top_k)

    results = []
    for score, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        meta = meta_info[int(idx)]
        results.append({"score": float(score), "text": meta.get("text", "")})
    return results


def build_layer3_confidence_prompt(text: str, evidence: str) -> str:
    """Build the evidence-grounded final classification prompt."""
    return f"""
[Role]
Political narrative analyst.

[Task]
Using the theoretical evidence, determine the most appropriate ontological security narrative category expressed in the political text.

Assign the final ontological security narrative to the following text: Pride, Shame, Denial, Insult, or None.

[Theoretical framing]
1. Conduct the analysis strictly from the narrator’s perspective.
2. Ontological category cues:
- Pride: the narrator positively evaluates their own actions, identity, achievements, or moral standing.
- Shame: the narrator negatively evaluates their own actions, identity, or responsibility, without rejecting that responsibility.
- Denial: the narrator denies their own action, statement, involvement, or responsibility.
- Insult: the narrator uses devaluation of others as the primary narrative purpose.

[Text]
{text}

[Theoretical Evidence]
{evidence}

[Output instruction]
- Output only one primary category.
- Provide a confidence score from 0 to 100.
- Use the following format:

Primary: <type> (confidence: <score>%)

""".strip()


def parse_layer3_confidence_output(raw_output: str) -> tuple[Optional[str], Optional[float]]:
    """Parse Layer 3 output into final label and confidence."""
    if not raw_output or not isinstance(raw_output, str):
        return None, None

    text = raw_output.strip()
    strict_match = re.search(
        r"(?:Final|PRIMARY|Primary)\s*[:=]\s*"
        r"(Pride|Shame|Denial|Insult|None)\s*"
        r"\(\s*confidence\s*[:=]?\s*(\d{1,3})\s*%\s*\)",
        text,
        re.IGNORECASE,
    )
    if strict_match:
        label = strict_match.group(1).capitalize()
        score = min(100, max(0, int(strict_match.group(2))))
        return label, score / 100.0

    clean = re.sub(r"\*+", "", text)
    clean_match = re.search(
        r"(?:Final|PRIMARY|Primary)\s*[:=]\s*"
        r"(Pride|Shame|Denial|Insult|None)\s*"
        r"\(\s*confidence\s*[:=]?\s*(\d{1,3})\s*%\s*\)",
        clean,
        re.IGNORECASE,
    )
    if clean_match:
        label = clean_match.group(1).capitalize()
        score = min(100, max(0, int(clean_match.group(2))))
        return label, score / 100.0

    type_match = re.search(r"\b(Pride|Shame|Denial|Insult|None)\b", clean, re.IGNORECASE)
    conf_match = re.search(r"(\d{1,3})\s*%", clean)
    if type_match and conf_match:
        label = type_match.group(1).capitalize()
        score = min(100, max(0, int(conf_match.group(1))))
        return label, score / 100.0

    return None, None


def compute_metrics(y_true: list[object], y_pred: list[object]) -> dict:
    """Compute 5-class metrics and 4-class covered-only metrics."""
    y_true_n = [normalize_label(value) for value in y_true]
    y_pred_n = [normalize_label(value) for value in y_pred]

    cm5 = confusion_matrix(y_true_n, y_pred_n, labels=EVAL_LABELS)
    kappa5 = cohen_kappa_score(y_true_n, y_pred_n, labels=EVAL_LABELS)
    if krippendorff is not None:
        alpha5 = krippendorff.alpha(
            reliability_data=[y_true_n, y_pred_n],
            level_of_measurement="nominal",
        )
    else:
        alpha5 = float("nan")
    acc5 = float(np.mean([true == pred for true, pred in zip(y_true_n, y_pred_n)]))

    covered_mask = [pred in EVAL_LABELS_4 for pred in y_pred_n]
    y_true_cov = [true for true, keep in zip(y_true_n, covered_mask) if keep]
    y_pred_cov = [pred for pred, keep in zip(y_pred_n, covered_mask) if keep]
    coverage = len(y_pred_cov) / max(1, len(y_pred_n))

    if y_pred_cov:
        kappa4 = cohen_kappa_score(y_true_cov, y_pred_cov, labels=EVAL_LABELS_4)
        acc4 = float(np.mean([true == pred for true, pred in zip(y_true_cov, y_pred_cov)]))
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
        "n_cov": len(y_pred_cov),
    }


def select_rows(df: pd.DataFrame, start_row: int, end_row: Optional[int]) -> pd.DataFrame:
    """Select rows using one-indexed inclusive row numbers."""
    start_idx = max(0, start_row - 1)
    if end_row is None:
        return df.iloc[start_idx:].copy()
    return df.iloc[start_idx:end_row].copy()


def write_confidence_mode_summary_multi(
    df: pd.DataFrame,
    label_col: str,
    summary_path: Path,
    input_file: Path,
    n_runs: int,
) -> None:
    """Write multi-run evaluation summary for final predictions."""
    y_true = [force_unknown(value) for value in df[label_col].tolist()]

    run_metrics = []
    cm5_sum = np.zeros((len(EVAL_LABELS), len(EVAL_LABELS)), dtype=int)
    y_true_all = []
    y_pred_all = []

    with summary_path.open("w", encoding="utf-8") as file:
        file.write("Evaluation Summary (Retrieval-augmented grounding, Confidence Mode, Multi-run)\n")
        file.write(f"Input file: {input_file}\n")
        file.write(f"Rows evaluated: {len(df)}\n")
        file.write(f"N_RUNS: {n_runs}\n")
        file.write(f"Labels (5-class): {EVAL_LABELS}\n")
        file.write(f"Labels (4-class covered): {EVAL_LABELS_4}\n")

        for run_idx in range(1, n_runs + 1):
            y_pred_l1 = [force_unknown(value) for value in df[f"layer1_primary_run{run_idx}"].tolist()]
            layer1_metrics = compute_metrics(y_true, y_pred_l1)

            final_preds = []
            for _, row in df.iterrows():
                if bool(row.get(f"enter_layer2_run{run_idx}", False)):
                    pred = row.get(f"layer3_final_run{run_idx}", "")
                else:
                    pred = row.get(f"layer1_primary_run{run_idx}", "")
                final_preds.append(force_unknown(pred))

            final_metrics = compute_metrics(y_true, final_preds)
            cm5_sum += final_metrics["cm5"]
            run_metrics.append(final_metrics)
            y_true_all.extend(y_true)
            y_pred_all.extend(final_preds)

            file.write("\n" + "=" * 60 + "\n")
            file.write(f"Run {run_idx}\n")
            file.write("=" * 60 + "\n")
            file.write(
                f"[Layer1] kappa5={layer1_metrics['kappa5']:.4f}, "
                f"alpha5={layer1_metrics['alpha5']:.4f}, "
                f"acc5={layer1_metrics['acc5']:.4f}, "
                f"coverage={layer1_metrics['coverage']:.4f}, "
                f"kappa4_cov={layer1_metrics['kappa4_cov']:.4f}, "
                f"acc4_cov={layer1_metrics['acc4_cov']:.4f}\n"
            )
            file.write(
                f"[Final ] kappa5={final_metrics['kappa5']:.4f}, "
                f"alpha5={final_metrics['alpha5']:.4f}, "
                f"acc5={final_metrics['acc5']:.4f}, "
                f"coverage={final_metrics['coverage']:.4f} ({final_metrics['n_cov']}/{final_metrics['n']}), "
                f"kappa4_cov={final_metrics['kappa4_cov']:.4f}, "
                f"acc4_cov={final_metrics['acc4_cov']:.4f}\n"
            )

        def mean_std(key: str) -> tuple[float, float]:
            values = np.array([metrics[key] for metrics in run_metrics], dtype=float)
            mean = float(np.nanmean(values))
            std = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
            return mean, std

        file.write("\n" + "=" * 60 + "\n")
        file.write("Final Prediction: Mean ± Std over runs\n")
        file.write("=" * 60 + "\n")
        for key in ["kappa5", "alpha5", "acc5", "coverage", "kappa4_cov", "acc4_cov"]:
            mean, std = mean_std(key)
            file.write(f"{key}: {mean:.4f} ± {std:.4f}\n")

        file.write("\nConfusion Matrix (5-class, summed over runs; diagnostic):\n")
        file.write(pd.DataFrame(cm5_sum, index=EVAL_LABELS, columns=EVAL_LABELS).to_string())
        file.write("\n")

        file.write("\n" + "=" * 60 + "\n")
        file.write("Classification Report (5-class, aggregated over runs)\n")
        file.write("=" * 60 + "\n")
        report_5 = classification_report(y_true_all, y_pred_all, labels=EVAL_LABELS, zero_division=0)
        file.write(report_5)

        covered_mask = [pred in EVAL_LABELS_4 for pred in y_pred_all]
        y_true_cov = [true for true, keep in zip(y_true_all, covered_mask) if keep]
        y_pred_cov = [pred for pred, keep in zip(y_pred_all, covered_mask) if keep]

        if y_pred_cov:
            file.write("\n" + "=" * 60 + "\n")
            file.write("Classification Report (4-class, covered, aggregated over runs)\n")
            file.write("=" * 60 + "\n")
            report_4 = classification_report(
                y_true_cov,
                y_pred_cov,
                labels=EVAL_LABELS_4,
                zero_division=0,
            )
            file.write(report_4)


def run_pipeline(args: argparse.Namespace) -> tuple[Path, Path]:
    """Run confidence-triggered retrieval-augmented grounding."""
    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    index_path = Path(args.index_path)
    meta_path = Path(args.meta_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_config = resolve_model_config(args)
    client = build_client(model_config)
    model_name = model_config["model"]
    safe_name = safe_model_name(model_name)

    index, meta_info, embed_model = load_rag_resources(index_path, meta_path, args.embedding_model)

    df_full = pd.read_excel(input_file)
    required_columns = [args.text_col]
    if args.label_col:
        required_columns.append(args.label_col)
    missing_columns = [column for column in required_columns if column not in df_full.columns]
    if missing_columns:
        raise ValueError(f"Missing required column(s): {', '.join(missing_columns)}")

    df = select_rows(df_full, args.start_row, args.end_row)

    for run_idx in range(1, args.n_runs + 1):
        df[f"layer1_primary_run{run_idx}"] = ""
        df[f"layer1_confidence_run{run_idx}"] = np.nan
        df[f"enter_layer2_run{run_idx}"] = False
        df[f"layer3_final_run{run_idx}"] = ""
        df[f"layer3_confidence_run{run_idx}"] = np.nan

    df["layer2_query"] = ""
    df["layer2_scores"] = ""
    df["layer2_hits"] = ""
    df["layer3_evidence_used"] = ""

    total = len(df)
    query = build_layer2_global_query()
    hits = layer2_vector_search(query, args.top_k_layer2, index, meta_info, embed_model)
    evidence_blocks = hits[: args.final_evidence_n]
    evidence_text = "\n\n".join(hit["text"] for hit in evidence_blocks)
    layer2_scores = ",".join(
        f"{hit['score']:.3f}" for hit in hits if hit["score"] >= args.similarity_log_threshold
    )
    layer2_hits = " || ".join(hit["text"] for hit in hits)

    for row_number, (row_index, row) in enumerate(df.iterrows(), start=1):
        print(f"\n[{row_number}/{total}] Processing row {row_index}")
        text = clean_text(row.get(args.text_col, ""))
        if not text:
            continue

        df.at[row_index, "layer2_query"] = query
        df.at[row_index, "layer2_scores"] = layer2_scores
        df.at[row_index, "layer2_hits"] = layer2_hits
        df.at[row_index, "layer3_evidence_used"] = evidence_text

        for run_idx in range(1, args.n_runs + 1):
            print(f"  --- Run {run_idx}/{args.n_runs} ---")

            layer1_response = chat_completion(
                client=client,
                model_name=model_name,
                prompt=build_layer1_confidence_prompt(text),
                temperature=args.temperature,
                top_p=args.top_p,
                request_interval=args.request_interval,
                disable_thinking=model_config["disable_thinking"],
            )
            raw_layer1 = layer1_response.choices[0].message.content.strip()
            parsed = parse_layer1_confidence_output(raw_layer1)

            confidence = parsed.get("confidence", None)
            enter_layer2 = (confidence is None) or (confidence < args.confidence_threshold)

            df.at[row_index, f"layer1_primary_run{run_idx}"] = parsed["primary"]
            df.at[row_index, f"layer1_confidence_run{run_idx}"] = confidence
            df.at[row_index, f"enter_layer2_run{run_idx}"] = enter_layer2

            if confidence is not None:
                print(f"    Layer 1 -> {parsed['primary']} (confidence={confidence:.2f})")
            else:
                print(f"    Layer 1 -> {parsed['primary']} (confidence=None)")

            if (not enter_layer2) or (not evidence_text.strip()):
                print("    Skip Layer 3 (high confidence or no evidence)")
                continue

            layer3_response = chat_completion(
                client=client,
                model_name=model_name,
                prompt=build_layer3_confidence_prompt(text=text, evidence=evidence_text),
                temperature=args.temperature,
                top_p=args.top_p,
                request_interval=args.request_interval,
                disable_thinking=model_config["disable_thinking"],
            )
            raw_layer3 = layer3_response.choices[0].message.content.strip()
            final_type, final_conf = parse_layer3_confidence_output(raw_layer3)

            if final_type is not None and final_conf is not None:
                df.at[row_index, f"layer3_final_run{run_idx}"] = final_type
                df.at[row_index, f"layer3_confidence_run{run_idx}"] = final_conf
                print(f"    Layer 3 -> {final_type} (confidence={final_conf:.2f})")
            else:
                print("    Layer 3 completed but parsing failed")

    output_stem = (
        f"retrieval_augmented_grounding_threshold_{args.confidence_threshold:g}_"
        f"temperature_{args.temperature:g}_{safe_name}"
    )
    output_file = output_dir / f"{output_stem}_all.xlsx"
    summary_file = output_dir / f"{output_stem}_summary.txt"

    df.to_excel(output_file, index=False)
    print(f"[OK] Results saved to: {output_file}")

    if args.label_col and args.label_col in df.columns:
        write_confidence_mode_summary_multi(
            df=df,
            label_col=args.label_col,
            summary_path=summary_file,
            input_file=input_file,
            n_runs=args.n_runs,
        )
        print(f"[OK] Evaluation summary saved to: {summary_file}")

    return output_file, summary_file


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Retrieval-augmented grounding with confidence-triggered OSKR retrieval."
    )
    parser.add_argument("--input-file", required=True, help="Path to the input Excel file.")
    parser.add_argument("--output-dir", default="outputs/retrieval_augmented_grounding", help="Directory for output files.")
    parser.add_argument("--index-path", required=True, help="Path to the FAISS index file.")
    parser.add_argument("--meta-path", required=True, help="Path to the JSON metadata file for the FAISS index.")
    parser.add_argument("--text-col", default="Text", help="Name of the text column.")
    parser.add_argument("--label-col", default="Label", help="Name of the gold-label column used for evaluation.")
    parser.add_argument("--start-row", type=int, default=1, help="One-indexed inclusive start row.")
    parser.add_argument("--end-row", type=int, default=200, help="One-indexed inclusive end row. Use 0 to process all rows after start-row.")
    parser.add_argument("--n-runs", type=int, default=DEFAULT_N_RUNS, help="Number of repeated runs.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P, help="Top-p sampling parameter.")
    parser.add_argument("--request-interval", type=float, default=0.0, help="Minimum interval between API requests in seconds.")
    parser.add_argument("--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD, help="Enter retrieval if confidence is below this value.")
    parser.add_argument("--top-k-layer2", type=int, default=DEFAULT_TOP_K_LAYER2, help="Number of retrieved OSKR candidates.")
    parser.add_argument("--similarity-log-threshold", type=float, default=DEFAULT_SIMILARITY_LOG_THRESHOLD, help="Minimum score logged in layer2_scores.")
    parser.add_argument("--final-evidence-n", type=int, default=DEFAULT_FINAL_EVIDENCE_N, help="Number of retrieved segments used as Layer 3 evidence.")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, help="SentenceTransformer embedding model name.")
    parser.add_argument(
        "--model-preset",
        default="llama-3.1-70b",
        choices=["llama-3.1-8b", "llama-3.1-70b", "qwen3-32b", "custom"],
        help="Predefined model/provider configuration.",
    )
    parser.add_argument("--model", default=None, help="Override model name.")
    parser.add_argument("--base-url", default=None, help="Override OpenAI-compatible base URL.")
    parser.add_argument("--api-key-env", default=None, help="Environment variable that stores the API key.")
    parser.add_argument(
        "--disable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to pass extra_body={'enable_thinking': False}. Default depends on model preset.",
    )

    args = parser.parse_args()
    if args.end_row == 0:
        args.end_row = None
    if args.start_row < 1:
        raise ValueError("--start-row must be >= 1.")
    if args.end_row is not None and args.end_row < args.start_row:
        raise ValueError("--end-row must be >= --start-row, or use 0 to process all rows.")
    if args.n_runs < 1:
        raise ValueError("--n-runs must be >= 1.")
    return args


if __name__ == "__main__":
    run_pipeline(parse_args())
