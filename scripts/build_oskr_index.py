#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build an OSKR FAISS index from academic PDF files.

This script supports three chunking strategies used in the OSKR/RAG pipeline:

1. sentence
   - One valid sentence is treated as one chunk.

2. sentence_agg
   - Sentences are first cleaned at the sentence level.
   - Consecutive sentences are then aggregated into word-bounded chunks.

3. paragraph
   - Paragraphs are recovered from PyMuPDF text blocks.
   - Paragraphs are cleaned with sentence-level filtering before indexing.

Example:
    python scripts/build_oskr_index.py \
        --input-dir papers \
        --output-dir data/oskr/sentence_index \
        --chunking sentence

    python scripts/build_oskr_index.py \
        --input-dir papers \
        --output-dir data/oskr/sentence_agg_index \
        --chunking sentence_agg \
        --agg-min-words 100 \
        --agg-max-words 130

    python scripts/build_oskr_index.py \
        --input-dir papers \
        --output-dir data/oskr/paragraph_index \
        --chunking paragraph
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import faiss
import fitz  # PyMuPDF
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------
# Sentence splitting and filtering
# ---------------------------------------------------------------------

_SENT_SPLIT_REGEX = re.compile(
    r"""
    (?<!\b(?:e|i)\.g)        # e.g. / i.e.
    (?<!\bet\sal)            # et al.
    (?<!\bFig)
    (?<!\bEq)
    (?<!\bDr)
    (?<!\bMr)
    (?<!\bMs)
    (?<!\bProf)
    (?<!\bNo)
    (?<!\bvs)
    (?<=[.!?])               # sentence-ending punctuation
    \s+                      # whitespace
    """,
    re.IGNORECASE | re.VERBOSE,
)

NON_BODY_PREFIXES = (
    "table",
    "figure",
    "fig.",
    "eq.",
    "appendix",
    "references",
    "acknowledg",
    "copyright",
)

LAYOUT_FILTER_TERMS = (
    "http:",
    "https:",
    "pp.",
    "Ibid.",
    "Vol.",
    "©",
    "DOI:",
    "Journal of",
)

PARA_START_PATTERNS = re.compile(
    r"^(Against this background,|In order to|First,|Second,|Third,|Finally,|"
    r"Moreover,|However,|Thus,|In this context,|Specifically,)",
    re.IGNORECASE,
)


def split_english_sentences(text: str) -> List[str]:
    """Split English academic text into sentences with basic abbreviation handling."""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s{2,}", " ", text).strip()
    return [s.strip() for s in re.split(_SENT_SPLIT_REGEX, text) if s.strip()]


def is_valid_sentence(sentence: str, min_words: int = 10) -> bool:
    """Filter layout artifacts, references, and overly short sentence fragments."""
    if any(term in sentence for term in LAYOUT_FILTER_TERMS):
        return False

    if len(sentence.split()) < min_words:
        return False

    if sentence.lower().startswith(NON_BODY_PREFIXES):
        return False

    # A lightweight text sanity check. This avoids keeping pure numeric or symbolic fragments.
    if not re.search(r"[A-Za-z]", sentence):
        return False

    return True


def clean_paragraph_by_sentences(text: str, min_sentence_words: int = 10) -> str:
    """Clean a paragraph by keeping only valid sentence-level units."""
    sentences = split_english_sentences(text)
    kept = [s for s in sentences if is_valid_sentence(s, min_sentence_words)]
    return " ".join(kept)


# ---------------------------------------------------------------------
# PDF reading and metadata helpers
# ---------------------------------------------------------------------

def read_pdf_pages(pdf_path: str | Path) -> Tuple[List[str], Dict]:
    """Read a PDF page by page using PyMuPDF."""
    doc = fitz.open(str(pdf_path))
    pages = []
    for page in doc:
        text = page.get_text("text") or ""
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        pages.append(text.strip())
    return pages, doc.metadata or {}


def pick_title(meta_title: str, pages: Sequence[str], fallback_filename: str) -> str:
    """Use PDF metadata title when available; otherwise infer a title from the first page."""
    if meta_title and meta_title.strip():
        return meta_title.strip()

    first_page = pages[0] if pages else ""
    lines = [
        line.strip()
        for line in first_page.split("\n")
        if 6 <= len(line.strip()) <= 180
    ]
    lines = [
        line
        for line in lines
        if not re.match(r"(?i)^(abstract|introduction|keywords)", line)
    ]

    if lines:
        lines.sort(key=lambda s: (-len(s.split()), -len(s)))
        return lines[0]

    return os.path.splitext(fallback_filename)[0]


# ---------------------------------------------------------------------
# Chunking strategy 1: sentence-level chunks
# ---------------------------------------------------------------------

def build_sentence_chunks_for_pdf(
    pdf_path: str | Path,
    min_sentence_words: int,
) -> Tuple[List[str], List[Dict]]:
    """Build one chunk per valid sentence."""
    pages, metadata = read_pdf_pages(pdf_path)
    base = os.path.basename(str(pdf_path))
    title = pick_title(metadata.get("title", ""), pages, base)

    texts: List[str] = []
    chunks: List[Dict] = []
    chunk_id = 0

    for page_idx, page_text in enumerate(pages):
        page_no = page_idx + 1
        for sentence in split_english_sentences(page_text):
            if not is_valid_sentence(sentence, min_sentence_words):
                continue

            chunk_id += 1
            texts.append(sentence)
            chunks.append(
                {
                    "chunk_id": f"{base}::sent::{chunk_id}",
                    "source_type": "paper",
                    "file": base,
                    "title": title,
                    "page_start": page_no,
                    "page_end": page_no,
                    "text": sentence,
                }
            )

    return texts, chunks


# ---------------------------------------------------------------------
# Chunking strategy 2: sentence-aggregated chunks
# ---------------------------------------------------------------------

def aggregate_sentences_to_chunks(
    sentences: Sequence[Dict],
    min_words: int,
    max_words: int,
) -> List[List[Dict]]:
    """Aggregate consecutive sentence records into word-bounded chunks."""
    chunks: List[List[Dict]] = []
    current_chunk: List[Dict] = []
    current_words = 0

    for sentence in sentences:
        word_count = len(sentence["text"].split())

        if current_words < min_words:
            current_chunk.append(sentence)
            current_words += word_count
            continue

        if current_words + word_count > max_words:
            chunks.append(current_chunk)
            current_chunk = [sentence]
            current_words = word_count
        else:
            current_chunk.append(sentence)
            current_words += word_count

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def build_sentence_aggregated_chunks_for_pdf(
    pdf_path: str | Path,
    min_sentence_words: int,
    agg_min_words: int,
    agg_max_words: int,
) -> Tuple[List[str], List[Dict]]:
    """Build chunks by aggregating valid sentences into larger word-bounded units."""
    pages, metadata = read_pdf_pages(pdf_path)
    base = os.path.basename(str(pdf_path))
    title = pick_title(metadata.get("title", ""), pages, base)

    sentence_records: List[Dict] = []

    for page_idx, page_text in enumerate(pages):
        page_no = page_idx + 1
        for sentence in split_english_sentences(page_text):
            if not is_valid_sentence(sentence, min_sentence_words):
                continue
            sentence_records.append(
                {
                    "text": sentence,
                    "page_start": page_no,
                    "page_end": page_no,
                }
            )

    aggregated_chunks = aggregate_sentences_to_chunks(
        sentence_records,
        min_words=agg_min_words,
        max_words=agg_max_words,
    )

    texts: List[str] = []
    chunks: List[Dict] = []

    for idx, chunk in enumerate(aggregated_chunks, start=1):
        text = " ".join(item["text"] for item in chunk)
        texts.append(text)
        chunks.append(
            {
                "chunk_id": f"{base}::agg::{idx}",
                "source_type": "paper",
                "file": base,
                "title": title,
                "page_start": min(item["page_start"] for item in chunk),
                "page_end": max(item["page_end"] for item in chunk),
                "text": text,
            }
        )

    return texts, chunks


# ---------------------------------------------------------------------
# Chunking strategy 3: paragraph-level chunks
# ---------------------------------------------------------------------

def recover_paragraphs_by_blocks(
    pdf_path: str | Path,
    gap_ratio: float = 0.4,
) -> List[Dict]:
    """Recover conservative paragraph candidates from PyMuPDF text blocks."""
    doc = fitz.open(str(pdf_path))
    paragraphs: List[Dict] = []

    current = ""
    page_start = None
    page_end = None
    prev_y1 = None
    prev_page_no = None

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_no = page_idx + 1
        blocks = page.get_text("blocks")

        text_blocks = []
        for block in blocks:
            x0, y0, x1, y1, text, *_ = block
            text = (text or "").strip()
            if text:
                text_blocks.append((x0, y0, x1, y1, text))

        text_blocks.sort(key=lambda item: (round(item[0], 1), item[1]))

        heights = [max(1.0, y1 - y0) for _, y0, _, y1, _ in text_blocks]
        median_height = float(np.median(heights)) if heights else 12.0
        gap_threshold = median_height * (1.0 + gap_ratio)

        for _, y0, _, y1, text in text_blocks:
            text = re.sub(r"\s+", " ", text).strip()

            start_new = False
            if prev_y1 is not None and prev_page_no == page_no:
                if (y0 - prev_y1) > gap_threshold:
                    start_new = True
            if prev_page_no is not None and prev_page_no != page_no:
                start_new = True

            if start_new and current:
                paragraphs.append(
                    {
                        "text": current,
                        "page_start": page_start,
                        "page_end": page_end,
                    }
                )
                current = ""
                page_start = None
                page_end = None

            if not current:
                current = text
                page_start = page_no
                page_end = page_no
            else:
                if PARA_START_PATTERNS.match(text):
                    paragraphs.append(
                        {
                            "text": current,
                            "page_start": page_start,
                            "page_end": page_end,
                        }
                    )
                    current = text
                    page_start = page_no
                    page_end = page_no
                else:
                    current += " " + text
                    page_end = page_no

            prev_y1 = y1
            prev_page_no = page_no

    if current:
        paragraphs.append(
            {
                "text": current,
                "page_start": page_start,
                "page_end": page_end,
            }
        )

    return paragraphs


def is_body_paragraph(
    paragraph: Dict,
    min_chars: int,
    max_chars: int,
) -> bool:
    """Filter paragraph chunks to retain likely body-text paragraphs."""
    text = paragraph["text"].strip()
    length = len(text)

    if length < min_chars or length > max_chars:
        return False

    if text.lower().startswith(
        (
            "keywords",
            "index terms",
            "acknowledg",
            "table",
            "figure",
            "appendix",
            "references",
        )
    ):
        return False

    if not re.search(r"[A-Za-z]", text):
        return False

    return True


def build_paragraph_chunks_for_pdf(
    pdf_path: str | Path,
    min_sentence_words: int,
    paragraph_min_chars: int,
    paragraph_max_chars: int,
    paragraph_gap_ratio: float,
) -> Tuple[List[str], List[Dict]]:
    """Build paragraph-level chunks using block-based paragraph recovery."""
    pages, metadata = read_pdf_pages(pdf_path)
    base = os.path.basename(str(pdf_path))
    title = pick_title(metadata.get("title", ""), pages, base)

    paragraphs = recover_paragraphs_by_blocks(pdf_path, gap_ratio=paragraph_gap_ratio)

    cleaned: List[Dict] = []
    for paragraph in paragraphs:
        text = clean_paragraph_by_sentences(
            paragraph["text"],
            min_sentence_words=min_sentence_words,
        )
        if text:
            cleaned.append(
                {
                    "text": text,
                    "page_start": paragraph["page_start"],
                    "page_end": paragraph["page_end"],
                }
            )

    texts: List[str] = []
    chunks: List[Dict] = []

    for idx, paragraph in enumerate(cleaned, start=1):
        if not is_body_paragraph(
            paragraph,
            min_chars=paragraph_min_chars,
            max_chars=paragraph_max_chars,
        ):
            continue

        text = paragraph["text"]
        texts.append(text)
        chunks.append(
            {
                "chunk_id": f"{base}::para::{idx}",
                "source_type": "paper",
                "file": base,
                "title": title,
                "page_start": paragraph["page_start"],
                "page_end": paragraph["page_end"],
                "text": text,
            }
        )

    return texts, chunks


# ---------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------

def get_output_filenames(chunking: str) -> Tuple[str, str]:
    """Return mode-compatible FAISS and metadata filenames."""
    if chunking == "sentence":
        return "sentence_index.faiss", "sentence_meta.json"
    if chunking == "sentence_agg":
        return "sentence_agg_index.faiss", "sentence_agg_meta.json"
    if chunking == "paragraph":
        return "paragraph_index.faiss", "paragraph_meta.json"
    raise ValueError(f"Unsupported chunking strategy: {chunking}")


def build_chunks_for_pdf(pdf_path: str | Path, args: argparse.Namespace) -> Tuple[List[str], List[Dict]]:
    """Dispatch chunk construction according to the selected chunking strategy."""
    if args.chunking == "sentence":
        return build_sentence_chunks_for_pdf(
            pdf_path,
            min_sentence_words=args.min_sentence_words,
        )

    if args.chunking == "sentence_agg":
        return build_sentence_aggregated_chunks_for_pdf(
            pdf_path,
            min_sentence_words=args.min_sentence_words,
            agg_min_words=args.agg_min_words,
            agg_max_words=args.agg_max_words,
        )

    if args.chunking == "paragraph":
        return build_paragraph_chunks_for_pdf(
            pdf_path,
            min_sentence_words=args.min_sentence_words,
            paragraph_min_chars=args.paragraph_min_chars,
            paragraph_max_chars=args.paragraph_max_chars,
            paragraph_gap_ratio=args.paragraph_gap_ratio,
        )

    raise ValueError(f"Unsupported chunking strategy: {args.chunking}")


def build_faiss_index(
    texts: Sequence[str],
    output_dir: str | Path,
    chunking: str,
    embedding_model_name: str,
    batch_size: int,
    device: str,
    metadata: Sequence[Dict],
) -> None:
    """Encode chunk texts and save FAISS index plus metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f">>> Loading embedding model: {embedding_model_name}")
    model = SentenceTransformer(embedding_model_name, device=device)

    print(">>> Encoding chunks...")
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")

    print(">>> Building FAISS index...")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    index_filename, meta_filename = get_output_filenames(chunking)

    faiss.write_index(index, str(output_dir / index_filename))

    with open(output_dir / meta_filename, "w", encoding="utf-8") as f:
        json.dump(list(metadata), f, ensure_ascii=False, indent=2)

    with open(output_dir / "embed_model_name.txt", "w", encoding="utf-8") as f:
        f.write(embedding_model_name + "\n")

    with open(output_dir / "chunking_config.json", "w", encoding="utf-8") as f:
        config = {
            "chunking": chunking,
            "embedding_model": embedding_model_name,
            "batch_size": batch_size,
            "index_file": index_filename,
            "metadata_file": meta_filename,
        }
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"[OK] FAISS index saved to: {output_dir / index_filename}")
    print(f"[OK] Metadata saved to: {output_dir / meta_filename}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an OSKR FAISS index from academic PDF files."
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing input PDF files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where FAISS index and metadata will be saved.",
    )
    parser.add_argument(
        "--chunking",
        choices=["sentence", "sentence_agg", "paragraph"],
        default="sentence",
        help="Chunking strategy.",
    )
    parser.add_argument(
        "--embedding-model",
        default="BAAI/bge-m3",
        help="SentenceTransformer embedding model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for embedding.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for embedding.",
    )
    parser.add_argument(
        "--min-sentence-words",
        type=int,
        default=10,
        help="Minimum number of words for a valid sentence.",
    )

    # Sentence-aggregated chunking parameters
    parser.add_argument(
        "--agg-min-words",
        type=int,
        default=100,
        help="Minimum words before closing a sentence-aggregated chunk.",
    )
    parser.add_argument(
        "--agg-max-words",
        type=int,
        default=130,
        help="Maximum words for a sentence-aggregated chunk.",
    )

    # Paragraph chunking parameters
    parser.add_argument(
        "--paragraph-min-chars",
        type=int,
        default=150,
        help="Minimum characters for a valid paragraph chunk.",
    )
    parser.add_argument(
        "--paragraph-max-chars",
        type=int,
        default=4000,
        help="Maximum characters for a valid paragraph chunk.",
    )
    parser.add_argument(
        "--paragraph-gap-ratio",
        type=float,
        default=0.4,
        help="Gap ratio used for block-based paragraph recovery.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.time()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"[Embedding] Using device: {device}")
    print(f"[Chunking] Strategy: {args.chunking}")

    pdf_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.pdf")))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in input directory: {args.input_dir}")

    all_texts: List[str] = []
    all_metadata: List[Dict] = []

    for pdf_path in pdf_paths:
        print(f"\n-> Processing {os.path.basename(pdf_path)}")
        texts, metadata = build_chunks_for_pdf(pdf_path, args)
        print(f"   Generated chunks: {len(texts)}")
        all_texts.extend(texts)
        all_metadata.extend(metadata)

    print(f"\nTotal chunks: {len(all_texts)}")
    if not all_texts:
        raise RuntimeError("No chunks were generated. Please check the PDF content and filtering settings.")

    build_faiss_index(
        texts=all_texts,
        output_dir=args.output_dir,
        chunking=args.chunking,
        embedding_model_name=args.embedding_model,
        batch_size=args.batch_size,
        device=device,
        metadata=all_metadata,
    )

    elapsed = time.time() - start_time
    print(f"\nDone. Elapsed time: {elapsed:.1f}s")
    print(
        "Metadata example:",
        {
            key: all_metadata[0].get(key)
            for key in ["chunk_id", "file", "page_start", "page_end"]
        },
    )


if __name__ == "__main__":
    main()
