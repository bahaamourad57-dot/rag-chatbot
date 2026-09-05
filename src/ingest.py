"""
Ingest documents from data/docs/, split them into overlapping chunks,
and build a TF-IDF index over the chunks.

Why TF-IDF instead of a neural embedding model: it needs no API calls
or GPU, it's fully deterministic and inspectable, and for a small,
topic-focused document set like this one it retrieves well. The
retrieval step (this file + retrieve.py) is intentionally decoupled
from the generation step (chat.py) — swapping in a real embedding
model later would only mean changing this file.
"""

import json
import re
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "data" / "docs"
INDEX_DIR = ROOT / "index"
INDEX_DIR.mkdir(exist_ok=True)

CHUNK_SIZE = 500       # target characters per chunk
CHUNK_OVERLAP = 100    # characters of overlap between consecutive chunks


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """
    Split text into overlapping chunks on paragraph boundaries where
    possible, falling back to a hard character split for long
    paragraphs. Overlap helps avoid losing context at chunk edges.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 1 <= chunk_size:
            current = f"{current}\n\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            if len(para) > chunk_size:
                # hard-split an overly long paragraph
                for i in range(0, len(para), chunk_size - overlap):
                    chunks.append(para[i:i + chunk_size])
                current = ""
            else:
                current = para

    if current:
        chunks.append(current)

    # add overlap between consecutive chunks
    overlapped = []
    for i, c in enumerate(chunks):
        if i == 0:
            overlapped.append(c)
        else:
            prefix = chunks[i - 1][-overlap:]
            overlapped.append(f"{prefix}\n{c}")
    return overlapped


def load_and_chunk_docs():
    records = []  # each: {"doc_id", "source", "chunk_id", "text"}
    for path in sorted(DOCS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            records.append({
                "doc_id": f"{path.stem}::{i}",
                "source": path.name,
                "chunk_id": i,
                "text": chunk,
            })
    return records


def build_index():
    records = load_and_chunk_docs()
    if not records:
        raise RuntimeError(f"No .md documents found in {DOCS_DIR}")

    texts = [r["text"] for r in records]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)

    joblib.dump(vectorizer, INDEX_DIR / "vectorizer.joblib")
    joblib.dump(matrix, INDEX_DIR / "tfidf_matrix.joblib")
    with open(INDEX_DIR / "chunks.json", "w") as f:
        json.dump(records, f, indent=2)

    print(f"Indexed {len(records)} chunks from "
          f"{len(set(r['source'] for r in records))} documents.")
    print(f"Saved index to {INDEX_DIR}")


if __name__ == "__main__":
    build_index()
