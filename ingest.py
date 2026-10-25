"""
ingest.py
Loads documents from the data/ folder (.txt, .md, .pdf), splits them into
overlapping chunks, embeds each chunk with a free local model, and saves
a FAISS index + the chunk texts for retrieval at query time.

Run this once (and again whenever you add new documents):
    python ingest.py
"""

import os
import pickle
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
INDEX_DIR = Path("index")
INDEX_DIR.mkdir(exist_ok=True)

CHUNK_SIZE = 500       # characters per chunk (simple, no tokenizer needed)
CHUNK_OVERLAP = 100    # overlap so context isn't lost at chunk boundaries
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # free, local, no API calls


def load_text_from_file(path: Path) -> str:
    """Read raw text out of a .txt, .md, or .pdf file."""
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping character-based chunks."""
    chunks = []
    start = 0
    text = text.replace("\n", " ").strip()
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def build_index():
    print(f"Loading documents from {DATA_DIR.resolve()} ...")
    files = [
        p for p in DATA_DIR.glob("**/*")
        if p.suffix.lower() in (".txt", ".md", ".pdf")
    ]

    if not files:
        print(f"No .txt, .md, or .pdf files found in {DATA_DIR}/. "
              f"Add your notes or papers there and re-run this script.")
        return

    all_chunks = []       # chunk text
    all_sources = []      # which file each chunk came from

    for path in files:
        print(f"  Reading {path.name} ...")
        text = load_text_from_file(path)
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        all_chunks.extend(chunks)
        all_sources.extend([path.name] * len(chunks))
        print(f"    -> {len(chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Loading embedding model ({EMBED_MODEL_NAME}) ...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    print("Embedding chunks (this runs locally, no API calls)...")
    embeddings = model.encode(
        all_chunks, show_progress_bar=True, convert_to_numpy=True
    ).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    with open(INDEX_DIR / "chunks.pkl", "wb") as f:
        pickle.dump({"chunks": all_chunks, "sources": all_sources}, f)

    print(f"\nDone. Index saved to {INDEX_DIR}/")
    print(f"  - {len(all_chunks)} chunks from {len(files)} files")
    print(f"  - Embedding dimension: {dimension}")


if __name__ == "__main__":
    build_index()
