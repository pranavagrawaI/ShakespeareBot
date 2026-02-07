"""Build BM25 + embedding indexes from chunks.jsonl.

Produces:
    index/bm25.pkl       -- BM25Okapi object over tokenized corpus
    index/embeddings.npy  -- float32 (N, 384) L2-normalised embeddings
    index/meta.pkl        -- list of chunk metadata dicts aligned with rows
"""

import json
import pickle
import re
import sys

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import DATA_DIR, INDEX_DIR, EMBED_MODEL


# ── Tokenisation (BM25) ────────────────────────────────────────


def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation (keep apostrophes), split on whitespace."""
    text = text.lower()
    # Keep letters, digits, apostrophes; replace everything else with space
    text = re.sub(r"[^a-z0-9']", " ", text)
    return text.split()


# ── Loading ─────────────────────────────────────────────────────


def load_chunks(path=None):
    """Read chunks.jsonl and return (texts, metadata_list)."""
    path = path or (DATA_DIR / "chunks.jsonl")
    if not path.exists():
        print(f"ERROR: {path} not found. Run chunk.py first.", file=sys.stderr)
        sys.exit(3)

    texts = []
    metas = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            texts.append(chunk["text"])
            metas.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "play": chunk["play"],
                    "act": chunk["act"],
                    "scene": chunk["scene"],
                    "speaker": chunk.get("speaker"),
                    "line_start": chunk.get("line_start"),
                    "line_end": chunk.get("line_end"),
                }
            )
    return texts, metas


# ── BM25 Index ──────────────────────────────────────────────────


def build_bm25(texts: list[str]):
    """Tokenize corpus and build BM25Okapi index."""
    print(f"  Tokenising {len(texts)} chunks for BM25...")
    corpus_tokens = [tokenize(t) for t in texts]
    print("  Building BM25 index...")
    bm25 = BM25Okapi(corpus_tokens)
    return bm25, corpus_tokens


# ── Embedding Index ─────────────────────────────────────────────


def build_embeddings(texts: list[str]) -> np.ndarray:
    """Encode all chunk texts and L2-normalise."""
    print(f"  Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    print(f"  Encoding {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=128,
        normalize_embeddings=True,  # L2-norm so dot product = cosine sim
    )
    return embeddings.astype(np.float32)


# ── Save / Load helpers ────────────────────────────────────────


def save_index(bm25, corpus_tokens, embeddings, metas):
    """Persist all index artifacts to INDEX_DIR."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    bm25_path = INDEX_DIR / "bm25.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump({"bm25": bm25, "corpus_tokens": corpus_tokens}, f)
    print(f"  Saved BM25 index -> {bm25_path}")

    emb_path = INDEX_DIR / "embeddings.npy"
    np.save(emb_path, embeddings)
    print(f"  Saved embeddings ({embeddings.shape}) -> {emb_path}")

    meta_path = INDEX_DIR / "meta.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump(metas, f)
    print(f"  Saved metadata ({len(metas)} entries) -> {meta_path}")


def load_index():
    """Load all index artifacts. Returns (bm25, corpus_tokens, embeddings, metas).

    Raises SystemExit(2) if index files are missing.
    """
    bm25_path = INDEX_DIR / "bm25.pkl"
    emb_path = INDEX_DIR / "embeddings.npy"
    meta_path = INDEX_DIR / "meta.pkl"

    for p in [bm25_path, emb_path, meta_path]:
        if not p.exists():
            print(
                f"ERROR: Index file {p} not found. Run: python index.py",
                file=sys.stderr,
            )
            sys.exit(2)

    with open(bm25_path, "rb") as f:
        data = pickle.load(f)
    bm25 = data["bm25"]
    corpus_tokens = data["corpus_tokens"]

    embeddings = np.load(emb_path)

    with open(meta_path, "rb") as f:
        metas = pickle.load(f)

    return bm25, corpus_tokens, embeddings, metas


# ── Main ────────────────────────────────────────────────────────


def build():
    """Full index build pipeline."""
    print("Loading chunks...")
    texts, metas = load_chunks()
    print(f"  {len(texts)} chunks loaded.\n")

    print("Building BM25 index...")
    bm25, corpus_tokens = build_bm25(texts)

    print("\nBuilding embedding index...")
    embeddings = build_embeddings(texts)

    print("\nSaving indexes...")
    save_index(bm25, corpus_tokens, embeddings, metas)

    print(f"\nDone. Index artifacts in {INDEX_DIR}/")
    return bm25, corpus_tokens, embeddings, metas


if __name__ == "__main__":
    build()
