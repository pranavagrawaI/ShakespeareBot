"""Hybrid BM25 + embedding retrieval with score fusion and diversity filtering."""

import json
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    DATA_DIR,
    EMBED_MODEL,
    BM25_K,
    EMBED_K,
    TOP_K,
    BM25_WEIGHT,
    EMBED_WEIGHT,
    MAX_PER_SCENE,
)
from index import load_index, tokenize


# ── Lazy globals (loaded once on first call) ────────────────────

_bm25 = None
_corpus_tokens = None
_embeddings = None
_metas = None
_embed_model = None
_chunks = None


def _ensure_loaded():
    """Load index artifacts and embedding model on first use."""
    global _bm25, _corpus_tokens, _embeddings, _metas, _embed_model, _chunks

    if _bm25 is not None:
        return

    _bm25, _corpus_tokens, _embeddings, _metas = load_index()

    # Load the chunk texts for returning full source objects
    chunks_path = DATA_DIR / "chunks.jsonl"
    if not chunks_path.exists():
        print("ERROR: chunks.jsonl not found.", file=sys.stderr)
        sys.exit(3)
    _chunks = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            _chunks.append(json.loads(line))

    _embed_model = SentenceTransformer(EMBED_MODEL)


# ── Core retrieval ──────────────────────────────────────────────


def _bm25_search(query_tokens: list[str], k: int) -> list[tuple[int, float]]:
    """Return top-k (index, score) pairs from BM25."""
    scores = _bm25.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]


def _embedding_search(query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Return top-k (index, cosine_similarity) pairs via dot product."""
    sims = _embeddings @ query_vec  # (N,) since both are L2-normalised
    top_indices = np.argsort(sims)[::-1][:k]
    return [(int(i), float(sims[i])) for i in top_indices]


def _min_max_normalise(scores: dict[int, float], eps: float = 1e-9) -> dict[int, float]:
    """Min-max normalise a dict of {idx: score} over the values present."""
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    rng = hi - lo + eps
    return {idx: (s - lo) / rng for idx, s in scores.items()}


def _extract_phrase(query: str) -> str | None:
    """If the query contains a quoted phrase (single or double), return it."""
    import re

    m = re.search(r"""['"](.+?)['"]""", query)
    return m.group(1) if m else None


def _strip_punct(text: str) -> str:
    """Remove all punctuation (keep letters, digits, spaces) for fuzzy matching."""
    import re as _re

    return _re.sub(r"[^a-z0-9 ]", "", text.lower())


def _phrase_boost(
    candidates: set[int],
    query: str,
    chunks: list[dict],
) -> dict[int, float]:
    """Give a large score boost to chunks that contain an exact phrase match.

    Punctuation is stripped from both needle and haystack so that
    "to be or not to be" matches "To be, or not to be—".

    Returns {idx: boost} for any candidate whose text contains the phrase.
    """
    phrase = _extract_phrase(query)
    if not phrase:
        if len(query.split()) <= 10:
            phrase = query
        else:
            return {}

    needle = _strip_punct(phrase)
    if len(needle) < 6:  # too short to be meaningful
        return {}

    boosts: dict[int, float] = {}
    for idx in candidates:
        haystack = _strip_punct(chunks[idx]["text"])
        if needle in haystack:
            boosts[idx] = 1.0  # will be added on top of fused score
    return boosts


def _apply_diversity(ranked: list[dict], max_per_scene: int) -> list[dict]:
    """Cap results to at most max_per_scene chunks from the same (play, act, scene)."""
    counts: dict[tuple, int] = {}
    filtered = []
    for item in ranked:
        key = (item["meta"]["play"], item["meta"]["act"], item["meta"]["scene"])
        counts[key] = counts.get(key, 0) + 1
        if counts[key] <= max_per_scene:
            filtered.append(item)
    return filtered


# ── Public API ──────────────────────────────────────────────────


def retrieve(
    query: str,
    k: int = TOP_K,
    play_filter: str | None = None,
) -> list[dict]:
    """Hybrid retrieval: BM25 + embeddings with score fusion.

    Returns a list of source dicts ready for the answer module:
        {"sid": "S1", "chunk_id": "...", "meta": {...}, "text": "..."}
    """
    _ensure_loaded()

    # --- BM25 ---
    query_tokens = tokenize(query)
    bm25_results = _bm25_search(query_tokens, BM25_K)
    bm25_scores = {idx: score for idx, score in bm25_results}

    # --- Embedding ---
    query_vec = _embed_model.encode(query, normalize_embeddings=True)
    emb_results = _embedding_search(query_vec, EMBED_K)
    emb_scores = {idx: score for idx, score in emb_results}

    # --- Union candidates ---
    candidates = set(bm25_scores.keys()) | set(emb_scores.keys())

    # --- Optional play filter ---
    if play_filter:
        pf = play_filter.lower()
        candidates = {idx for idx in candidates if pf in _metas[idx]["play"].lower()}

    if not candidates:
        return []

    # --- Normalise over candidate set ---
    bm25_cand = {idx: bm25_scores.get(idx, 0.0) for idx in candidates}
    emb_cand = {idx: emb_scores.get(idx, 0.0) for idx in candidates}

    bm25_norm = _min_max_normalise(bm25_cand)
    emb_norm = _min_max_normalise(emb_cand)

    # --- Fuse scores ---
    fused = {
        idx: BM25_WEIGHT * bm25_norm[idx] + EMBED_WEIGHT * emb_norm[idx]
        for idx in candidates
    }

    # --- Exact-phrase boost (critical for quote lookups) ---
    # Search ALL chunks, not just candidates — a phrase match may not
    # appear in the BM25/embedding top-k when the query is all stopwords.
    all_indices = set(range(len(_chunks)))
    if play_filter:
        pf = play_filter.lower()
        all_indices = {i for i in all_indices if pf in _metas[i]["play"].lower()}
    boosts = _phrase_boost(all_indices, query, _chunks)
    for idx, boost in boosts.items():
        if idx not in fused:
            candidates.add(idx)
        fused[idx] = fused.get(idx, 0.0) + boost

    # --- Sort and build source objects ---
    ranked_indices = sorted(fused, key=fused.get, reverse=True)

    sources = []
    for idx in ranked_indices:
        meta = _metas[idx]
        sources.append(
            {
                "sid": "",  # assigned after diversity filter
                "chunk_id": meta["chunk_id"],
                "meta": meta,
                "text": _chunks[idx]["text"],
                "score": fused[idx],
            }
        )

    # --- Diversity filter ---
    sources = _apply_diversity(sources, MAX_PER_SCENE)

    # --- Trim to k and assign SIDs ---
    sources = sources[:k]
    for i, src in enumerate(sources, 1):
        src["sid"] = f"S{i}"

    return sources


# ── Quick test ──────────────────────────────────────────────────

if __name__ == "__main__":
    query = "to be or not to be"
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])

    print(f"Query: {query}\n")
    results = retrieve(query)
    for src in results:
        m = src["meta"]
        loc = f"{m['play']} {m['act']}.{m['scene']}"
        speaker = m.get("speaker") or "?"
        lines = ""
        if m.get("line_start"):
            lines = f" (lines {m['line_start']}-{m['line_end']})"
        print(f"[{src['sid']}] {loc} — {speaker}{lines}  (score: {src['score']:.3f})")
        print(f"    {src['text'][:120]}...")
        print()
