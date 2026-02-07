"""Chunk parsed passages into citeable units and emit data/chunks.jsonl."""

import json
from collections import defaultdict
from pathlib import Path

from config import (
    PLAY_CODES, DATA_DIR,
    CHUNK_MIN_LINES, CHUNK_MAX_LINES, CHUNK_OVERLAP,
)
from ingest import ingest_all


def _make_chunks_for_scene(passages: list[dict]) -> list[dict]:
    """Split a scene's passages into overlapping chunks of 6-12 lines.

    Each passage is one speech (possibly multi-line).  We accumulate
    lines across speeches until we hit the target window, then emit a
    chunk and slide forward with overlap.
    """
    if not passages:
        return []

    play = passages[0]["play"]
    work_id = passages[0]["work_id"]
    act = passages[0]["act"]
    scene = passages[0]["scene"]
    code = PLAY_CODES[work_id]

    # Flatten all speech lines, keeping per-line metadata
    flat_lines = []  # list of (text_line, speaker, line_num_or_None)
    for p in passages:
        lines = p["text"].split("\n")
        l_start = p["line_start"]
        for i, line in enumerate(lines):
            line_num = (l_start + i) if l_start else None
            flat_lines.append((line, p["speaker"], line_num))

    if not flat_lines:
        return []

    chunks = []
    seq = 0
    start = 0

    while start < len(flat_lines):
        end = min(start + CHUNK_MAX_LINES, len(flat_lines))
        window = flat_lines[start:end]

        # If remaining lines are fewer than min, just take them all
        if len(window) < CHUNK_MIN_LINES and chunks:
            # merge remainder into previous chunk
            prev = chunks[-1]
            extra_text = "\n".join(w[0] for w in window)
            prev["text"] = prev["text"] + "\n" + extra_text
            if window[-1][2] is not None:
                prev["line_end"] = window[-1][2]
            break

        # Collect speakers in this window
        speakers = list(dict.fromkeys(
            w[1] for w in window if w[1]
        ))
        # Line range
        first_num = next((w[2] for w in window if w[2] is not None), None)
        last_num = next(
            (w[2] for w in reversed(window) if w[2] is not None), None
        )

        chunk_id = f"{code}_{act}_{scene}_{seq:04d}"
        chunks.append({
            "chunk_id": chunk_id,
            "play": play,
            "act": act,
            "scene": scene,
            "speaker": speakers[0] if len(speakers) == 1 else ", ".join(speakers) if speakers else None,
            "line_start": first_num,
            "line_end": last_num,
            "text": "\n".join(w[0] for w in window),
            "source_path": f"raw/{work_id}.html",
        })

        seq += 1
        # Slide forward by (window_size - overlap)
        step = max(len(window) - CHUNK_OVERLAP, 1)
        start += step

    return chunks


def build_chunks(passages: list[dict]) -> list[dict]:
    """Group passages by (play, act, scene) and chunk each scene."""
    scenes = defaultdict(list)
    for p in passages:
        key = (p["work_id"], p["act"], p["scene"])
        scenes[key].append(p)

    all_chunks = []
    for key in sorted(scenes):
        all_chunks.extend(_make_chunks_for_scene(scenes[key]))

    return all_chunks


def save_chunks(chunks: list[dict], out_path: Path | None = None) -> Path:
    """Write chunks to JSONL file."""
    out_path = out_path or (DATA_DIR / "chunks.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"Saved {len(chunks)} chunks to {out_path}")
    return out_path


def run():
    """Full pipeline: ingest -> chunk -> save."""
    passages = ingest_all()
    chunks = build_chunks(passages)
    save_chunks(chunks)
    return chunks


if __name__ == "__main__":
    chunks = run()
    # Quick stats
    plays = set(c["play"] for c in chunks)
    print(f"\n{len(chunks)} chunks across {len(plays)} plays")
    if chunks:
        c = chunks[0]
        print(f"\nFirst chunk [{c['chunk_id']}]:")
        print(f"  {c['play']} {c['act']}.{c['scene']} — {c['speaker']}")
        print(f"  Lines {c['line_start']}–{c['line_end']}")
        print(f"  Text: {c['text'][:200]}")
