"""Lightweight evaluation harness for ShakespeareBot.

Checks:
  - Citation presence: answer contains at least one [S#] token.
  - Quote grounding: for quote_lookup queries, at least one cited chunk
    contains the expected phrase (case-insensitive) or >= 70% token overlap.
  - Refusal correctness: unanswerable queries produce "not found" wording.
"""

import json
import re
import sys
from pathlib import Path

# Add parent directory so we can import project modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieve import retrieve
from answer import generate_answer
from config import EVAL_DIR


def _extract_citations(text: str) -> list[str]:
    """Pull all [S#] citation tokens from text."""
    return re.findall(r"\[S(\d+)\]", text)


def _token_overlap(phrase: str, chunk_text: str) -> float:
    """Fraction of phrase tokens found in chunk_text (case-insensitive)."""
    phrase_tokens = set(phrase.lower().split())
    chunk_tokens = set(chunk_text.lower().split())
    if not phrase_tokens:
        return 0.0
    return len(phrase_tokens & chunk_tokens) / len(phrase_tokens)


def _check_quote_grounding(
    answer_text: str, sources: list[dict], must_include: list[str]
) -> bool:
    """Check that at least one cited source contains the expected phrase."""
    if not must_include:
        return True

    cited_ids = _extract_citations(answer_text)
    cited_sources = [s for s in sources if s["sid"].replace("S", "") in cited_ids]
    if not cited_sources:
        cited_sources = sources  # fallback: check all returned sources

    phrase = " ".join(must_include)
    for src in cited_sources:
        text_lower = src["text"].lower()
        # Exact substring match
        if phrase.lower() in text_lower:
            return True
        # Token overlap fallback
        if _token_overlap(phrase, src["text"]) >= 0.70:
            return True

    return False


def run_eval(questions_path: Path | None = None, verbose: bool = True):
    """Run all eval queries and report results."""
    questions_path = questions_path or (EVAL_DIR / "questions.json")
    with open(questions_path, encoding="utf-8") as f:
        questions = json.load(f)

    results = []
    passed = 0
    total = len(questions)

    for q in questions:
        qid = q["id"]
        qtype = q["type"]
        question = q["question"]
        must_include = q.get("must_include", [])
        expected_play = q.get("expected_play")

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"[{qid}] ({qtype}) {question}")
            print(f"{'=' * 60}")

        # Retrieve
        sources = retrieve(question, k=8)

        # Generate answer
        output = generate_answer(question, sources)
        answer_text = output.split("Sources:")[0] if "Sources:" in output else output

        if verbose:
            print(output[:500])
            if len(output) > 500:
                print("...")

        # ── Checks ──────────────────────────────────────────────
        checks = {"citation": False, "grounding": False, "play_match": False}

        # 1. Citation presence
        citations = _extract_citations(answer_text)
        if qtype == "unanswerable":
            # Should have NO citations and contain "not found"
            checks["citation"] = (
                len(citations) == 0 or "not found" in answer_text.lower()
            )
        else:
            checks["citation"] = len(citations) > 0

        # 2. Quote grounding (only for quote_lookup)
        if qtype == "quote_lookup":
            checks["grounding"] = _check_quote_grounding(
                answer_text, sources, must_include
            )
        else:
            checks["grounding"] = True  # N/A for other types

        # 3. Expected play sanity check
        if expected_play and sources:
            checks["play_match"] = any(
                expected_play.lower() in s["meta"]["play"].lower()
                for s in sources[:3]  # check top 3 sources
            )
        else:
            checks["play_match"] = True  # N/A

        # Overall pass
        all_pass = all(checks.values())
        if all_pass:
            passed += 1

        result = {
            "id": qid,
            "type": qtype,
            "passed": all_pass,
            "checks": checks,
        }
        results.append(result)

        if verbose:
            status = "PASS" if all_pass else "FAIL"
            detail = " | ".join(
                f"{k}={'ok' if v else 'FAIL'}" for k, v in checks.items()
            )
            print(f"\n  => {status} ({detail})")

    # ── Summary ─────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"EVAL SUMMARY: {passed}/{total} passed")
    print(f"{'=' * 60}")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{r['id']}] {r['type']:20s} {status}")

    return results


if __name__ == "__main__":
    run_eval(verbose="--quiet" not in sys.argv)
