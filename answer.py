"""LLM answer synthesis with citation enforcement via OpenRouter / DeepSeek."""

import re
import sys

from openai import OpenAI

from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, LLM_MODEL


# ── System prompt (behavioural contract) ────────────────────────

SYSTEM_PROMPT = """\
You are a Shakespeare scholar assistant. You answer questions using ONLY \
the source passages provided below. Follow these rules strictly:

1. CITATION DISCIPLINE: Every factual statement you make about the text \
must include at least one inline citation like [S1], [S2], etc. referring \
to the source passages listed below.

2. GROUNDING: If the provided sources do not contain enough evidence to \
answer the question, you MUST say "This information was not found in the \
provided text." Do NOT invent lines, speakers, scenes, or plot details.

3. QUOTING POLICY: Paraphrase by default. Only quote the original text \
directly when the user explicitly asks for a quote or the exact wording.

4. FORMAT: Write 2-10 clear sentences with inline citations. Do not use \
any knowledge outside the sources provided.
"""


def _build_context(sources: list[dict]) -> str:
    """Format source chunks for the prompt."""
    parts = []
    for src in sources:
        m = src["meta"]
        header = f"[{src['sid']}] {m['play']} {m['act']}.{m['scene']}"
        if m.get("speaker"):
            header += f" — {m['speaker']}"
        if m.get("line_start"):
            header += f" (lines {m['line_start']}-{m['line_end']})"
        parts.append(f"{header}\n{src['text']}")
    return "\n\n".join(parts)


def _format_source_line(src: dict) -> str:
    """Format one source for the Sources footer."""
    m = src["meta"]
    line = f"[{src['sid']}] {m['play']} {m['act']}.{m['scene']}"
    if m.get("speaker"):
        line += f" — {m['speaker']}"
    if m.get("line_start"):
        line += f" (lines {m['line_start']}-{m['line_end']})"
    return line


def generate_answer(question: str, sources: list[dict]) -> str:
    """Call LLM to produce a grounded, cited answer.

    Returns the full formatted output (Answer + Sources block).
    """
    if not sources:
        return (
            "Answer:\n"
            "This information was not found in the provided text.\n\n"
            "Sources:\n(none)"
        )

    if not OPENROUTER_API_KEY:
        print(
            "ERROR: OPENROUTER_API_KEY environment variable not set.", file=sys.stderr
        )
        sys.exit(1)

    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        max_retries=5,
        timeout=120,
    )

    context = _build_context(sources)
    user_message = (
        f"SOURCES:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        f"Answer using only the sources above, with inline citations."
    )

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
        max_tokens=1024,
    )

    if not response.choices:
        raise RuntimeError(f"LLM returned no choices. Response: {response}")

    message = response.choices[0].message
    # R1 models may put the answer in `content` or return None with reasoning
    raw_text = message.content or ""
    raw_text = raw_text.strip()

    # Strip <think>...</think> reasoning blocks from R1 models
    answer_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()

    if not answer_text:
        answer_text = "This information was not found in the provided text."

    # Build the Sources footer
    source_lines = [_format_source_line(s) for s in sources]

    return f"Answer:\n{answer_text}\n\nSources:\n" + "\n".join(source_lines)


if __name__ == "__main__":
    # Quick standalone test with dummy sources
    dummy_sources = [
        {
            "sid": "S1",
            "chunk_id": "HAMLET_3_1_0005",
            "meta": {
                "play": "Hamlet",
                "act": 3,
                "scene": 1,
                "speaker": "HAMLET",
                "line_start": 1748,
                "line_end": 1758,
            },
            "text": (
                "To be, or not to be- that is the question:\n"
                "Whether 'tis nobler in the mind to suffer\n"
                "The slings and arrows of outrageous fortune\n"
                "Or to take arms against a sea of troubles,\n"
                "And by opposing end them."
            ),
        }
    ]
    result = generate_answer("Where does 'to be or not to be' appear?", dummy_sources)
    print(result)
