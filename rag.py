"""CLI entrypoint: orchestrates retrieval + answer synthesis."""

import argparse
import sys

from retrieve import retrieve
from answer import generate_answer


def main():
    parser = argparse.ArgumentParser(
        description="ShakespeareBot — ask questions about Shakespeare's plays.",
    )
    parser.add_argument(
        "question",
        help="Natural-language question about Shakespeare's works.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=8,
        help="Number of source chunks to retrieve (default: 8).",
    )
    parser.add_argument(
        "--play",
        type=str,
        default=None,
        help='Filter retrieval to a specific play (e.g. "Hamlet").',
    )
    parser.add_argument(
        "--show_context",
        action="store_true",
        help="Print the retrieved chunks before the answer.",
    )
    args = parser.parse_args()

    # ── Retrieve ────────────────────────────────────────────────
    try:
        sources = retrieve(args.question, k=args.k, play_filter=args.play)
    except SystemExit as e:
        sys.exit(e.code)

    # ── Optionally show retrieved context ───────────────────────
    if args.show_context:
        print("=" * 60)
        print("RETRIEVED CONTEXT")
        print("=" * 60)
        for src in sources:
            m = src["meta"]
            loc = f"{m['play']} {m['act']}.{m['scene']}"
            speaker = m.get("speaker") or "?"
            lines = ""
            if m.get("line_start"):
                lines = f" (lines {m['line_start']}-{m['line_end']})"
            print(f"\n[{src['sid']}] {loc} — {speaker}{lines}")
            print("-" * 40)
            print(src["text"])
        print("\n" + "=" * 60 + "\n")

    # ── Generate answer ─────────────────────────────────────────
    try:
        output = generate_answer(args.question, sources)
    except SystemExit as e:
        sys.exit(e.code)
    except Exception as e:
        print(f"ERROR generating answer: {e}", file=sys.stderr)
        sys.exit(1)

    print(output)


if __name__ == "__main__":
    main()
