"""Interactive terminal UI for ShakespeareBot."""

import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text
from rich.theme import Theme
from rich.live import Live
from rich.table import Table

from config import PLAYS

# ── Theme ────────────────────────────────────────────────────────
custom_theme = Theme(
    {
        "title": "bold magenta",
        "prompt": "bold cyan",
        "answer": "white",
        "source": "dim yellow",
        "cmd": "bold green",
        "err": "bold red",
        "info": "dim cyan",
    }
)
console = Console(theme=custom_theme)

# ── State ────────────────────────────────────────────────────────
play_filter: str | None = None
show_context: bool = False
top_k: int = 8


# ── Helpers ──────────────────────────────────────────────────────


HELP_TEXT = """\
[cmd]/play <name>[/cmd]  — filter results to a specific play (e.g. /play Hamlet)
[cmd]/play[/cmd]         — clear the play filter
[cmd]/context[/cmd]      — toggle showing retrieved passages before the answer
[cmd]/k <n>[/cmd]        — set number of source chunks to retrieve (default: 8)
[cmd]/clear[/cmd]        — clear the screen
[cmd]/help[/cmd]         — show this message
[cmd]/quit[/cmd]         — exit"""


def _print_banner():
    banner = Text.assemble(
        ("  ShakespeareBot\n", "bold magenta"),
        ("  Ask questions about Shakespeare's complete works.\n", "dim white"),
        ("  Type ", "dim white"),
        ("/help", "bold green"),
        (" for commands, or just ask a question.", "dim white"),
    )
    console.print(Panel(banner, border_style="magenta", expand=False))
    console.print()


def _print_status():
    parts = []
    if play_filter:
        parts.append(f"[info]play filter:[/info] [bold]{play_filter}[/bold]")
    if show_context:
        parts.append("[info]context: on[/info]")
    parts.append(f"[info]k={top_k}[/info]")
    console.print("  ".join(parts))
    console.print()


def _print_sources_table(sources: list[dict]):
    """Render sources as a compact table."""
    table = Table(
        title="Sources",
        title_style="bold yellow",
        border_style="dim",
        show_lines=False,
        pad_edge=False,
    )
    table.add_column("ID", style="bold yellow", width=4)
    table.add_column("Location", style="white")
    table.add_column("Speaker", style="cyan")
    table.add_column("Lines", style="dim")

    for src in sources:
        m = src["meta"]
        loc = f"{m['play']} {m['act']}.{m['scene']}"
        speaker = m.get("speaker") or "—"
        lines = f"{m['line_start']}–{m['line_end']}" if m.get("line_start") else "—"
        table.add_row(src["sid"], loc, speaker, lines)

    console.print(table)


def _print_context(sources: list[dict]):
    """Show the full retrieved passages."""
    console.rule("[bold]Retrieved Context[/bold]", style="dim")
    for src in sources:
        m = src["meta"]
        loc = f"{m['play']} {m['act']}.{m['scene']}"
        speaker = m.get("speaker") or "?"
        lines = f" (lines {m['line_start']}–{m['line_end']})" if m.get("line_start") else ""
        header = f"[bold yellow][{src['sid']}][/bold yellow] {loc} — [cyan]{speaker}[/cyan]{lines}"
        console.print(header)
        console.print(Text(src["text"], style="dim white"))
        console.print()
    console.rule(style="dim")
    console.print()


def _print_answer(answer_text: str):
    """Display the LLM answer (just the answer portion, not sources)."""
    console.print()
    console.print(Panel(answer_text.strip(), title="Answer", title_align="left", border_style="green", padding=(1, 2)))


# ── Loading ──────────────────────────────────────────────────────

def _load_pipeline():
    """Import and warm up retrieval (loads indexes + embedding model)."""
    with console.status("[bold cyan]Loading indexes & embedding model...[/bold cyan]", spinner="dots"):
        from retrieve import retrieve  # noqa: F811 — triggers lazy load
        retrieve("warmup", k=1)  # forces all lazy globals to initialise
    console.print("[bold green]  Ready.[/bold green]\n")
    return retrieve


# ── Command dispatch ─────────────────────────────────────────────


def _handle_command(line: str) -> bool:
    """Handle a /command. Returns True if the main loop should continue."""
    global play_filter, show_context, top_k

    parts = line.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else None

    if cmd in ("/quit", "/exit", "/q"):
        console.print("[dim]Farewell, good gentleperson.[/dim]\n")
        sys.exit(0)

    elif cmd == "/help":
        console.print(HELP_TEXT)

    elif cmd == "/clear":
        console.clear()
        _print_banner()

    elif cmd == "/context":
        show_context = not show_context
        state = "on" if show_context else "off"
        console.print(f"[info]Context display: {state}[/info]")

    elif cmd == "/play":
        if arg:
            # Validate against known plays
            match = None
            for title in PLAYS.values():
                if arg.lower() in title.lower():
                    match = title
                    break
            if match:
                play_filter = match
                console.print(f"[info]Filtering to:[/info] [bold]{play_filter}[/bold]")
            else:
                console.print(f"[err]No play matching '{arg}'. Try a partial name like 'hamlet' or 'romeo'.[/err]")
        else:
            play_filter = None
            console.print("[info]Play filter cleared — searching all plays.[/info]")

    elif cmd == "/k":
        if arg and arg.isdigit() and int(arg) > 0:
            top_k = int(arg)
            console.print(f"[info]Top-k set to {top_k}.[/info]")
        else:
            console.print("[err]Usage: /k <positive integer>[/err]")

    else:
        console.print(f"[err]Unknown command: {cmd}[/err]  Type [cmd]/help[/cmd] for available commands.")

    return True


# ── Main loop ────────────────────────────────────────────────────


def main():
    console.clear()
    _print_banner()

    retrieve_fn = _load_pipeline()
    from answer import generate_answer

    while True:
        _print_status()
        try:
            question = console.input("[prompt]You >[/prompt] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Farewell, good gentleperson.[/dim]\n")
            break

        if not question:
            continue

        # Slash commands
        if question.startswith("/"):
            _handle_command(question)
            console.print()
            continue

        # ── Retrieve ─────────────────────────────────────────
        with console.status("[bold cyan]Searching the complete works...[/bold cyan]", spinner="dots"):
            try:
                sources = retrieve_fn(question, k=top_k, play_filter=play_filter)
            except Exception as e:
                console.print(f"[err]Retrieval error: {e}[/err]")
                continue

        if show_context and sources:
            _print_context(sources)

        # ── Generate answer ──────────────────────────────────
        with console.status("[bold cyan]Composing answer...[/bold cyan]", spinner="dots"):
            try:
                raw_output = generate_answer(question, sources)
            except Exception as e:
                console.print(f"[err]Answer generation error: {e}[/err]")
                continue

        # Parse the raw output into answer + sources sections
        if "\n\nSources:\n" in raw_output:
            answer_body, _ = raw_output.split("\n\nSources:\n", 1)
            answer_body = answer_body.removeprefix("Answer:\n")
        else:
            answer_body = raw_output

        _print_answer(answer_body)

        if sources:
            _print_sources_table(sources)

        console.print()


if __name__ == "__main__":
    main()
