"""Parse scraped HTML into structured passages (play, act, scene, speaker, lines)."""

import re
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString

from config import PLAYS, RAW_DIR


def _roman_to_int(roman: str) -> int:
    """Convert a Roman numeral string to an integer."""
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
    total = 0
    for i, ch in enumerate(roman):
        if i + 1 < len(roman) and vals[ch] < vals[roman[i + 1]]:
            total -= vals[ch]
        else:
            total += vals[ch]
    return total


def _parse_scene_title(text: str):
    """Extract (act_int, scene_int) from a scene title like 'Act III, Scene 2'."""
    m = re.match(r"Act\s+([IVXLC]+),?\s+Scene\s+(\d+)", text.strip())
    if not m:
        return None
    return _roman_to_int(m.group(1)), int(m.group(2))


def _clean_text(raw: str) -> str:
    """Normalise whitespace while preserving newlines for line counting."""
    lines = raw.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if line:
            cleaned.append(line)
    return "\n".join(cleaned)


def _extract_speaker(li) -> str | None:
    """Pull the speaker name from the <strong><a>Speaker</a></strong> pattern."""
    strong = li.find("strong")
    if not strong:
        return None
    a_tag = strong.find("a")
    if a_tag:
        return a_tag.get_text(strip=True).upper()
    # fallback: just use the bold text minus trailing period
    name = strong.get_text(strip=True).rstrip(".")
    return name.upper() if name else None


def _extract_line_start(li) -> int | None:
    """Get the starting line number from the <a name='N'> anchor."""
    for a in li.find_all("a"):
        name = a.get("name", "")
        if name.isdigit():
            return int(name)
    return None


def _extract_speech_text(li) -> str:
    """Get the speech text, stripping speaker markup and line-number spans."""
    # Remove the speaker <strong> tag
    strong = li.find("strong")
    if strong:
        strong.decompose()
    # Remove line-number spans
    for span in li.find_all("span", class_="playlinenum"):
        span.decompose()
    # Remove anchor tags (line number markers) but keep surrounding text
    for a in li.find_all("a"):
        a.unwrap()
    # Convert <br> to newlines
    for br in li.find_all("br"):
        br.replace_with("\n")
    text = li.get_text()
    return _clean_text(text)


def parse_play(work_id: str, html_path: Path) -> list[dict]:
    """Parse one play's HTML into a list of passage dicts."""
    html = html_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    title = PLAYS[work_id]

    passages = []
    current_act = 0
    current_scene = 0

    # Walk through scene titles and speech items in document order
    elements = soup.find_all(["p", "li"])
    for el in elements:
        # Scene header
        if el.name == "p" and "scenetitle" in (el.get("class") or []):
            parsed = _parse_scene_title(el.get_text())
            if parsed:
                current_act, current_scene = parsed
            continue

        # Speech line
        if el.name == "li" and "playtext" in (el.get("class") or []):
            if current_act == 0:
                continue  # skip anything before first scene header

            speaker = _extract_speaker(el)
            line_start = _extract_line_start(el)
            text = _extract_speech_text(el)
            if not text.strip():
                continue

            num_lines = len(text.split("\n"))
            line_end = (line_start + num_lines - 1) if line_start else None

            passages.append({
                "play": title,
                "work_id": work_id,
                "act": current_act,
                "scene": current_scene,
                "speaker": speaker,
                "line_start": line_start,
                "line_end": line_end,
                "text": text,
            })

    return passages


def ingest_all() -> list[dict]:
    """Parse every downloaded play and return all passages."""
    all_passages = []
    for work_id in PLAYS:
        html_path = RAW_DIR / f"{work_id}.html"
        if not html_path.exists():
            print(f"  warning: {html_path} not found, skipping")
            continue
        passages = parse_play(work_id, html_path)
        print(f"  {PLAYS[work_id]}: {len(passages)} passages")
        all_passages.extend(passages)
    print(f"\nTotal passages: {len(all_passages)}")
    return all_passages


if __name__ == "__main__":
    passages = ingest_all()
    # Quick sanity check
    if passages:
        p = passages[0]
        print(f"\nFirst passage: {p['play']} {p['act']}.{p['scene']} "
              f"— {p['speaker']} (line {p['line_start']})")
        print(p["text"][:200])
