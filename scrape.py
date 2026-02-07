"""Download all 37 Shakespeare plays from Open Source Shakespeare as raw HTML."""

import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import PLAYS, OSS_BASE, SCRAPE_DELAY, RAW_DIR

# Polite session with retries and a real User-Agent
_session = requests.Session()
_session.headers.update(
    {"User-Agent": "ShakespeareBot/1.0 (educational project; public-domain texts)"}
)
_retry = Retry(total=4, backoff_factor=3, status_forcelist=[429, 500, 502, 503, 504])
_session.mount("https://", HTTPAdapter(max_retries=_retry))


def scrape_play(work_id: str, out_dir: Path) -> Path:
    """Fetch the full text of one play and save as HTML."""
    out_path = out_dir / f"{work_id}.html"
    if out_path.exists():
        print(f"  skip {work_id} (already downloaded)")
        return out_path

    url = f"{OSS_BASE}/play_view.php?WorkID={work_id}&Scope=entire&pleasewait=1&msg=pl"
    resp = _session.get(url, timeout=60)
    resp.raise_for_status()
    out_path.write_text(resp.text, encoding="utf-8")
    print(f"  saved {work_id} ({len(resp.text):,} bytes)")
    return out_path


def scrape_all():
    """Download every play in the catalogue."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    total = len(PLAYS)
    for i, (work_id, title) in enumerate(PLAYS.items(), 1):
        print(f"[{i}/{total}] {title}")
        try:
            scrape_play(work_id, RAW_DIR)
        except requests.RequestException as e:
            print(f"  ERROR on {work_id}: {e}")
            print("  waiting 10s before continuing...")
            time.sleep(10)
            continue
        if i < total:
            time.sleep(SCRAPE_DELAY)
    print(f"\nDone. {total} plays in {RAW_DIR}/")


if __name__ == "__main__":
    scrape_all()
