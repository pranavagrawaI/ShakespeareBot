"""Central configuration for ShakespeareBot RAG pipeline."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from project root (override any existing shell vars)
load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

# ── Directories ──────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
RAW_DIR = ROOT_DIR / "raw"
DATA_DIR = ROOT_DIR / "data"
INDEX_DIR = ROOT_DIR / "index"
EVAL_DIR = ROOT_DIR / "eval"

# ── LLM (OpenAI) ────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = "gpt-5-mini-2025-08-07"

# ── Embeddings (local, no API key needed) ────────────────────
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384

# ── Retrieval defaults ───────────────────────────────────────
BM25_K = 50  # BM25 candidate pool size
EMBED_K = 50  # embedding candidate pool size
RERANK_K = 30  # candidates sent to cross-encoder reranker
TOP_K = 8  # final results returned
BM25_WEIGHT = 0.4
EMBED_WEIGHT = 0.6
MAX_PER_SCENE = 3  # diversity cap: max chunks from same (play, act, scene)

# ── Reranker (cross-encoder) ────────────────────────────────
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Chunking defaults ────────────────────────────────────────
CHUNK_MIN_LINES = 15
CHUNK_MAX_LINES = 30
CHUNK_OVERLAP = 5

# ── Scraping ─────────────────────────────────────────────────
OSS_BASE = "https://www.opensourceshakespeare.org/views/plays"
SCRAPE_DELAY = 3  # seconds between requests (be polite)

# ── Complete play catalogue ──────────────────────────────────
#    WorkID (URL slug) -> display title
PLAYS = {
    "allswell": "All's Well That Ends Well",
    "asyoulikeit": "As You Like It",
    "comedyerrors": "Comedy of Errors",
    "loveslabours": "Love's Labour's Lost",
    "measure": "Measure for Measure",
    "merchantvenice": "Merchant of Venice",
    "merrywives": "Merry Wives of Windsor",
    "midsummer": "Midsummer Night's Dream",
    "muchado": "Much Ado about Nothing",
    "tamingshrew": "Taming of the Shrew",
    "tempest": "Tempest",
    "12night": "Twelfth Night",
    "twogents": "Two Gentlemen of Verona",
    "winterstale": "Winter's Tale",
    "henry4p1": "Henry IV, Part I",
    "henry4p2": "Henry IV, Part II",
    "henry5": "Henry V",
    "henry6p1": "Henry VI, Part I",
    "henry6p2": "Henry VI, Part II",
    "henry6p3": "Henry VI, Part III",
    "henry8": "Henry VIII",
    "kingjohn": "King John",
    "pericles": "Pericles",
    "richard2": "Richard II",
    "richard3": "Richard III",
    "antonycleo": "Antony and Cleopatra",
    "coriolanus": "Coriolanus",
    "cymbeline": "Cymbeline",
    "hamlet": "Hamlet",
    "juliuscaesar": "Julius Caesar",
    "kinglear": "King Lear",
    "macbeth": "Macbeth",
    "othello": "Othello",
    "romeojuliet": "Romeo and Juliet",
    "timonathens": "Timon of Athens",
    "titus": "Titus Andronicus",
    "troilus": "Troilus and Cressida",
}

# Stable short codes for chunk IDs (uppercase, no spaces)
PLAY_CODES = {
    "allswell": "ALLSWELL",
    "asyoulikeit": "ASYOULIKEIT",
    "comedyerrors": "COMEDYERRORS",
    "loveslabours": "LOVESLABOURS",
    "measure": "MEASURE",
    "merchantvenice": "MERCHANTVENICE",
    "merrywives": "MERRYWIVES",
    "midsummer": "MIDSUMMER",
    "muchado": "MUCHADO",
    "tamingshrew": "TAMINGSHREW",
    "tempest": "TEMPEST",
    "12night": "TWELFTHNIGHT",
    "twogents": "TWOGENTS",
    "winterstale": "WINTERSTALE",
    "henry4p1": "HENRY4P1",
    "henry4p2": "HENRY4P2",
    "henry5": "HENRY5",
    "henry6p1": "HENRY6P1",
    "henry6p2": "HENRY6P2",
    "henry6p3": "HENRY6P3",
    "henry8": "HENRY8",
    "kingjohn": "KINGJOHN",
    "pericles": "PERICLES",
    "richard2": "RICHARD2",
    "richard3": "RICHARD3",
    "antonycleo": "ANTONYCLEO",
    "coriolanus": "CORIOLANUS",
    "cymbeline": "CYMBELINE",
    "hamlet": "HAMLET",
    "juliuscaesar": "JULIUSCAESAR",
    "kinglear": "KINGLEAR",
    "macbeth": "MACBETH",
    "othello": "OTHELLO",
    "romeojuliet": "ROMEOJULIET",
    "timonathens": "TIMONATHENS",
    "titus": "TITUS",
    "troilus": "TROILUS",
}
