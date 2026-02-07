# ShakespeareBot

A cite-first RAG system over Shakespeare's complete works. Ask natural-language questions and get answers grounded in the text with inline citations to play/act/scene.

## Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your OpenRouter API key
set OPENROUTER_API_KEY=your-key-here        # Windows
# export OPENROUTER_API_KEY=your-key-here   # macOS/Linux

# 4. Download the corpus (37 plays from Open Source Shakespeare)
python scrape.py

# 5. Parse and chunk the corpus
python chunk.py

# 6. Build search indexes (BM25 + embeddings)
python index.py

# 7. Ask a question
python rag.py "Where does 'to be or not to be' appear?"
```

## Usage

```
python rag.py "question" [--k 8] [--play "Hamlet"] [--show_context]
```

- `--k N` — number of source chunks to retrieve (default: 8)
- `--play "Name"` — filter retrieval to a specific play
- `--show_context` — print the retrieved passages before the answer

## Architecture

```
Ingest -> Chunk -> Index -> Retrieve -> Synthesize -> Cite
```

| Module       | Purpose                                      |
|-------------|----------------------------------------------|
| `scrape.py`  | Download plays from opensourceshakespeare.org |
| `ingest.py`  | Parse HTML into structured passages           |
| `chunk.py`   | Split into 6-12 line citeable chunks          |
| `index.py`   | Build BM25 + embedding indexes                |
| `retrieve.py`| Hybrid retrieval with score fusion            |
| `answer.py`  | LLM synthesis with citation enforcement       |
| `rag.py`     | CLI entrypoint                                |

## Evaluation

```bash
python eval/run_eval.py
```

Runs 20 queries across 4 types (quote lookup, attribution, theme, disambiguation) and checks citation presence, quote grounding, and refusal correctness.

## Tech Stack

- **Corpus**: 37 Shakespeare plays from [Open Source Shakespeare](https://www.opensourceshakespeare.org/)
- **LLM**: DeepSeek R1 (free) via OpenRouter
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (local)
- **BM25**: rank-bm25
- **Retrieval**: Hybrid (0.4 BM25 + 0.6 embedding) with diversity filter
