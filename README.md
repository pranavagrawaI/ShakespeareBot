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

# 3. Set your OpenAI API key (or add it to .env — see .env.example)
set OPENAI_API_KEY=your-key-here        # Windows
# export OPENAI_API_KEY=your-key-here   # macOS/Linux

# 4. Download the corpus (37 plays from Open Source Shakespeare)
python scrape.py

# 5. Parse and chunk the corpus
python chunk.py

# 6. Build search indexes (BM25 + embeddings)
python index.py

# 7. Launch the interactive TUI
python tui.py
```

## Usage

### Interactive TUI (recommended)

```
python tui.py
```

Inside the TUI you can use slash commands:

- `/play <name>` — filter results to a specific play (e.g. `/play Hamlet`)
- `/play` — clear the play filter
- `/context` — toggle showing retrieved passages before the answer
- `/k <n>` — set number of source chunks to retrieve (default: 8)
- `/clear` — clear the screen
- `/help` — show all commands
- `/quit` — exit

### CLI

```
python rag.py "question" [--k 8] [--play "Hamlet"] [--show_context]
```

## Architecture

```
Scrape -> Ingest -> Chunk -> Index -> Retrieve -> Rerank -> Synthesize -> Cite
```

| Module       | Purpose                                               |
|-------------|-------------------------------------------------------|
| `scrape.py`  | Download plays from opensourceshakespeare.org          |
| `ingest.py`  | Parse HTML into structured passages                    |
| `chunk.py`   | Split into 15-30 line citeable chunks with overlap     |
| `index.py`   | Build BM25 + embedding indexes                        |
| `retrieve.py`| Hybrid retrieval with cross-encoder reranking          |
| `answer.py`  | LLM synthesis with citation enforcement (OpenAI)       |
| `tui.py`     | Interactive terminal UI                                |
| `rag.py`     | CLI entrypoint                                         |

## Evaluation

```bash
python eval/run_eval.py
```

Runs 20 queries across 4 types (quote lookup, attribution, theme, disambiguation) and checks citation presence, quote grounding, and refusal correctness.

## Tech Stack

- **Corpus**: 37 Shakespeare plays from [Open Source Shakespeare](https://www.opensourceshakespeare.org/)
- **LLM**: OpenAI gpt-5-mini
- **Reranker**: cross-encoder/ms-marco-MiniLM-L-6-v2 (local)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (local)
- **BM25**: rank-bm25
- **Retrieval**: Hybrid (BM25 + embedding) with cross-encoder reranking and diversity filter
