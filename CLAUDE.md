# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

A documentation retrieval assistant for LangChain docs. The pipeline crawls LangChain's documentation site with Tavily, saves each page as raw text under `crawled_pages/` (gitignored), and produces embeddings via HuggingFace models for later vector search.

The project is early-stage: `main.py` is a stub, no retrieval/query layer exists yet, and no tests are wired up.

## Environment & tooling

- Python **3.14+** required (`.python-version` pins `3.14`).
- Dependencies are managed with **uv** (`uv.lock` is committed; `pyproject.toml` lists top-level deps).
- Common commands:
  - `uv sync` — install/refresh the environment from the lockfile.
  - `uv run python main.py` — run the stub entrypoint.
  - `uv run python ingest/ingest_web_docs.py` — crawl + extract docs into `crawled_pages/`.
  - `uv add <pkg>` — add a dependency (updates `pyproject.toml` and `uv.lock`).

## Required environment variables

Loaded from `.env` via `python-dotenv`:

- `CRAWL_URL` — root URL for `TavilyMap` to crawl (used by `ingest/ingest_web_docs.py`).
- `TAVILY_API_KEY` — required by `langchain-tavily`.
- `MODEL_NAME` — HuggingFace model id consumed by `EmbeddingConfig` (e.g. a BGE or E5 model).
- `DEVICE` — torch device string, defaults to `"cpu"`.

## Architecture

Two independent modules today; they are not yet wired together.

### `embedder/` — embedding abstraction

- `BaseEmbedder` (`base_embedder.py`): abstract contract with sync + async `embed_query` / `embed_documents`. Any new embedding backend must implement all four.
- `EmbeddingConfig` (`embedding_config.py`): frozen dataclass that centralizes model settings, batch size, normalization, and optional `query_instruction` / `document_instruction` prefixes. These prefixes exist because instruction-tuned retrievers (BGE, E5) need model-specific prompts prepended to queries and passages — `TextEmbedder._format_text` applies them.
- `TextEmbedder` (`text_embedder.py`): concrete `BaseEmbedder` backed by `langchain_huggingface.HuggingFaceEmbeddings`, configured from an `EmbeddingConfig`.

Note: `embedder/text_embedder.py` currently has broken imports (`from base_embedder import ...` should be a package-relative import once this module is imported from outside the folder, and `import ConversationBufferMemory` on line 8 is not a valid statement). Expect to fix these before the module can run.

### `ingest/ingest_web_docs.py` — crawler

Uses three Tavily tools in sequence:

1. `TavilyMap` discovers URLs under `CRAWL_URL` (bounded by `max_depth=5`, `max_breadth=15`, `limit=500`).
2. `TavilyExtract` fetches page content in batches of 20 (async, via `ainvoke`), tolerating per-batch failures.
3. Each result's `raw_content` is written to `crawled_pages/<sanitized-title>.txt`.

`sanitize_filename` normalizes titles or URLs into safe filenames (non-alphanumeric → `_`, capped at 200 chars). SSL is pinned to `certifi`'s CA bundle at import time.

## Conventions worth preserving

- Environment-driven config: prefer reading via `os.environ` in a dataclass factory (see `EmbeddingConfig`) rather than passing raw env vars deep into call sites.
- Async-first for I/O-bound ingestion; keep batch sizes explicit and log per-batch progress.
- Keep new embedder backends behind `BaseEmbedder` so retrieval code stays backend-agnostic.
