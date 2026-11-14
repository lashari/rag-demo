# Hybrid RAG Demo 🔎🤖

[![CI](https://github.com/lashari/rag-demo/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/lashari/rag-demo/actions/workflows/ci.yml)

A small, clean **Retrieval-Augmented Generation** service that shows how to do RAG
*properly*: **hybrid retrieval** (dense vector search **+** BM25 keyword search)
fused with **Reciprocal Rank Fusion (RRF)**, grounded answers with **inline
citations**, exposed over a **FastAPI** API and a CLI.

It is intentionally compact and dependency-light, and it **runs fully offline with
zero setup** (deterministic hash embeddings + extractive answers) — then upgrades
to LLM-generated answers from **Claude** the moment you add an `ANTHROPIC_API_KEY`
(or real semantic embeddings + answers with an `OPENAI_API_KEY`).

> This mirrors the architecture of a production RAG system I build and operate that
> answers from **12M+ document vectors in ~0.07s** — distilled down to a readable,
> runnable reference. — [Naeem Lashari](https://www.linkedin.com/in/naeem-lashari)

---

## Why this is interesting (not just another RAG wrapper)

- **Hybrid retrieval + RRF** — combines semantic recall with lexical precision and
  fuses the rankings without score calibration. (See `ragdemo/retrieve.py`.)
- **Runs offline, scales up** — same code path works with no API key (hash
  embeddings) or with OpenAI embeddings + `gpt-4o-mini`. Great for reviewers.
- **Grounded + cited** — the model is instructed to answer *only* from retrieved
  context and to cite `[n]`; sources are returned with every answer.
- **Swappable components** — embedder, vector store, and generator are small,
  documented interfaces. The in-memory store has a clear path to Qdrant.
- **Tested** — pure-function unit tests for RRF and chunking run with no network.

---

## Architecture

```
          ingest                          query
 docs ─► chunk ─► embed ─► vector store      question
                              │                 │
                              ▼                 ▼
                      ┌───────────────┐   embed query ─► dense ranking ┐
                      │  in-memory /  │                                 ├─► RRF ─► top-k ─► LLM ─► cited answer
                      │    Qdrant     │   BM25 keyword ─► lexical ranking┘
                      └───────────────┘
```

---

## Quickstart

```bash
pip install -r requirements.txt

# 1) Ask a question (runs offline — no API key needed)
python -m ragdemo.cli ask "What is reciprocal rank fusion and why use it?"

# 2) Run the API
uvicorn app:app --reload
#   GET  http://127.0.0.1:8000/health
#   POST http://127.0.0.1:8000/query   {"question": "How do you scale RAG to millions of vectors?"}
#   Interactive docs: http://127.0.0.1:8000/docs
```

### Generate answers with Claude

```bash
pip install anthropic     # optional; needs Python 3.11+
cp .env.example .env      # then set ANTHROPIC_API_KEY=sk-ant-...
python -m ragdemo.cli ask "How does quantization affect search latency?"
```

With `ANTHROPIC_API_KEY` set, answers are written by Claude through the official
Anthropic SDK, grounded in the retrieved passages with inline `[n]` citations.

- **Model:** `claude-opus-5-5` by default. Set `CLAUDE_MODEL` to change it, e.g.
  `claude-sonnet-5-5` or `claude-haiku-5-5` for lower cost per question.
- **Effort:** `CLAUDE_EFFORT` (`low` | `medium` | `high` | `xhigh` | `max`),
  default `medium`. `low` is usually enough for short factual answers.
- **Refusal fallback:** on models that support it, the request opts into the
  API's server-side fallback, so if Claude declines a question the API retries it
  on a fallback model within the same call. If every model declines, the answer
  says so plainly.

Claude doesn't provide an embeddings API, so retrieval keeps the offline hash
embedder unless you also set `OPENAI_API_KEY`.

### Or use OpenAI

```bash
cp .env.example .env      # then set OPENAI_API_KEY=sk-...
```

With an OpenAI key, the pipeline uses OpenAI embeddings (`text-embedding-3-small`)
and, if no Anthropic key is set, generates answers with `gpt-4o-mini`. With both
keys, you get OpenAI embeddings and Claude answers.

Without any key, it uses the offline hash embedder and returns the top passages
with citations.

---

## Example (offline mode)

```
$ python -m ragdemo.cli ask "Why combine dense and lexical search?"

Answer:
(extractive answer — set ANTHROPIC_API_KEY or OPENAI_API_KEY for a generated one)

Dense vector search and lexical (keyword) search have complementary strengths,
so the strongest retrieval systems combine both — this is called hybrid retrieval. [1] ...

Sources:
  [1] 02_hybrid_retrieval.md (score 0.0312) — Dense vector search and lexical ...
  [2] 01_rag_basics.md       (score 0.0164) — A good RAG system lives or dies ...

(embedder + generator: hash / extractive)
```

---

## Project layout

```
ragdemo/
  chunk.py      sentence-aware chunking with overlap
  embed.py      OpenAI + offline hash embedders (pluggable)
  store.py      in-memory cosine vector store (Qdrant in production)
  retrieve.py   hybrid dense + BM25 retrieval, RRF fusion
  generate.py   grounded answers w/ citations (Claude, OpenAI + extractive fallback)
  pipeline.py   ingest -> retrieve -> answer
  cli.py        `python -m ragdemo.cli ask "..."`
app.py          FastAPI service (/query, /health, /docs)
data/           sample corpus (RAG concepts)
tests/          offline unit tests (pytest)
```

## Tests

Run from the repo root:

```bash
python -m pytest -q
```

Use `python -m pytest`, not bare `pytest`: it puts the repo root on `sys.path`
so the `ragdemo` package imports without being installed.

## CI

[GitHub Actions](.github/workflows/ci.yml) runs the test suite on every push and
pull request to `main`, on **Python 3.11 and 3.12**.

- **Offline by design** — tests run with `pytest-socket --disable-socket`, so any
  test that tries to reach the network fails. No API keys or secrets are needed.
- **Every version reports** — `fail-fast: false` keeps one Python version's
  failure from cancelling the other.
- **Superseded runs are cancelled** when you push again to the same branch or PR.
- **Read-only token** — the workflow runs with `permissions: contents: read`.
- **Coverage report** — `pytest-cov` prints line coverage for `ragdemo/` in the
  job log, adds a coverage table to the run summary, and uploads `coverage.xml`
  (from the Python 3.12 job) as a build artifact.
- **Cached pip installs**, keyed on `requirements.txt` and `requirements-dev.txt`.

To match CI locally:

```bash
pip install -r requirements.txt -r requirements-dev.txt pytest-socket pytest-cov
python -m pytest -q --disable-socket --cov=ragdemo --cov-report=term-missing
```

If a test genuinely needs the network, mark it with `@pytest.mark.enable_socket`.

## Design notes & honest scope

- The **in-memory store** is for clarity; in production I use **Qdrant** (HNSW +
  int8 quantization) — the docstrings point to where that swaps in.
- An **optional cross-encoder re-rank** step would slot in after RRF for a final
  precision boost; it's left out here to keep dependencies light.
- **Hash embeddings are lexical, not semantic** — they exist so the repo runs with
  zero setup. Use the OpenAI embedder for real semantic retrieval.

## Roadmap
- [ ] Qdrant backend toggle (`STORE=qdrant`)
- [ ] Optional cross-encoder re-ranker
- [ ] Streaming responses (SSE)
- [x] Anthropic / Claude generator option

## License
MIT © 2026 Naeem Lashari
