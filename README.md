# Hybrid RAG Demo 🔎🤖

[![CI](https://github.com/lashari/rag-demo/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/lashari/rag-demo/actions/workflows/ci.yml)

A small, clean **Retrieval-Augmented Generation** service that shows how to do RAG
*properly*: **hybrid retrieval** (dense vector search **+** BM25 keyword search)
fused with **Reciprocal Rank Fusion (RRF)**, grounded answers with **inline
citations**, exposed over a **FastAPI** API and a CLI.

It is intentionally compact and dependency-light, and it **runs fully offline with
zero setup** (deterministic hash embeddings + extractive answers) — then upgrades
to real semantic embeddings and LLM-generated answers the moment you add an
`OPENAI_API_KEY`.

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

### Enable real semantic search + generated answers

```bash
cp .env.example .env      # then set OPENAI_API_KEY=sk-...
python -m ragdemo.cli ask "How does quantization affect search latency?"
```

With a key, the pipeline uses OpenAI embeddings (`text-embedding-3-small`) and
generates fluent, cited answers with `gpt-4o-mini`. Without a key, it uses the
offline hash embedder and returns the top passages with citations.

---

## Example (offline mode)

```
$ python -m ragdemo.cli ask "Why combine dense and lexical search?"

Answer:
(extractive answer — set OPENAI_API_KEY for a generated one)

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
  generate.py   grounded answers w/ citations (OpenAI + extractive fallback)
  pipeline.py   ingest -> retrieve -> answer
  cli.py        `python -m ragdemo.cli ask "..."`
app.py          FastAPI service (/query, /health, /docs)
data/           sample corpus (RAG concepts)
tests/          offline unit tests (pytest)
```

## Tests

```bash
pytest -q
```

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
- [ ] Anthropic / Claude generator option

## License
MIT © 2026 Naeem Lashari
