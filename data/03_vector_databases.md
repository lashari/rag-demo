# Vector Databases, Indexing, and Scaling RAG

A vector database stores embeddings and answers nearest-neighbour queries quickly.
Examples include Qdrant, FAISS, Milvus, and pgvector.

## Approximate nearest neighbour (HNSW)

Exact nearest-neighbour search is O(n) per query and does not scale to millions of
vectors. Approximate methods like HNSW (Hierarchical Navigable Small World) build a
navigable graph that returns near-exact results in roughly logarithmic time, which
is what makes large-scale semantic search practical.

## Quantization and memory

Embeddings are large: one million 1024-dimension float32 vectors is about 4 GB.
Scalar quantization (for example int8) compresses vectors roughly 4x with minimal
recall loss, letting far more vectors stay resident in RAM. Keeping the index and
quantized vectors in memory — rather than reading them from disk per query — is
often the single biggest lever for search latency.

## Latency tips

- Keep the HNSW index and quantized vectors in RAM; avoid swapping.
- Tune ef_search to trade a little recall for a lot of speed.
- Cache embeddings for repeated or templated queries.
- Retrieve a small candidate set, then re-rank — do expensive work on few items.

With these techniques a corpus of well over ten million vectors can be searched in
a fraction of a second on modest hardware.
