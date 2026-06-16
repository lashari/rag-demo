# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) is a technique that grounds a large language
model's answers in an external knowledge source instead of relying only on the
model's parametric memory. At query time the system retrieves the most relevant
passages from a corpus and passes them to the model as context, so answers stay
accurate, up to date, and traceable to a source.

## Why RAG

- Reduces hallucination: the model answers from supplied passages, not guesses.
- Keeps knowledge current without retraining the model.
- Enables citations: every answer can point back to the source passage.
- Cheaper than fine-tuning for most factual question-answering use cases.

## The RAG pipeline

1. Ingestion: documents are split into overlapping chunks.
2. Embedding: each chunk is converted into a vector.
3. Indexing: vectors are stored in a vector database for fast similarity search.
4. Retrieval: the user query is embedded and the nearest chunks are fetched.
5. Generation: the chunks plus the question are sent to the LLM, which writes a
   grounded answer with citations.

A good RAG system lives or dies on retrieval quality: if the right passage is not
retrieved, the model cannot answer correctly no matter how capable it is.
