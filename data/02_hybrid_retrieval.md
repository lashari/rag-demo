# Hybrid Retrieval and Reciprocal Rank Fusion

Dense vector search and lexical (keyword) search have complementary strengths, so
the strongest retrieval systems combine both — this is called hybrid retrieval.

## Dense vector search

Embeddings map text into a vector space where semantically similar text is close
together. Dense search excels at meaning: a query about "automobiles" can match a
passage about "cars" even with no shared words. Its weakness is exact terms — part
numbers, error codes, names, and rare jargon can get lost.

## Lexical search (BM25)

BM25 is a classic ranking function over exact token matches. It excels at
precision on specific terms and is strong when the query and document share rare
words. Its weakness is that it has no sense of meaning — synonyms and paraphrases
are invisible to it.

## Reciprocal Rank Fusion (RRF)

Dense scores (cosine similarity) and BM25 scores live on different, incomparable
scales, so you cannot simply add them. Reciprocal Rank Fusion sidesteps this by
fusing the two *rankings* rather than their scores. For each result it sums
1 / (k + rank) across the ranked lists it appears in, where k is a small constant
(commonly 60). Documents that rank highly in either list — and especially in both
— rise to the top. RRF is simple, robust, and requires no score calibration.

A common pattern is: take the top N candidates from each retriever, fuse with RRF,
then optionally re-rank the merged top candidates with a cross-encoder for a final
precision boost before sending them to the LLM.
