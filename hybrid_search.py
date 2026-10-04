import re
from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text):
    return _TOKEN_RE.findall(text.lower())


def _chunk_key(chunk):
    # Chunk dicts aren't hashable, so key on the fields that uniquely identify
    # one (source + location + text) to dedupe the same chunk across both
    # ranked lists during fusion.
    return (chunk["source"], chunk["location"], chunk["text"])


class HybridSearch:
    """
    Combines FAISS dense vector search with BM25 sparse keyword search via
    Reciprocal Rank Fusion (RRF). Dense embeddings are great at "meaning" but
    can miss exact acronyms, product IDs, or proper nouns; BM25 catches those.
    RRF fuses the two ranked lists without needing their raw scores to be on
    comparable scales (cosine similarity and BM25 scores aren't).

    BM25 is rebuilt from the vector store's current chunk list on demand
    rather than persisted separately. At personal-notes scale (hundreds to
    low thousands of chunks) this rebuild is fast, and it keeps this module
    always in sync with the vector store with nothing extra to persist or
    fall out of sync.
    """

    def __init__(self, vector_store):
        self.vector_store = vector_store
        self._bm25 = None
        self._bm25_fingerprint = None  # rebuild trigger: the corpus text changed

    def _ensure_bm25(self):
        chunks = self.vector_store.chunks
        # Size alone isn't enough: removing one source and adding another with
        # the same chunk count would leave a stale index pointing at the wrong
        # chunks. Hashing the chunk texts catches any change (string hashes are
        # cached, so this stays cheap per query).
        fingerprint = (len(chunks), hash(tuple(c["text"] for c in chunks)))
        if self._bm25 is None or self._bm25_fingerprint != fingerprint:
            tokenized_corpus = [_tokenize(c["text"]) for c in chunks]
            self._bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
            self._bm25_fingerprint = fingerprint

    def search(self, query_text, query_embedding, top_k=5, candidate_k=20, rrf_k=60):
        """
        query_text: raw query string, for BM25 keyword matching.
        query_embedding: encoded query vector, for FAISS dense matching.
        candidate_k: how many candidates EACH method contributes before fusion
                     (wider than top_k so RRF actually has room to re-rank).
        rrf_k: RRF's damping constant — 60 is the standard value from the
               original RRF paper; higher flattens the weight given to rank
               position, lower makes top ranks matter more.

        Returns a list of (fused_score, chunk) tuples, highest first.
        """
        chunks = self.vector_store.chunks
        if not chunks:
            return []

        self._ensure_bm25()

        dense_results = self.vector_store.search(query_embedding, top_k=candidate_k)

        bm25_scores = self._bm25.get_scores(_tokenize(query_text)) if self._bm25 else []
        # Only chunks that share at least one word with the query (score > 0).
        # Zero-score chunks are noise and shouldn't earn any fusion credit.
        sparse_ranked_indices = sorted(
            (i for i in range(len(bm25_scores)) if bm25_scores[i] > 0),
            key=lambda i: bm25_scores[i], reverse=True,
        )[:candidate_k]

        rrf_scores = {}
        chunk_by_key = {}

        for rank, (_score, chunk) in enumerate(dense_results):
            key = _chunk_key(chunk)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            chunk_by_key[key] = chunk

        for rank, idx in enumerate(sparse_ranked_indices):
            chunk = chunks[idx]
            key = _chunk_key(chunk)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            chunk_by_key[key] = chunk

        fused = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [(score, chunk_by_key[key]) for key, score in fused]