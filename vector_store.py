import os
import pickle
import numpy as np
import faiss

DEFAULT_DIM = 384  # output size of all-MiniLM-L6-v2


class VectorStore:
    """
    FAISS-backed vector store using an HNSW index for approximate nearest-neighbor
    search, replacing the old NumPy O(N) linear scan across all chunks.

    Cosine similarity is computed as inner product on L2-normalized vectors
    (equivalent to cosine similarity, and what FAISS's inner-product metric
    is built for).

    Persists to two files kept in sync:
      - `index_path`: the FAISS HNSW index itself (the vectors)
      - `meta_path`:  a parallel list of chunk metadata (text/source/location),
                       in the same insertion order as the index, so a FAISS
                       result index maps straight back to `self.chunks[idx]`
    """

    def __init__(self, dim=DEFAULT_DIM, index_path="vector_store.faiss",
                 meta_path="vector_store_meta.pkl"):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path
        self.chunks = []
        self.index = self._load_index()
        self._load_meta()

    def _new_index(self):
        # 32 = neighbors-per-node (M), a standard default balancing recall vs. memory.
        index = faiss.IndexHNSWFlat(self.dim, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200  # build-time search depth (higher = better index, slower build)
        index.hnsw.efSearch = 64         # query-time search depth (higher = better recall, slower query)
        return index

    def _load_index(self):
        if os.path.exists(self.index_path):
            return faiss.read_index(self.index_path)
        return self._new_index()

    def _load_meta(self):
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "rb") as f:
                self.chunks = pickle.load(f)

    def _persist(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.chunks, f)

    @staticmethod
    def _normalize(vectors):
        vectors = np.asarray(vectors, dtype="float32")
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12  # guard against a zero-vector embedding
        return vectors / norms

    def add(self, new_chunks, embeddings):
        """
        new_chunks: list of chunk dicts (text/source/source_type/location).
                    No "embedding" key needed — FAISS holds the vectors, not the chunks.
        embeddings: array-like, shape (len(new_chunks), dim) — raw embeddings
                    straight from the encoder (normalization happens here).
        """
        if not new_chunks:
            return
        self.index.add(self._normalize(embeddings))
        self.chunks.extend(new_chunks)
        self._persist()

    def search(self, query_embedding, top_k=5):
        """Returns a list of (score, chunk) tuples, highest similarity first."""
        if self.index.ntotal == 0:
            return []
        q = self._normalize(query_embedding)
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS pads with -1 if fewer than k results exist
                continue
            results.append((float(score), self.chunks[idx]))
        return results

    def remove_source(self, source):
        """
        Removes every chunk belonging to `source`. FAISS's HNSW index has no
        safe arbitrary-deletion API (the graph structure doesn't support it),
        so this rebuilds a fresh index instead — but reconstructs the kept
        vectors straight from the existing index (HNSWFlat's storage is a
        plain flat array under the hood) rather than re-running the encoder
        model on anything.
        """
        keep_indices = [i for i, c in enumerate(self.chunks) if c["source"] != source]
        if len(keep_indices) == len(self.chunks):
            return  # nothing matched that source

        kept_chunks = [self.chunks[i] for i in keep_indices]
        kept_vectors = (
            np.array([self.index.reconstruct(i) for i in keep_indices], dtype="float32")
            if keep_indices else np.empty((0, self.dim), dtype="float32")
        )

        self.index = self._new_index()
        self.chunks = kept_chunks
        if len(kept_chunks) > 0:
            self.index.add(kept_vectors)
        self._persist()

    def clear(self):
        self.index = self._new_index()
        self.chunks = []
        for path in (self.index_path, self.meta_path):
            if os.path.exists(path):
                os.remove(path)

    def __len__(self):
        return len(self.chunks)