"""Step 5: find candidate chunks. Dense (meaning) search, or hybrid (meaning + keywords)."""
import re

import numpy as np
from rank_bm25 import BM25Okapi


def _tokens(text: str):
    return re.findall(r"\w+", text.lower())


class Retriever:
    def __init__(self, store, embedder):
        self.store = store
        self.embedder = embedder
        self._bm25 = None
        self._bm25_chunks = []

    def dense(self, query: str, k: int):
        """Returns list of (chunk, cosine_score)."""
        return self.store.search(self.embedder.encode_query(query), k)

    def _get_bm25(self):
        chunks = self.store.all_chunks()
        if self._bm25 is None or len(chunks) != len(self._bm25_chunks):
            self._bm25 = BM25Okapi([_tokens(c["text"]) for c in chunks])
            self._bm25_chunks = chunks
        return self._bm25, self._bm25_chunks

    def hybrid(self, query: str, k: int, rrf_k: int = 60):
        """Dense + BM25 keyword search, merged with Reciprocal Rank Fusion."""
        dense_hits = self.dense(query, k)
        bm25, chunks = self._get_bm25()
        if not chunks:
            return []
        scores = bm25.get_scores(_tokens(query))
        top_ids = np.argsort(scores)[::-1][:k]

        fused, info = {}, {}
        for rank, (chunk, score) in enumerate(dense_hits):
            cid = chunk["chunk_id"]
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
            info[cid] = (chunk, score)
        for rank, i in enumerate(top_ids):
            chunk = chunks[int(i)]
            cid = chunk["chunk_id"]
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
            info.setdefault(cid, (chunk, None))  # keyword-only hit: no dense score
        best = sorted(fused, key=fused.get, reverse=True)[:k]
        return [info[cid] for cid in best]
