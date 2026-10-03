"""Glue: ingest PDFs and answer questions end to end."""
import os
import time

from .config import settings
from .embed import Embedder
from .generate import REFUSAL, Generator, extract_citations
from .ingest import chunk_pages, extract_pdf
from .rerank import Reranker
from .retrieve import Retriever
from .store import make_store


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000


class DocQA:
    def __init__(self, cfg=settings):
        self.cfg = cfg
        self.embedder = Embedder(cfg.embed_model)
        self.store = make_store(self.embedder.dim, cfg)
        self.retriever = Retriever(self.store, self.embedder)
        self._reranker = None
        self._generator = None

    # models that are loaded only when first needed
    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = Reranker(self.cfg.rerank_model)
        return self._reranker

    @property
    def generator(self):
        if self._generator is None:
            self._generator = Generator(self.cfg.llm_backend, self.cfg.llm_model)
        return self._generator

    # ---------- ingestion ----------
    def ingest_pdf(self, path: str, doc_id: str = None):
        doc_id = doc_id or os.path.basename(path)
        if self.store.has_doc(doc_id):
            return {"doc_id": doc_id, "chunks_added": 0, "skipped": True}
        pages = extract_pdf(path)
        chunks = chunk_pages(doc_id, pages, self.cfg.chunk_words, self.cfg.chunk_overlap)
        if chunks:
            embs = self.embedder.encode_docs([c["text"] for c in chunks])
            self.store.add(chunks, embs)
        return {"doc_id": doc_id, "pages": len(pages), "chunks_added": len(chunks), "skipped": False}

    # ---------- retrieval ----------
    def retrieve(self, question: str, retrieve_k: int, use_rerank: bool = True, hybrid: bool = False):
        """Returns (ranked_chunks, timings). Each chunk has dense_score and rerank_score."""
        timings = {}
        t0 = time.perf_counter()
        cands = (self.retriever.hybrid if hybrid else self.retriever.dense)(question, retrieve_k)
        timings["retrieve_ms"] = _ms(t0)

        if use_rerank and cands:
            t0 = time.perf_counter()
            scores = self.reranker.score(question, [c["text"] for c, _ in cands])
            timings["rerank_ms"] = _ms(t0)
            order = sorted(range(len(cands)), key=lambda i: scores[i], reverse=True)
            ranked = [{**cands[i][0], "dense_score": cands[i][1], "rerank_score": scores[i]} for i in order]
        else:
            ranked = [{**c, "dense_score": d, "rerank_score": None} for c, d in cands]
        return ranked, timings

    def _confidence(self, ranked, use_rerank):
        if not ranked:
            return 0.0
        if use_rerank:
            return ranked[0]["rerank_score"]
        return max((c["dense_score"] or 0.0) for c in ranked)

    def _is_supported(self, ranked, use_rerank):
        threshold = self.cfg.rerank_threshold if use_rerank else self.cfg.dense_threshold
        return self._confidence(ranked, use_rerank) >= threshold

    # ---------- full question answering ----------
    def query(self, question, use_rerank=True, hybrid=False, retrieve_k=None, final_k=None):
        retrieve_k = retrieve_k or self.cfg.retrieve_k
        final_k = final_k or self.cfg.final_k
        ranked, timings = self.retrieve(question, retrieve_k, use_rerank, hybrid)
        confidence = self._confidence(ranked, use_rerank)

        # Guard 1: the retrieved evidence is too weak -> refuse instead of guessing
        if not ranked or not self._is_supported(ranked, use_rerank):
            timings["total_ms"] = sum(timings.values())
            return {"answer": REFUSAL, "supported": False, "confidence": confidence,
                    "sources": [], "invalid_citations": [], "timings": timings}

        context = ranked[:final_k]
        t0 = time.perf_counter()
        answer = self.generator.answer(question, context)
        timings["generate_ms"] = _ms(t0)
        timings["total_ms"] = sum(timings.values())

        # Guard 2: the model itself said it can't answer
        if REFUSAL.rstrip(".").lower() in answer.lower():
            return {"answer": REFUSAL, "supported": False, "confidence": confidence,
                    "sources": [], "invalid_citations": [], "timings": timings}

        # Guard 3: keep only citations that point to chunks we really provided
        cited, invalid = extract_citations(answer, len(context))
        used = [context[n - 1] for n in cited] if cited else context
        sources = [{
            "doc_id": c["doc_id"], "page": c["page"], "chunk_id": c["chunk_id"],
            "rerank_score": c["rerank_score"], "dense_score": c["dense_score"],
            "snippet": c["text"][:300],
        } for c in used]
        return {"answer": answer, "supported": True, "confidence": confidence,
                "sources": sources, "invalid_citations": invalid, "timings": timings}
