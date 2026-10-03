
import json
import os

import numpy as np


class FaissStore:
    def __init__(self, dim: int, data_dir: str, index_type: str = "flat"):
        import faiss

        self.faiss = faiss
        self.dim = dim
        self.index_type = index_type
        self.dir = os.path.join(data_dir, "faiss")
        os.makedirs(self.dir, exist_ok=True)
        self.meta_path = os.path.join(self.dir, "chunks.json")
        self.emb_path = os.path.join(self.dir, "embeddings.npy")

        self.chunks = []
        self.embs = np.zeros((0, dim), dtype="float32")
        if os.path.exists(self.meta_path) and os.path.exists(self.emb_path):
            with open(self.meta_path) as f:
                self.chunks = json.load(f)
            self.embs = np.load(self.emb_path)
        self.index = self._build_index()

    def _build_index(self):
        faiss = self.faiss
        if self.index_type == "hnsw":
            index = faiss.IndexHNSWFlat(self.dim, 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efSearch = 64
        else:
            index = faiss.IndexFlatIP(self.dim) 
        if len(self.embs):
            index.add(self.embs)
        return index

    def add(self, chunks, embs):
        self.chunks.extend(chunks)
        self.embs = np.vstack([self.embs, embs]) if len(self.embs) else embs
        self.index.add(embs)
        with open(self.meta_path, "w") as f:
            json.dump(self.chunks, f)
        np.save(self.emb_path, self.embs)

    def search(self, qvec, k):
        if not self.chunks:
            return []
        scores, ids = self.index.search(qvec[None, :], min(k, len(self.chunks)))
        return [(self.chunks[i], float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    def all_chunks(self):
        return self.chunks

    def has_doc(self, doc_id):
        return any(c["doc_id"] == doc_id for c in self.chunks)


class PgVectorStore:
    def __init__(self, dim: int, database_url: str):
        import psycopg2
        from pgvector.psycopg2 import register_vector

        self.conn = psycopg2.connect(database_url)
        self.conn.autocommit = True
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        with self.conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    page INT NOT NULL,
                    text TEXT NOT NULL,
                    embedding vector({int(dim)})
                )""")
            cur.execute("CREATE INDEX IF NOT EXISTS chunks_hnsw ON chunks "
                        "USING hnsw (embedding vector_cosine_ops)")

    def add(self, chunks, embs):
        from psycopg2.extras import execute_values

        rows = [(c["chunk_id"], c["doc_id"], c["page"], c["text"], e) for c, e in zip(chunks, embs)]
        with self.conn.cursor() as cur:
            execute_values(cur, "INSERT INTO chunks (chunk_id, doc_id, page, text, embedding) "
                                "VALUES %s ON CONFLICT (chunk_id) DO NOTHING", rows)

    def search(self, qvec, k):
        with self.conn.cursor() as cur:
            cur.execute("SELECT chunk_id, doc_id, page, text, 1 - (embedding <=> %s) AS score "
                        "FROM chunks ORDER BY embedding <=> %s LIMIT %s", (qvec, qvec, k))
            rows = cur.fetchall()
        return [({"chunk_id": r[0], "doc_id": r[1], "page": r[2], "text": r[3]}, float(r[4])) for r in rows]

    def all_chunks(self):
        with self.conn.cursor() as cur:
            cur.execute("SELECT chunk_id, doc_id, page, text FROM chunks ORDER BY chunk_id")
            return [{"chunk_id": r[0], "doc_id": r[1], "page": r[2], "text": r[3]} for r in cur.fetchall()]

    def has_doc(self, doc_id):
        with self.conn.cursor() as cur:
            cur.execute("SELECT 1 FROM chunks WHERE doc_id = %s LIMIT 1", (doc_id,))
            return cur.fetchone() is not None


def make_store(dim: int, cfg):
    if cfg.vector_backend == "pgvector":
        return PgVectorStore(dim, cfg.database_url)
    return FaissStore(dim, cfg.data_dir, cfg.index_type)
