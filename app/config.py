"""All settings in one place. Override any of them with environment variables."""
import os
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass
class Settings:
    # Models (all from Hugging Face)
    embed_model: str = _env("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    rerank_model: str = _env("RERANK_MODEL", "BAAI/bge-reranker-base")
    llm_backend: str = _env("LLM_BACKEND", "local")  # "local" or "extractive" (no LLM, for quick tests)
    llm_model: str = _env("LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

    # Vector store
    vector_backend: str = _env("VECTOR_BACKEND", "faiss")  # "faiss" or "pgvector"
    index_type: str = _env("INDEX_TYPE", "flat")  # FAISS only: "flat" or "hnsw"
    data_dir: str = _env("DATA_DIR", "data")
    database_url: str = _env("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/docqa")

    # Chunking (measured in words, to keep things simple)
    chunk_words: int = int(_env("CHUNK_WORDS", "200"))
    chunk_overlap: int = int(_env("CHUNK_OVERLAP", "40"))

    # Retrieval
    retrieve_k: int = int(_env("RETRIEVE_K", "20"))  # candidates fetched from the vector DB
    final_k: int = int(_env("FINAL_K", "4"))  # chunks given to the LLM

    # "Don't make things up" thresholds. TUNE THESE on your eval set.
    rerank_threshold: float = float(_env("RERANK_THRESHOLD", "0.1"))  # used when reranking is on
    dense_threshold: float = float(_env("DENSE_THRESHOLD", "0.45"))  # used when reranking is off


settings = Settings()
