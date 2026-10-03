
import os
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass
class Settings:
    
    embed_model: str = _env("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    rerank_model: str = _env("RERANK_MODEL", "BAAI/bge-reranker-base")
    llm_backend: str = _env("LLM_BACKEND", "local")  
    llm_model: str = _env("LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

   
    vector_backend: str = _env("VECTOR_BACKEND", "faiss")  
    index_type: str = _env("INDEX_TYPE", "flat")  
    data_dir: str = _env("DATA_DIR", "data")
    database_url: str = _env("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/docqa")

    
    chunk_words: int = int(_env("CHUNK_WORDS", "200"))
    chunk_overlap: int = int(_env("CHUNK_OVERLAP", "40"))

    
    retrieve_k: int = int(_env("RETRIEVE_K", "20"))  
    final_k: int = int(_env("FINAL_K", "4"))  

    
    rerank_threshold: float = float(_env("RERANK_THRESHOLD", "0.1"))  
    dense_threshold: float = float(_env("DENSE_THRESHOLD", "0.45"))  


settings = Settings()
