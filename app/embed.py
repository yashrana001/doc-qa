
import numpy as np
from sentence_transformers import SentenceTransformer

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.use_prefix = "bge" in model_name.lower()

    def encode_docs(self, texts, batch_size: int = 32) -> np.ndarray:
        vecs = self.model.encode(texts, batch_size=batch_size,
                                 normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype="float32")

    def encode_query(self, query: str) -> np.ndarray:
        if self.use_prefix:
            query = BGE_QUERY_PREFIX + query
        vec = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        return np.asarray(vec, dtype="float32")
