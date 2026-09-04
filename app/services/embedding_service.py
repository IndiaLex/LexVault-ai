from sentence_transformers import SentenceTransformer
import numpy as np
from app.config import EMBEDDING_MODEL

_model = SentenceTransformer(EMBEDDING_MODEL)

def embed_texts(texts: list[str]) -> np.ndarray:
    return np.array(_model.encode(texts, convert_to_numpy=True), dtype="float32")
