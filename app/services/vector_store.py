import faiss
import numpy as np
import pickle
import os
from app.config import VECTOR_INDEX_PATH, VECTOR_META_PATH, VECTOR_EMBED_DIM


class VectorStore:
    def __init__(self):
        os.makedirs(os.path.dirname(VECTOR_INDEX_PATH) or ".", exist_ok=True)
        if os.path.exists(VECTOR_INDEX_PATH):
            self.index = faiss.read_index(VECTOR_INDEX_PATH)
            with open(VECTOR_META_PATH, "rb") as f:
                self.metadata = pickle.load(f)
        else:
            self.index = faiss.IndexFlatIP(VECTOR_EMBED_DIM)  # cosine via normalized vectors
            self.metadata = []  # list of dicts, index-aligned with faiss vectors

    def add(self, embeddings: np.ndarray, records: list[dict]):
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.metadata.extend(records)
        self._save()

    def search(self, query_vec: np.ndarray, allowed_roles: list[str], top_k: int = 5):
        faiss.normalize_L2(query_vec)
        scores, idxs = self.index.search(query_vec, min(top_k * 4, self.index.ntotal or 1))
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            record = self.metadata[idx]
            # RBAC filter: only chunks with allowed roles
            if set(record["access_roles"]) & set(allowed_roles):
                results.append({**record, "score": float(score)})
            if len(results) >= top_k:
                break
        return results

    def _save(self):
        faiss.write_index(self.index, VECTOR_INDEX_PATH)
        with open(VECTOR_META_PATH, "wb") as f:
            pickle.dump(self.metadata, f)

vector_store = VectorStore()
