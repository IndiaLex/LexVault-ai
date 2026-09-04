import faiss
import numpy as np
import pickle
import os

INDEX_PATH = "vector_store/index.faiss"
META_PATH = "vector_store/meta.pkl"
EMBED_DIM = 384  # for all-MiniLM-L6-v2

class VectorStore:
    def __init__(self):
        os.makedirs("vector_store", exist_ok=True)
        if os.path.exists(INDEX_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            with open(META_PATH, "rb") as f:
                self.metadata = pickle.load(f)
        else:
            self.index = faiss.IndexFlatIP(EMBED_DIM)  # cosine via normalized vectors
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
            # RBAC filter: sirf allowed roles wale chunks
            if set(record["access_roles"]) & set(allowed_roles):
                results.append({**record, "score": float(score)})
            if len(results) >= top_k:
                break
        return results

    def _save(self):
        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "wb") as f:
            pickle.dump(self.metadata, f)

vector_store = VectorStore()