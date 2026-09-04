from app.services.embedding_service import embed_texts
from app.services.vector_store import vector_store
from app.config import LOCAL_LLM_URL, LOCAL_LLM_MODEL, LOCAL_LLM_TOP_K
import requests


def answer_query(query: str, user_roles: list[str], top_k: int = LOCAL_LLM_TOP_K) -> dict:
    query_vec = embed_texts([query])
    hits = vector_store.search(query_vec, allowed_roles=user_roles, top_k=top_k)

    if not hits:
        return {
            "answer": "Is query ka answer accessible documents me nahi mila.",
            "sources": []
        }

    context_block = "\n\n".join(
        f"[{i+1}] {h['chunk_text']}" for i, h in enumerate(hits)
    )

    prompt = f"""You are a legal document assistant. Answer ONLY using the context below.
Cite every claim using [n] matching the source number. If the answer isn't in the context, say so.

Context:
{context_block}

Question: {query}

Answer (with [n] citations):"""

    resp = requests.post(LOCAL_LLM_URL, json={
        "model": LOCAL_LLM_MODEL,
        "prompt": prompt,
        "stream": False
    })
    answer_text = resp.json().get("response", "").strip()

    sources = [
        {
            "citation_id": i + 1,
            "document_id": h["document_id"],
            "storage_key": h["storage_key"],
            "page": h["page"],
            "doc_class": h["doc_class"],
            "excerpt": h["chunk_text"][:200] + "...",
            "confidence": round(h["score"], 3)
        }
        for i, h in enumerate(hits)
    ]

    return {"answer": answer_text, "sources": sources}