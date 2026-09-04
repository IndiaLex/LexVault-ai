# RAG Chatbot Module — LexVault Backend AI Service

This module adds a **Retrieval-Augmented Generation (RAG) chatbot** on top of the
existing OCR + NER + PII pipeline. It lets an officer ask natural-language
questions about processed documents and get an answer grounded in the actual
document text — with **clickable source citations** pointing back to the exact
report the answer came from.

This is an extension of the existing `Backend AI Service`. It reuses the OCR
output already produced by `/ai/process` and does not require re-scanning any
document.

---

## Why this exists

Every answer the chatbot gives is backed by retrieved chunks of real OCR text
from indexed documents — not the model's general knowledge. Each chunk is
tagged with its source document, storage key, and page, so the response can
show exactly where the information came from and let the officer open that
report directly.

Retrieval is also **RBAC-aware**: a query only searches chunks the requesting
user's roles are allowed to access, so an officer on Case A cannot retrieve
content from Case B even if both are indexed in the same vector store.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         /ai/process                              │
│   (existing pipeline: Loader → OCR → NER → PII → Policy → Boxes) │
│                              │                                    │
│                              ▼                                    │
│                     index_document()                              │
│              chunk_text → embed_texts → vector_store.add          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                          /ai/chat                                 │
│                              │                                    │
│   query ──► embed_texts ──► vector_store.search (RBAC filtered)   │
│                              │                                    │
│                              ▼                                    │
│              build context with [1] [2] [3] citation tags         │
│                              │                                    │
│                              ▼                                    │
│                    local LLM (Ollama) generates answer             │
│                              │                                    │
│                              ▼                                    │
│         { answer, sources: [{document_id, storage_key, page,      │
│                                excerpt, confidence}] }              │
└─────────────────────────────────────────────────────────────────┘
```

---

## New files

```
app/
├── routers/
│   └── chat.py                 # POST /ai/chat endpoint
└── services/
    ├── chunking.py             # Splits OCR text into overlapping chunks
    ├── embedding_service.py    # sentence-transformers embedding wrapper
    ├── vector_store.py         # FAISS-backed vector index + RBAC filtering
    └── chat_service.py         # Retrieval + prompt building + LLM call
```

`app/routers/ai.py` was modified: after `/ai/process` finishes OCR and
classification, it now calls `index_document()` to chunk, embed, and store
the document in the vector index before returning the response.

`app/main.py` was modified: registers the new `chat` router and pre-warms
the embedding model at startup alongside PaddleOCR and spaCy.

---

## How indexing works

Whenever `/ai/process` runs on a document, its OCR text is:

1. Split into overlapping chunks (`chunk_text`, default ~400 words with
   80-word overlap) — keeps context intact across chunk boundaries.
2. Embedded into vectors using `sentence-transformers` (`all-MiniLM-L6-v2`,
   384-dim).
3. Stored in a local FAISS index (`vector_store/index.faiss`) along with
   metadata (`document_id`, `storage_key`, `doc_class`, `page`,
   `access_roles`).

No extra step is needed — indexing happens automatically as part of
`/ai/process`.

---

## How querying works

`POST /ai/chat`

**Request:**
```json
{
  "query": "What was the vehicle number mentioned in the FIR?",
  "user_roles": ["default"]
}
```

**Response:**
```json
{
  "answer": "The vehicle registered at the scene was UP32CA1234 [1].",
  "sources": [
    {
      "citation_id": 1,
      "document_id": "fir-003",
      "storage_key": "fir_003.pdf",
      "page": 0,
      "doc_class": "FIR",
      "excerpt": "...Vehicle UP32CA1234 registered at the scene...",
      "confidence": 0.91
    }
  ]
}
```

Flow:
1. The query is embedded with the same model used for indexing.
2. `vector_store.search()` retrieves the top-k most similar chunks,
   filtered so only chunks whose `access_roles` overlap with the
   requester's `user_roles` are considered (RBAC enforcement at
   retrieval time).
3. Retrieved chunks are numbered `[1]`, `[2]`, ... and passed to the local
   LLM as context, with an instruction to answer only from that context and
   cite the chunk number for every claim.
4. The LLM's citation markers are mapped back to full source metadata
   (`document_id`, `storage_key`, `page`, `excerpt`) so the frontend can
   render each citation as a clickable badge that opens the exact source
   document/page.
5. If no relevant chunks are found for the user's role, the endpoint returns
   a "not found in accessible documents" answer instead of guessing.

---

## Setup

### 1. Install dependencies

```bash
pip install faiss-cpu sentence-transformers requests --break-system-packages
```

Add `faiss-cpu`, `sentence-transformers`, and `requests` to `pyproject.toml`
so teammates get them on `pip install -e ".[dev]"`.

### 2. Set up a local LLM (Ollama)

Data sovereignty requirement (per project scope) means the LLM must run
locally, not call an external API.

```bash
# Install Ollama from https://ollama.com
ollama pull llama3
ollama serve
```

This starts a local server at `http://localhost:11434`, which
`chat_service.py` calls by default. To use a different model or port, update
`LOCAL_LLM_URL` / `LOCAL_LLM_MODEL` in `app/services/chat_service.py`.

### 3. Run the service

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

On startup, logs should show:
```
Pre-warming models...
PaddleOCR loaded
spaCy model loaded
Embedding model loaded
Model warm-up completed in X.XXs
```

---

## Testing

API docs: `http://localhost:8000/docs`

1. **Index documents** — call `/ai/process` for each sample doc
   (`fir_001.png` through `fir_006_poor_quality.png`). Each call OCRs the
   document and automatically indexes it for retrieval.

2. **Query the chatbot** — call `/ai/chat`:
   ```json
   {
     "query": "Was any criminal intimidation case reported?",
     "user_roles": ["default"]
   }
   ```
   Expect an `answer` string with `[n]` citations and a matching `sources`
   array.

3. **Verify RBAC filtering** — index a document with a restricted
   `access_roles` value (e.g. `["case_123"]` instead of `["default"]`) and
   confirm that a query with `user_roles: ["default"]` cannot retrieve it,
   while `user_roles: ["case_123"]` can.

---

## Configuration

| Setting | Location | Default | Notes |
|---|---|---|---|
| Chunk size / overlap | `chunking.py` | 400 words / 80 overlap | Larger chunks = more context per citation, fewer citations per answer |
| Embedding model | `embedding_service.py` | `all-MiniLM-L6-v2` | 384-dim, CPU-friendly, good default for hackathon scale |
| Vector index location | `vector_store.py` | `vector_store/index.faiss` + `vector_store/meta.pkl` | Local FAISS flat index; swap for pgvector/Elasticsearch for production scale |
| Local LLM endpoint | `chat_service.py` | `http://localhost:11434/api/generate` | Ollama by default |
| Local LLM model | `chat_service.py` | `llama3` | Any Ollama-pulled model works |
| Top-k retrieved chunks | `chat_service.py` (`answer_query`) | 5 | Increase for broader recall, decrease for tighter/faster answers |

---

## Known limitations (current implementation)

- **Single page per document**: indexing currently tags every chunk with
  `page: 0` because `/ai/process` OCRs only `pages[0]`. If multi-page OCR is
  added later, pass the actual page number into `index_document()` per chunk.
- **`access_roles` is hardcoded to `["default"]`** in `/ai/process`. For a
  real RBAC demo, this should be passed in dynamically (e.g. from the
  officer's assigned case ID) rather than defaulted.
- **FAISS index is local and in-memory-backed on disk** (`IndexFlatIP`) —
  fine for demo/hackathon scale (hundreds of documents). Not built for
  concurrent writes at production scale; migrate to pgvector or
  Elasticsearch for that.
- **No persistence versioning** — re-indexing the same document twice will
  add duplicate chunks. Add a check/delete-by-`document_id` step before
  re-indexing if documents can be reprocessed.

---

## Where this fits in the SIH pitch

This module is the working implementation of the **RBAC-aware RAG**
differentiator: retrieval is tied directly to access control, so search
results are scoped to what the officer is authorized to see, and every
answer is traceable to a specific, openable source document — supporting the
court-exportable audit trail requirement as well.