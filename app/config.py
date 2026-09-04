"""Application configuration constants.

Central configuration for model paths, confidence thresholds,
storage settings, and preprocessing options.

All values read from environment variables with sensible defaults.
"""
import os
from dotenv import load_dotenv

load_dotenv()


def _bool(val: str, default: bool = False) -> bool:
    """Parse a string env var as boolean."""
    return val.lower() in ("true", "1", "yes") if val else default


def _int(val: str, default: int = 0) -> int:
    """Parse a string env var as int."""
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _float(val: str, default: float = 0.0) -> float:
    """Parse a string env var as float."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ─── spaCy / PaddleOCR model settings ──────────────────────────
MODEL_PATHS = {
    "spacy": os.getenv("SPACY_MODEL", "en_core_web_sm"),
    "paddleocr_lang": os.getenv("PADDLEOCR_LANG", "en"),
}

# ─── Core pipeline ──────────────────────────────────────────────
CONFIDENCE_THRESHOLD = _float(os.getenv("CONFIDENCE_THRESHOLD"), 0.7)
TIMEOUT_SECONDS = _int(os.getenv("TIMEOUT_SECONDS"), 30)

# ─── MinIO object storage ──────────────────────────────────────
STORAGE_CONFIG = {
    "minio_endpoint": os.getenv("MINIO_ENDPOINT", "localhost:9000"),
    "minio_access_key": os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
    "minio_secret_key": os.getenv("MINIO_SECRET_KEY", "minioadmin"),
    "minio_bucket": os.getenv("MINIO_BUCKET", "indialex-docs"),
}

MINIO_ENDPOINT = STORAGE_CONFIG["minio_endpoint"]
MINIO_ACCESS_KEY = STORAGE_CONFIG["minio_access_key"]
MINIO_SECRET_KEY = STORAGE_CONFIG["minio_secret_key"]
MINIO_SECURE = _bool(os.getenv("MINIO_SECURE"), False)

# ─── Image preprocessing ───────────────────────────────────────
PREPROCESSING_CONFIG = {
    "enabled": _bool(os.getenv("PREPROCESSING_ENABLED"), True),
    "deskew": _bool(os.getenv("PREPROCESSING_DESKEW"), True),
    "denoise": _bool(os.getenv("PREPROCESSING_DENOISE"), True),
    "normalize_contrast": _bool(os.getenv("PREPROCESSING_NORMALIZE_CONTRAST"), True),
}

# ─── RAG Chatbot - Embedding ───────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ─── RAG Chatbot - Vector Store ────────────────────────────────
VECTOR_INDEX_PATH = os.getenv("VECTOR_INDEX_PATH", "vector_store/index.faiss")
VECTOR_META_PATH = os.getenv("VECTOR_META_PATH", "vector_store/meta.pkl")
VECTOR_EMBED_DIM = _int(os.getenv("VECTOR_EMBED_DIM"), 384)

# ─── RAG Chatbot - Chunking ────────────────────────────────────
CHUNK_SIZE = _int(os.getenv("CHUNK_SIZE"), 400)
CHUNK_OVERLAP = _int(os.getenv("CHUNK_OVERLAP"), 80)

# ─── RAG Chatbot - LLM (Ollama) ───────────────────────────────
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://localhost:11434/api/generate")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "llama3")
LOCAL_LLM_TOP_K = _int(os.getenv("LOCAL_LLM_TOP_K"), 5)

# ─── RAG Chatbot - RBAC ───────────────────────────────────────
DEFAULT_ACCESS_ROLES = [
    r.strip() for r in os.getenv("DEFAULT_ACCESS_ROLES", "default").split(",")
]
