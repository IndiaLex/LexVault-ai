"""Backend AI Service - FastAPI application entry point.

This module initializes the FastAPI application, registers route handlers,
and pre-warms ML models at startup for faster first-request response times.
"""
import os
import time
import logging
from fastapi import FastAPI

# Disable oneDNN optimization to avoid PaddlePaddle compatibility issues on Windows
os.environ["FLAGS_use_mkldnn"] = "0"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from app.routers.health import router as health_router
from app.routers.ai import router as ai_router
from app.routers.chat import router as chat_router

app = FastAPI(
    title="Backend AI Service",
    version="0.1.0",
    description="OCR, NER, and PII detection for scanned police documents",
)

app.include_router(health_router)
app.include_router(ai_router)
app.include_router(chat_router)


@app.on_event("startup")
def warm_models():
    """Pre-load ML models at startup to avoid cold-start latency.

    Loads PaddleOCR, spaCy, and the RAG embedding model into memory during
    application startup. First request after startup will be fast since
    models are already loaded.
    """
    logger.info("Pre-warming models...")
    start = time.time()

    try:
        from app.services.ocr_service import _ocr_paddle
        from PIL import Image
        dummy = Image.new("RGB", (100, 100), "white")
        _ocr_paddle(dummy)
        logger.info("PaddleOCR loaded")
    except Exception as e:
        logger.warning(f"PaddleOCR warm-up failed: {e}")

    try:
        from app.services.ner_service import _get_nlp
        _get_nlp()
        logger.info("spaCy model loaded")
    except Exception as e:
        logger.warning(f"spaCy warm-up failed: {e}")

    try:
        from app.services.embedding_service import embed_texts
        embed_texts(["warm up"])
        logger.info("Embedding model loaded")
    except Exception as e:
        logger.warning(f"Embedding model warm-up failed: {e}")

    elapsed = time.time() - start
    logger.info(f"Model warm-up completed in {elapsed:.2f}s")