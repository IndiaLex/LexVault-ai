# ─── Build stage ─────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps for PaddleOCR, spaCy, PyMuPDF, faiss
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]" 2>/dev/null || \
    pip install --no-cache-dir \
    fastapi uvicorn[standard] \
    paddleocr paddlepaddle \
    pytesseract spacy pymupdf pillow \
    minio python-multipart \
    faiss-cpu sentence-transformers requests numpy \
    python-dotenv \
    pytest httpx pytest-asyncio

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# ─── Runtime stage ──────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy spaCy model
COPY --from=builder /usr/local/lib/python3.11/site-packages/en_core_web_sm /usr/local/lib/python3.11/site-packages/en_core_web_sm

# Copy application code
COPY app/ ./app/
COPY sample_docs/ ./sample_docs/
COPY pyproject.toml .
COPY .env.example .env

# Create vector store directory
RUN mkdir -p vector_store

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/ai/health')" || exit 1

# Run
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
