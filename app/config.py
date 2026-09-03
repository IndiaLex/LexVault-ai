"""Application configuration constants.

Central configuration for model paths, confidence thresholds,
storage settings, and preprocessing options.
"""

# spaCy and PaddleOCR language/model settings
MODEL_PATHS = {
    "spacy": "en_core_web_sm",
    "paddleocr_lang": "en",
}

# Confidence threshold below which entities are flagged for human review
CONFIDENCE_THRESHOLD = 0.7

# Maximum processing time before timeout (seconds)
TIMEOUT_SECONDS = 30

# MinIO object storage configuration (used when storage_key starts with minio://)
STORAGE_CONFIG = {
    "minio_endpoint": "localhost:9000",
    "minio_access_key": "minioadmin",
    "minio_secret_key": "minioadmin",
    "minio_bucket": "indialex-docs",
}

# Expose config values as module-level constants for import convenience
MINIO_ENDPOINT = STORAGE_CONFIG["minio_endpoint"]
MINIO_ACCESS_KEY = STORAGE_CONFIG["minio_access_key"]
MINIO_SECRET_KEY = STORAGE_CONFIG["minio_secret_key"]
MINIO_SECURE = False

# Image preprocessing pipeline settings
PREPROCESSING_CONFIG = {
    "enabled": True,
    "deskew": True,
    "denoise": True,
    "normalize_contrast": True,
}
