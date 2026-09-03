"""MinIO object storage client.

Provides a wrapper around the MinIO Python client for downloading
documents from S3-compatible object storage. Used when storage_key
starts with "minio://".

The client is initialized lazily and cached for reuse across requests.
"""
import tempfile
from pathlib import Path
from minio import Minio
from minio.error import S3Error
import logging

logger = logging.getLogger(__name__)

# Cached MinIO client instance
_client = None


def get_minio_client() -> Minio:
    """Get or create the MinIO client.

    Uses lazy initialization with module-level caching.
    Connection settings come from app.config.
    """
    global _client
    if _client is None:
        from app.config import MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_SECURE
        _client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE,
        )
    return _client


def download_from_minio(storage_key: str) -> Path:
    """Download a file from MinIO to a temporary location.

    Parses the storage_key format: minio://bucket/path/to/file.pdf

    Args:
        storage_key: MinIO key in format 'minio://bucket/path/to/file'.

    Returns:
        Path to downloaded temporary file.

    Raises:
        ValueError: If storage_key format is invalid.
        RuntimeError: If download fails.
    """
    if not storage_key.startswith("minio://"):
        raise ValueError(f"Not a MinIO key: {storage_key}")

    parts = storage_key.replace("minio://", "").split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid MinIO key format: {storage_key}")

    bucket, object_name = parts
    client = get_minio_client()

    suffix = Path(object_name).suffix
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_path = Path(temp_file.name)
    temp_file.close()

    try:
        client.fget_object(bucket, object_name, str(temp_path))
        logger.info(f"Downloaded {storage_key} to {temp_path}")
        return temp_path
    except S3Error as e:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"MinIO download failed: {e}") from e
