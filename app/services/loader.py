"""Document loader service.

Loads documents from local filesystem or MinIO object storage.
Supports PDF (multi-page) and image files (PNG, JPG, TIFF, BMP).
Optionally applies image preprocessing for poor quality scans.
"""
from pathlib import Path
from PIL import Image
import pymupdf
import tempfile
import logging

logger = logging.getLogger(__name__)

# Default directory for sample documents
SAMPLE_DOCS_DIR = Path(__file__).parent.parent.parent / "sample_docs"


def load_document(storage_key: str, apply_preprocessing: bool = False) -> list[Image.Image]:
    """Load a document and return list of page images.

    Args:
        storage_key: Path to local file or MinIO key (minio://bucket/path).
        apply_preprocessing: Whether to apply preprocessing for poor scans.

    Returns:
        List of PIL Images, one per page.

    Raises:
        FileNotFoundError: If local file does not exist.
        ValueError: If file type is unsupported.
    """
    if storage_key.startswith("minio://"):
        return _load_from_minio(storage_key, apply_preprocessing)
    return _load_local(storage_key, apply_preprocessing)


def _load_local(storage_key: str, apply_preprocessing: bool = False) -> list[Image.Image]:
    """Load from local filesystem.

    First tries the exact path, then looks in sample_docs/ directory.
    """
    path = Path(storage_key)
    if not path.exists():
        path = SAMPLE_DOCS_DIR / storage_key
    if not path.exists():
        raise FileNotFoundError(f"File not found: {storage_key}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        images = _load_pdf(path)
    elif suffix in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
        images = [Image.open(path).convert("RGB")]
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    if apply_preprocessing:
        from app.services.preprocessing import preprocess_image
        images = [preprocess_image(img) for img in images]

    return images


def _load_pdf(path: Path) -> list[Image.Image]:
    """Convert PDF pages to PIL Images at 200 DPI.

    Each page becomes a separate Image in the returned list.
    """
    doc = pymupdf.open(str(path))
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
    return images


def _load_from_minio(storage_key: str, apply_preprocessing: bool = False) -> list[Image.Image]:
    """Load from MinIO object storage.

    Downloads the file to a temp location, loads it, then cleans up.
    """
    try:
        from app.services.minio_client import download_from_minio
        temp_path = download_from_minio(storage_key)
        suffix = temp_path.suffix.lower()

        if suffix == ".pdf":
            images = _load_pdf(temp_path)
        elif suffix in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
            images = [Image.open(temp_path).convert("RGB")]
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        temp_path.unlink(missing_ok=True)

        if apply_preprocessing:
            from app.services.preprocessing import preprocess_image
            images = [preprocess_image(img) for img in images]

        return images
    except Exception as e:
        logger.error(f"MinIO load failed: {e}")
        raise
