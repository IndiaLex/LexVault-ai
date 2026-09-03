"""Image preprocessing for poor scan quality.

Applies image transformations to improve OCR accuracy on scanned
documents that have skew, noise, or poor contrast. Only applies
preprocessing when the image quality is detected as poor.
"""
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import logging

logger = logging.getLogger(__name__)


def needs_preprocessing(image: Image.Image) -> bool:
    """Detect if an image needs preprocessing.

    Checks three quality indicators:
    - Low standard deviation → flat/low-contrast image
    - Low mean brightness → underexposed scan
    - High mean brightness → overexposed/washed-out scan

    Returns:
        True if image quality is poor and needs preprocessing.
    """
    img_array = np.array(image.convert("L"))
    std_dev = np.std(img_array)
    mean_val = np.mean(img_array)
    return std_dev < 40 or mean_val < 80 or mean_val > 200


def deskew(image: Image.Image) -> Image.Image:
    """Correct rotation/skew in document images.

    Uses edge detection to estimate the dominant angle of text lines,
    then rotates the image to correct any skew.
    """
    img_array = np.array(image.convert("L"))
    edges = np.array(image.convert("L").filter(ImageFilter.FIND_EDGES))

    rows_sum = np.sum(edges, axis=1)
    cols_sum = np.sum(edges, axis=0)

    if np.max(rows_sum) == 0 or np.max(cols_sum) == 0:
        return image

    rows_center = np.argmax(rows_sum) / len(rows_sum)
    cols_center = np.argmax(cols_sum) / len(cols_sum)

    center = (image.width / 2, image.height / 2)
    rotation = np.arctan2(cols_center - 0.5, rows_center - 0.5) * (180 / np.pi)

    # Only correct significant skew (> 0.5 degrees)
    if abs(rotation) < 0.5:
        return image

    return image.rotate(rotation, resample=Image.BICUBIC, expand=False, fillcolor="white")


def denoise(image: Image.Image) -> Image.Image:
    """Remove noise from scanned documents.

    Uses a median filter which is effective at removing salt-and-pepper
    noise while preserving text edges.
    """
    return image.filter(ImageFilter.MedianFilter(size=3))


def normalize_contrast(image: Image.Image) -> Image.Image:
    """Normalize contrast for poor scans.

    Increases contrast by 50% to make text more readable.
    """
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(1.5)


def preprocess_image(image: Image.Image) -> Image.Image:
    """Apply full preprocessing pipeline for poor quality scans.

    Only applies preprocessing if the image is detected as poor quality.
    Pipeline: deskew → denoise → contrast normalization.

    Args:
        image: Input PIL Image.

    Returns:
        Preprocessed image (or original if quality is acceptable).
    """
    if not needs_preprocessing(image):
        return image

    logger.info("Applying preprocessing to poor quality scan")
    processed = deskew(image)
    processed = denoise(processed)
    processed = normalize_contrast(processed)
    return processed
