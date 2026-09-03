"""OCR service with dual-engine support.

Provides text extraction from images using PaddleOCR as primary engine
with pytesseract as fallback. Both engines return word-level bounding
boxes needed for the redaction coordinate mapping.

The OCRResult dataclass contains:
- text: Full extracted text (words joined by spaces)
- words: List of WordBox with per-word bounding boxes and char offsets
- confidence: Average confidence across all detected words
"""
from dataclasses import dataclass, field
from PIL import Image
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class WordBox:
    """A single detected word with its bounding box and position.

    Attributes:
        text: The detected word text.
        bbox: 4-point polygon [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] in pixels.
        confidence: Detection confidence (0.0-1.0).
        char_offset_start: Start offset in the full OCR text string.
        char_offset_end: End offset in the full OCR text string.
    """
    text: str
    bbox: list[list[float]]
    confidence: float
    char_offset_start: int = 0
    char_offset_end: int = 0


@dataclass
class OCRResult:
    """Complete OCR output for a single page image.

    Attributes:
        text: Full extracted text (all words joined by spaces).
        words: List of WordBox objects with bounding boxes.
        confidence: Average confidence across all words.
    """
    text: str
    words: list[WordBox] = field(default_factory=list)
    confidence: float = 0.0


def ocr_image(image: Image.Image) -> OCRResult:
    """OCR an image using PaddleOCR (primary) or pytesseract (fallback).

    Tries PaddleOCR first for better accuracy on scanned documents.
    Falls back to pytesseract if PaddleOCR fails or is unavailable.

    Args:
        image: PIL Image to process.

    Returns:
        OCRResult with text, word boxes, and confidence.
    """
    try:
        return _ocr_paddle(image)
    except Exception as e:
        logger.warning(f"PaddleOCR failed: {e}, falling back to pytesseract")
        return _ocr_tesseract(image)


def _ocr_paddle(image: Image.Image) -> OCRResult:
    """OCR using PaddleOCR engine.

    Uses PaddleOCR's detection + recognition pipeline to extract text
    with 4-point polygon bounding boxes for each detected text region.
    """
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(lang="en")
    img_array = np.array(image)
    result = ocr.ocr(img_array)

    if not result or not result[0]:
        raise RuntimeError("PaddleOCR returned empty result")

    words = []
    full_text_parts = []
    char_offset = 0

    for line in result[0]:
        bbox = line[0]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        text = line[1][0]
        conf = line[1][1]

        word = WordBox(
            text=text,
            bbox=bbox,
            confidence=conf,
            char_offset_start=char_offset,
            char_offset_end=char_offset + len(text),
        )
        words.append(word)
        full_text_parts.append(text)
        char_offset += len(text) + 1  # +1 for space between words

    full_text = " ".join(full_text_parts)
    avg_confidence = sum(w.confidence for w in words) / len(words) if words else 0.0

    return OCRResult(text=full_text, words=words, confidence=avg_confidence)


def _ocr_tesseract(image: Image.Image) -> OCRResult:
    """OCR using pytesseract as fallback engine.

    Converts pytesseract's left/top/width/height output to the same
    4-point polygon format used by PaddleOCR for consistency.
    """
    import pytesseract

    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    words = []
    full_text_parts = []
    char_offset = 0

    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        conf = int(data["conf"][i])

        if not text or conf <= 0:
            continue

        x = data["left"][i]
        y = data["top"][i]
        w = data["width"][i]
        h = data["height"][i]

        # Convert rectangle to 4-point polygon format
        bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]

        word = WordBox(
            text=text,
            bbox=bbox,
            confidence=conf / 100.0,  # pytesseract returns 0-100, normalize to 0-1
            char_offset_start=char_offset,
            char_offset_end=char_offset + len(text),
        )
        words.append(word)
        full_text_parts.append(text)
        char_offset += len(text) + 1

    full_text = " ".join(full_text_parts)
    avg_confidence = sum(w.confidence for w in words) / len(words) if words else 0.0

    return OCRResult(text=full_text, words=words, confidence=avg_confidence)
