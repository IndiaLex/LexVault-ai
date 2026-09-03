"""Redaction box mapper - converts character offsets to page coordinates.

The "box mapper" is the critical bridge between text-level entity detection
and visual redaction. It takes entities with character offsets (from NER/PII)
and maps them to normalized page coordinates (0.0-1.0) using the OCR word
bounding boxes.

This approach survives zoom/DPI differences because coordinates are
normalized relative to page dimensions, not absolute pixels.
"""
from dataclasses import dataclass
import uuid


@dataclass
class Entity:
    """Internal entity representation for box mapping.

    Attributes:
        text: Entity text content.
        label: Entity type label.
        start_char: Start character offset in OCR text.
        end_char: End character offset in OCR text.
        confidence: Detection confidence.
        entity_id: Unique identifier.
        role: Person role if applicable.
    """
    text: str
    label: str
    start_char: int
    end_char: int
    confidence: float
    entity_id: str
    role: str | None = None


@dataclass
class WordBox:
    """OCR word with bounding box (used for mapping).

    Attributes:
        text: The detected word.
        bbox: 4-point polygon [[x1,y1],[x2,y2],[x3,y3],[x4,y4]].
        confidence: Detection confidence.
        char_offset_start: Start offset in full OCR text.
        char_offset_end: End offset in full OCR text.
    """
    text: str
    bbox: list[list[float]]
    confidence: float
    char_offset_start: int
    char_offset_end: int


@dataclass
class OCRResult:
    """OCR result used for box mapping.

    Attributes:
        text: Full OCR text.
        words: List of WordBox objects.
        confidence: Average confidence.
    """
    text: str
    words: list[WordBox]
    confidence: float


@dataclass
class RedactionBox:
    """Normalized bounding box for redaction.

    Attributes:
        page: Page number (0-indexed).
        x: Normalized x-coordinate (0.0-1.0).
        y: Normalized y-coordinate (0.0-1.0).
        width: Normalized width (0.0-1.0).
        height: Normalized height (0.0-1.0).
        reason: Entity type that triggered this redaction.
        confidence: Average confidence of overlapping OCR words.
        entity_id: Links to the detected entity.
    """
    page: int
    x: float
    y: float
    width: float
    height: float
    reason: str
    confidence: float
    entity_id: str


def map_char_offsets_to_boxes(
    entities: list[Entity],
    ocr_result: OCRResult,
    page_width: int,
    page_height: int,
    page_num: int = 0,
) -> list[RedactionBox]:
    """Map entity char offsets to normalized page coordinates.

    For each entity, finds which OCR words overlap with the entity's
    character range, then computes a bounding box from those words'
    pixel coordinates, normalized to 0.0-1.0.

    Args:
        entities: List of detected entities with char offsets.
        ocr_result: OCR result with word bounding boxes.
        page_width: Page width in pixels.
        page_height: Page height in pixels.
        page_num: Page number (0-indexed).

    Returns:
        List of RedactionBox with normalized coordinates.
    """
    boxes = []

    for entity in entities:
        overlapping_words = _find_overlapping_words(entity, ocr_result.words)

        if not overlapping_words:
            continue

        # Compute bounding box from overlapping words
        min_x = min(w.bbox[0][0] for w in overlapping_words)
        min_y = min(w.bbox[0][1] for w in overlapping_words)
        max_x = max(w.bbox[2][0] for w in overlapping_words)
        max_y = max(w.bbox[2][1] for w in overlapping_words)

        # Normalize to 0.0-1.0 range
        x_norm = min_x / page_width
        y_norm = min_y / page_height
        w_norm = (max_x - min_x) / page_width
        h_norm = (max_y - min_y) / page_height

        # Clamp to valid range [0, 1]
        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))
        w_norm = max(0.0, min(1.0, w_norm))
        h_norm = max(0.0, min(1.0, h_norm))

        avg_confidence = sum(w.confidence for w in overlapping_words) / len(overlapping_words)

        boxes.append(
            RedactionBox(
                page=page_num,
                x=round(x_norm, 4),
                y=round(y_norm, 4),
                width=round(w_norm, 4),
                height=round(h_norm, 4),
                reason=entity.label,
                confidence=round(avg_confidence, 4),
                entity_id=entity.entity_id,
            )
        )

    return boxes


def _find_overlapping_words(entity: Entity, words: list[WordBox]) -> list[WordBox]:
    """Find OCR words whose character range overlaps with the entity.

    Uses character offset ranges to determine overlap between
    an entity and OCR-detected words.
    """
    overlapping = []
    for word in words:
        # Check if word char range overlaps with entity char range
        if word.char_offset_start < entity.end_char and word.char_offset_end > entity.start_char:
            overlapping.append(word)
    return overlapping


def generate_entity_id() -> str:
    """Generate a unique entity ID (ent-XXXXXXXX format)."""
    return f"ent-{uuid.uuid4().hex[:8]}"
