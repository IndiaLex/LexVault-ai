"""Pydantic models for API request/response schemas.

Defines the contract between the Backend AI service and its consumers.
All coordinates are normalized (0.0-1.0) to be resolution-independent.
"""
from pydantic import BaseModel


class ProcessRequest(BaseModel):
    """Request body for POST /ai/process.

    Attributes:
        document_id: Unique identifier for the document.
        storage_key: Path to local file or MinIO key (minio://bucket/path).
    """
    document_id: str
    storage_key: str


class Entity(BaseModel):
    """A detected named entity or PII item.

    Attributes:
        text: The detected text content.
        label: Entity type (AADHAAR, PAN, PHONE, PERSON, GPE, etc.).
        start_char: Start character offset in OCR text.
        end_char: End character offset in OCR text.
        confidence: Detection confidence (0.0-1.0).
        entity_id: Unique identifier for this entity.
        role: Person role if applicable (officer, victim, witness).
        action: Redaction action (redact, preserve) after policy applied.
    """
    text: str
    label: str
    start_char: int
    end_char: int
    confidence: float
    entity_id: str
    role: str | None = None
    action: str | None = None


class RedactionBox(BaseModel):
    """A normalized bounding box for redaction overlay.

    All coordinates are normalized to 0.0-1.0 range relative to page dimensions.
    The frontend multiplies these by actual page width/height to render overlays.

    Attributes:
        page: Page number (0-indexed).
        x: Normalized x-coordinate of top-left corner.
        y: Normalized y-coordinate of top-left corner.
        width: Normalized width of the box.
        height: Normalized height of the box.
        reason: Entity type that triggered this redaction.
        confidence: Average confidence of underlying OCR words.
        entity_id: Links back to the detected entity.
    """
    page: int
    x: float
    y: float
    width: float
    height: float
    reason: str
    confidence: float
    entity_id: str


class AIResult(BaseModel):
    """Response from POST /ai/process.

    Contains the full pipeline output: OCR text, detected entities,
    redaction boxes, document classification, and review flag.

    Attributes:
        document_id: Echo of the request's document_id.
        ocr_text: Full extracted text from the document.
        entities: All detected entities (NER + PII) with actions.
        redaction_boxes: Normalized coordinates for entities marked 'redact'.
        doc_class: Document type (FIR, WITNESS, MEDICAL, OTHER).
        confidence: Document classification confidence.
        needs_review: True if any entity has confidence below threshold.
    """
    document_id: str
    ocr_text: str
    entities: list[Entity]
    redaction_boxes: list[RedactionBox]
    doc_class: str
    confidence: float
    needs_review: bool


class RedactRequest(BaseModel):
    """Request body for POST /ai/redact.

    Attributes:
        document_id: Unique identifier for the document.
        storage_key: Path to the original document.
        boxes: List of redaction boxes to burn into the PDF.
    """
    document_id: str
    storage_key: str
    boxes: list[RedactionBox]


class RedactResponse(BaseModel):
    """Response from POST /ai/redact.

    Attributes:
        redacted_storage_key: Storage key for the redacted PDF.
    """
    redacted_storage_key: str
