"""AI processing endpoints.

Implements the core API routes:
- POST /ai/process: Full OCR + NER + PII pipeline
- POST /ai/redact: Generate burned-in redacted PDF

The /ai/process endpoint orchestrates the entire pipeline:
loader → OCR → NER → PII scan → confidence scoring → redaction policy → box mapping → classification.
"""
from fastapi import APIRouter, HTTPException
import pymupdf
import tempfile
from pathlib import Path

from app.schemas import (
    ProcessRequest,
    AIResult,
    Entity,
    RedactionBox,
    RedactRequest,
    RedactResponse,
)
from app.services.loader import load_document
from app.services.ocr_service import ocr_image, OCRResult
from app.services.ner_service import extract_entities
from app.services.pii_patterns import scan_text_for_pii
from app.services.redaction_service import (
    map_char_offsets_to_boxes,
    generate_entity_id,
)
from app.services.pii_service import classify_pii_entities, needs_review
from app.services.classifier import classify_document
from app.services.redaction_policy import apply_redaction_policy

router = APIRouter()


@router.post("/ai/process", response_model=AIResult)
def process_document(req: ProcessRequest) -> AIResult:
    """Process a document through the full AI pipeline.

    Steps:
    1. Load document (local file or MinIO)
    2. OCR first page with PaddleOCR/pytesseract
    3. Extract named entities with spaCy NER
    4. Scan for Indian PII with regex patterns
    5. Score confidence and detect roles
    6. Apply redaction policy (preserve vs redact)
    7. Map char offsets to normalized page coordinates
    8. Classify document type
    9. Return AIResult with all findings
    """
    try:
        pages = load_document(req.storage_key)
        ocr_result: OCRResult = ocr_image(pages[0])
        spacy_entities = extract_entities(ocr_result.text)
        pii_entities = scan_text_for_pii(ocr_result.text)

        all_entities = []
        for ent in spacy_entities:
            all_entities.append(
                Entity(
                    text=ent.text,
                    label=ent.label,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    confidence=ent.confidence,
                    entity_id=generate_entity_id(),
                )
            )
        for ent in pii_entities:
            all_entities.append(
                Entity(
                    text=ent.text,
                    label=ent.label,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    confidence=ent.confidence,
                    entity_id=generate_entity_id(),
                )
            )

        classified_entities = classify_pii_entities(all_entities, ocr_result.text)
        policy_entities = apply_redaction_policy(classified_entities)

        # Convert dataclass entities to Pydantic entities for API response
        classified_entities = [
            Entity(
                text=e.text,
                label=e.label,
                start_char=e.start_char,
                end_char=e.end_char,
                confidence=e.confidence,
                entity_id=e.entity_id,
                role=e.role,
                action=e.action,
            )
            for e in policy_entities
        ]

        page_width, page_height = pages[0].size
        raw_boxes = map_char_offsets_to_boxes(
            classified_entities, ocr_result, page_width, page_height
        )

        # Only include boxes for entities marked 'redact', convert to Pydantic
        redaction_boxes = [
            RedactionBox(
                page=b.page,
                x=b.x,
                y=b.y,
                width=b.width,
                height=b.height,
                reason=b.reason,
                confidence=b.confidence,
                entity_id=b.entity_id,
            )
            for b in raw_boxes
            if b.reason != "preserve"
        ]

        doc_class, doc_confidence = classify_document(ocr_result.text)
        review_needed = needs_review(classified_entities)

        return AIResult(
            document_id=req.document_id,
            ocr_text=ocr_result.text,
            entities=classified_entities,
            redaction_boxes=redaction_boxes,
            doc_class=doc_class,
            confidence=doc_confidence,
            needs_review=review_needed,
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.post("/ai/redact", response_model=RedactResponse)
def redact_document(req: RedactRequest) -> RedactResponse:
    """Generate a burned-in redacted PDF.

    Loads the original document, draws black rectangles at the specified
    normalized coordinates, and saves as a new PDF. Returns a storage key
    for the redacted file.
    """
    try:
        pages = load_document(req.storage_key)

        temp_dir = tempfile.mkdtemp()
        output_path = Path(temp_dir) / f"redacted_{req.document_id}.pdf"

        doc = pymupdf.open()

        for page_num, page_img in enumerate(pages):
            img_width, img_height = page_img.size
            page = doc.new_page(width=img_width, height=img_height)

            img_path = Path(temp_dir) / f"page_{page_num}.png"
            page_img.save(str(img_path))
            page.insert_image(rect=pymupdf.Rect(0, 0, img_width, img_height), filename=str(img_path))

            for box in req.boxes:
                if box.page == page_num:
                    # Convert normalized coordinates to pixel coordinates
                    x = box.x * img_width
                    y = box.y * img_height
                    w = box.width * img_width
                    h = box.height * img_height
                    rect = pymupdf.Rect(x, y, x + w, y + h)
                    page.draw_rect(rect, color=None, fill=(0, 0, 0))

        doc.save(str(output_path))
        doc.close()

        redacted_key = f"redacted_{req.document_id}.pdf"
        return RedactResponse(redacted_storage_key=redacted_key)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Redaction failed: {str(e)}")
