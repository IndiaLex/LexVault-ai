"""PII classifier - confidence scoring and role detection.

Enhances detected entities with confidence scores based on entity type
and context. Also detects person roles (officer vs victim/witness) by
analyzing surrounding text keywords.
"""
from dataclasses import dataclass


@dataclass
class Entity:
    """Entity with confidence scoring applied.

    Attributes:
        text: Entity text content.
        label: Entity type label.
        start_char: Start character offset.
        end_char: End character offset.
        confidence: Detection confidence.
        entity_id: Unique identifier.
        role: Person role (officer, victim, witness) or None.
    """
    text: str
    label: str
    start_char: int
    end_char: int
    confidence: float
    entity_id: str
    role: str | None = None


def classify_pii_entities(
    entities: list[Entity],
    context_text: str = "",
) -> list[Entity]:
    """Apply confidence scoring to PII entities.

    Adjusts confidence based on entity type:
    - AADHAAR, PAN, IFSC: 0.95 (regex exact match = high confidence)
    - PHONE: 0.90
    - VEHICLE: 0.85
    - PERSON: 0.80 (with role detection)
    - GPE, ORG, DATE, TIME: 0.75

    Args:
        entities: List of detected entities.
        context_text: Full OCR text for context analysis.

    Returns:
        List of entities with updated confidence scores.
    """
    classified = []

    for entity in entities:
        confidence = entity.confidence

        # Adjust confidence based on entity type and context
        if entity.label in ("AADHAAR", "PAN", "IFSC"):
            # Regex matches are high confidence
            confidence = max(confidence, 0.95)
        elif entity.label == "PHONE":
            confidence = max(confidence, 0.90)
        elif entity.label == "VEHICLE":
            confidence = max(confidence, 0.85)
        elif entity.label == "PERSON":
            # Check context for role
            role = _detect_person_role(entity.text, context_text)
            if role:
                entity = Entity(
                    text=entity.text,
                    label=entity.label,
                    start_char=entity.start_char,
                    end_char=entity.end_char,
                    confidence=entity.confidence,
                    entity_id=entity.entity_id,
                    role=role,
                )
            confidence = max(confidence, 0.80)
        elif entity.label in ("GPE", "ORG", "DATE", "TIME"):
            confidence = max(confidence, 0.75)

        classified.append(
            Entity(
                text=entity.text,
                label=entity.label,
                start_char=entity.start_char,
                end_char=entity.end_char,
                confidence=round(confidence, 4),
                entity_id=entity.entity_id,
                role=entity.role,
            )
        )

    return classified


def _detect_person_role(name: str, text: str) -> str | None:
    """Detect if a person name is near officer/victim keywords.

    Looks at the 200 characters before the name for role-indicating
    keywords like "Inspector", "Complainant", "Victim", etc.

    Returns:
        "officer", "victim", or None if no role detected.
    """
    pos = text.find(name)
    if pos == -1:
        return None

    # Look at surrounding context (200 chars before the name)
    context_start = max(0, pos - 200)
    context = text[context_start:pos].lower()

    officer_keywords = ["inspector", "sub-inspector", "si", "constable", "officer", "sho"]
    victim_keywords = ["complainant", "victim", "witness", "accused"]

    for kw in officer_keywords:
        if kw in context:
            return "officer"

    for kw in victim_keywords:
        if kw in context:
            return "victim"

    return None


def needs_review(entities: list[Entity], threshold: float = 0.7) -> bool:
    """Check if any entity needs human review.

    Returns True if any entity has confidence below the threshold,
    indicating it should be verified by a human before redaction.

    Args:
        entities: List of classified entities.
        threshold: Confidence threshold for review (default: 0.7).

    Returns:
        True if any entity needs review.
    """
    return any(e.confidence < threshold for e in entities)
