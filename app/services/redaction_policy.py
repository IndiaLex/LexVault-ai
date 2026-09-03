"""Redaction policy engine.

Implements the preserve/redact rules that determine which detected
entities should be redacted and which should be preserved in the
final document. Rules are based on entity type and role context.

Policy rules:
- Always REDACT: AADHAAR, PAN, BANK_ACCOUNT, IFSC, PHONE, VEHICLE, GPE
- Always PRESERVE: DATE, TIME, ORG, LAW, CARDINAL
- PERSON: Redact if victim/witness, preserve if officer
- Default: Preserve (conservative approach)
"""
from dataclasses import dataclass


@dataclass
class Entity:
    """Entity with redaction action applied.

    Attributes:
        text: Entity text content.
        label: Entity type label.
        start_char: Start character offset.
        end_char: End character offset.
        confidence: Detection confidence.
        entity_id: Unique identifier.
        role: Person role (officer, victim, witness) or None.
        action: Redaction action ("redact" or "preserve").
    """
    text: str
    label: str
    start_char: int
    end_char: int
    confidence: float
    entity_id: str
    role: str | None = None
    action: str = "preserve"


# Default policy rules by entity type
POLICY_RULES = {
    "AADHAAR": "redact",
    "PAN": "redact",
    "BANK_ACCOUNT": "redact",
    "IFSC": "redact",
    "PHONE": "redact",
    "VEHICLE": "redact",
    "DATE": "preserve",
    "TIME": "preserve",
    "GPE": "redact",
    "ORG": "preserve",
    "LAW": "preserve",
    "CARDINAL": "preserve",
}


def apply_redaction_policy(entities: list[Entity]) -> list[Entity]:
    """Apply preserve/redact rules to entities.

    For PERSON entities, uses role context:
    - "officer" role → preserve (police names are public record)
    - "victim" or "witness" role → redact (privacy protection)

    Args:
        entities: List of detected entities.

    Returns:
        Same entities with 'action' field set to 'redact' or 'preserve'.
    """
    classified = []
    for entity in entities:
        action = POLICY_RULES.get(entity.label, "preserve")

        if entity.label == "PERSON":
            if entity.role == "officer":
                action = "preserve"
            else:
                action = "redact"

        classified.append(
            Entity(
                text=entity.text,
                label=entity.label,
                start_char=entity.start_char,
                end_char=entity.end_char,
                confidence=entity.confidence,
                entity_id=entity.entity_id,
                role=entity.role,
                action=action,
            )
        )

    return classified
