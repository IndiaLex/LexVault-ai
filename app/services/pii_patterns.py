"""Indian PII regex pattern detection.

Scans OCR text for Indian-specific personally identifiable information
using compiled regex patterns. Returns entities with character offsets
for precise bounding box mapping.

Supported PII types:
- AADHAAR: 12-digit Indian identity number (starts with 2-9)
- PAN: 10-character Permanent Account Number (ABCDE1234F)
- PHONE: Indian mobile numbers (10 digits starting with 6-9)
- VEHICLE: Indian vehicle registration (e.g., UP32CA1234)
- IFSC: Indian Financial System Code (e.g., SBIN0001234)
- BANK_ACCOUNT: Generic bank account number (9-18 digits)
"""
import re
from dataclasses import dataclass


@dataclass
class PIIEntity:
    """A detected PII item with its position in text.

    Attributes:
        text: The detected PII value.
        label: PII type (AADHAAR, PAN, PHONE, etc.).
        start_char: Start character offset in the input text.
        end_char: End character offset in the input text.
        confidence: Detection confidence (regex matches: 1.0).
    """
    text: str
    label: str
    start_char: int
    end_char: int
    confidence: float = 1.0


# Compiled regex patterns for Indian PII types
PATTERNS = {
    "AADHAAR": re.compile(r"\b[2-9][0-9]{11}\b"),
    "PAN": re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
    "PHONE": re.compile(r"(?:\+91|91)?\s?[6-9][0-9]{9}\b"),
    "VEHICLE": re.compile(r"\b[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}\b"),
    "IFSC": re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),
    "BANK_ACCOUNT": re.compile(r"\b[0-9]{9,18}\b"),
}


def scan_text_for_pii(text: str) -> list[PIIEntity]:
    """Scan text for Indian PII patterns using regex.

    Args:
        text: Input text to scan.

    Returns:
        List of PIIEntity objects sorted by position in text.
    """
    entities = []

    for label, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            entities.append(
                PIIEntity(
                    text=match.group(),
                    label=label,
                    start_char=match.start(),
                    end_char=match.end(),
                )
            )

    # Sort by position in text for consistent ordering
    entities.sort(key=lambda e: e.start_char)
    return entities
