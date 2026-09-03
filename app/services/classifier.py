"""Document type classifier.

Classifies scanned police documents into categories based on
keyword matching in the OCR text. Uses a scoring system where
multiple keyword matches increase confidence.

Supported document types:
- FIR: First Information Report
- WITNESS: Witness statement / deposition
- MEDICAL: Medical report / injury report
- OTHER: Unclassified documents
"""
import re


def classify_document(text: str) -> tuple[str, float]:
    """Classify document type based on content keywords.

    Uses regex pattern matching to detect document type keywords
    and returns the best match with a confidence score.

    Args:
        text: OCR text from the document.

    Returns:
        Tuple of (doc_class, confidence).
        doc_class is one of: FIR, WITNESS, MEDICAL, OTHER.
        confidence ranges from 0.5 to 0.95.
    """
    text_lower = text.lower()

    # FIR detection keywords
    fir_patterns = [
        r"first information report",
        r"\bfir\b",
        r"\bfir no\b",
        r"\bfir number\b",
    ]
    fir_score = sum(1 for p in fir_patterns if re.search(p, text_lower))

    # Witness statement detection keywords
    witness_patterns = [
        r"witness statement",
        r"statement of witness",
        r"deposition",
        r"testimony",
    ]
    witness_score = sum(1 for p in witness_patterns if re.search(p, text_lower))

    # Medical report detection keywords
    medical_patterns = [
        r"medical report",
        r"injury",
        r"hospital",
        r"doctor",
        r"patient",
        r"clinical",
    ]
    medical_score = sum(1 for p in medical_patterns if re.search(p, text_lower))

    scores = {
        "FIR": fir_score,
        "WITNESS": witness_score,
        "MEDICAL": medical_score,
    }

    max_score = max(scores.values())
    if max_score == 0:
        return ("OTHER", 0.5)

    best_class = max(scores, key=scores.get)
    # Confidence increases with more keyword matches
    confidence = min(0.5 + (max_score * 0.15), 0.95)

    return (best_class, round(confidence, 4))
