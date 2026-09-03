"""Named Entity Recognition service using spaCy.

Extracts named entities (PERSON, GPE, ORG, DATE, TIME, LAW, etc.) from
OCR text using spaCy's pre-trained en_core_web_sm model.

The model is loaded once at module level and reused across requests
for performance.
"""
from dataclasses import dataclass
import spacy
import logging

logger = logging.getLogger(__name__)

# spaCy model loaded once at module level (lazy initialization)
_nlp = None


def _get_nlp():
    """Load or return cached spaCy NLP model.

    Downloads the model if not found locally.
    """
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model 'en_core_web_sm' not found. Downloading...")
            spacy.cli.download("en_core_web_sm")
            _nlp = spacy.load("en_core_web_sm")
    return _nlp


@dataclass
class NEREntity:
    """A named entity extracted by spaCy.

    Attributes:
        text: The entity text.
        label: Entity type (PERSON, GPE, ORG, DATE, TIME, LAW, etc.).
        start_char: Start character offset in the input text.
        end_char: End character offset in the input text.
        confidence: Detection confidence (spaCy default: 1.0).
    """
    text: str
    label: str
    start_char: int
    end_char: int
    confidence: float


def extract_entities(text: str) -> list[NEREntity]:
    """Extract named entities from text using spaCy NER.

    Args:
        text: Input text to process.

    Returns:
        List of NEREntity objects with entity type and position.
    """
    nlp = _get_nlp()
    doc = nlp(text)

    entities = []
    for ent in doc.ents:
        entities.append(
            NEREntity(
                text=ent.text,
                label=ent.label_,
                start_char=ent.start_char,
                end_char=ent.end_char,
                confidence=1.0,  # spaCy doesn't provide confidence by default
            )
        )

    return entities
