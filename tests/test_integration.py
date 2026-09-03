"""Integration tests for the full AI pipeline."""
import pytest
import os
import sys

os.environ["FLAGS_use_mkldnn"] = "0"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from app.services.loader import load_document
from app.services.ocr_service import ocr_image
from app.services.ner_service import extract_entities
from app.services.pii_patterns import scan_text_for_pii
from app.services.redaction_service import map_char_offsets_to_boxes, generate_entity_id
from app.services.pii_service import classify_pii_entities, needs_review
from app.services.classifier import classify_document
from app.services.redaction_policy import apply_redaction_policy


@pytest.fixture
def sample_fir():
    return "fir_001.png"


@pytest.fixture
def ocr_result(sample_fir):
    imgs = load_document(sample_fir)
    return ocr_image(imgs[0])


class TestOCR:
    def test_ocr_returns_text(self, ocr_result):
        assert len(ocr_result.text) > 0

    def test_ocr_has_words(self, ocr_result):
        assert len(ocr_result.words) > 0

    def test_ocr_confidence_positive(self, ocr_result):
        assert ocr_result.confidence > 0


class TestNER:
    def test_ner_extracts_entities(self, ocr_result):
        entities = extract_entities(ocr_result.text)
        assert len(entities) > 0

    def test_ner_has_labels(self, ocr_result):
        entities = extract_entities(ocr_result.text)
        for ent in entities:
            assert ent.label is not None
            assert len(ent.label) > 0


class TestPII:
    def test_pii_detects_aadhaar(self):
        text = "Aadhaar: 234567890123"
        pii = scan_text_for_pii(text)
        aadhaar = [p for p in pii if p.label == "AADHAAR"]
        assert len(aadhaar) > 0
        assert aadhaar[0].text == "234567890123"

    def test_pii_detects_pan(self):
        text = "PAN: ABCDE1234F"
        pii = scan_text_for_pii(text)
        pan = [p for p in pii if p.label == "PAN"]
        assert len(pan) > 0
        assert pan[0].text == "ABCDE1234F"

    def test_pii_detects_phone(self):
        text = "Phone: 9876543210"
        pii = scan_text_for_pii(text)
        phone = [p for p in pii if p.label == "PHONE"]
        assert len(phone) > 0

    def test_pii_detects_vehicle(self):
        text = "Vehicle: UP32CA1234"
        pii = scan_text_for_pii(text)
        vehicle = [p for p in pii if p.label == "VEHICLE"]
        assert len(vehicle) > 0
        assert vehicle[0].text == "UP32CA1234"

    def test_pii_detects_ifsc(self):
        text = "IFSC: SBIN0001234"
        pii = scan_text_for_pii(text)
        ifsc = [p for p in pii if p.label == "IFSC"]
        assert len(ifsc) > 0
        assert ifsc[0].text == "SBIN0001234"


class TestClassifier:
    def test_classifies_fir(self):
        text = "First Information Report FIR No. 123/2026"
        doc_class, confidence = classify_document(text)
        assert doc_class == "FIR"
        assert confidence > 0.5


class TestRedactionPolicy:
    def test_aadhaar_redacted(self):
        from app.services.redaction_policy import Entity
        entity = Entity(
            text="234567890123",
            label="AADHAAR",
            start_char=0,
            end_char=12,
            confidence=1.0,
            entity_id="test-1",
        )
        result = apply_redaction_policy([entity])
        assert result[0].action == "redact"

    def test_date_preserved(self):
        from app.services.redaction_policy import Entity
        entity = Entity(
            text="15-Jan-2026",
            label="DATE",
            start_char=0,
            end_char=11,
            confidence=1.0,
            entity_id="test-2",
        )
        result = apply_redaction_policy([entity])
        assert result[0].action == "preserve"


class TestFullPipeline:
    def test_process_document(self, sample_fir):
        from app.services.loader import load_document
        from app.services.ocr_service import ocr_image
        from app.services.ner_service import extract_entities
        from app.services.pii_patterns import scan_text_for_pii
        from app.services.redaction_service import map_char_offsets_to_boxes, generate_entity_id
        from app.services.pii_service import classify_pii_entities
        from app.services.classifier import classify_document
        from app.services.redaction_policy import apply_redaction_policy
        from app.schemas import Entity

        imgs = load_document(sample_fir)
        ocr = ocr_image(imgs[0])
        spacy_entities = extract_entities(ocr.text)
        pii_entities = scan_text_for_pii(ocr.text)

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

        classified_entities = classify_pii_entities(all_entities, ocr.text)
        classified_entities = apply_redaction_policy(classified_entities)

        page_width, page_height = imgs[0].size
        redaction_boxes = map_char_offsets_to_boxes(
            classified_entities, ocr, page_width, page_height
        )

        redaction_boxes = [box for box in redaction_boxes if box.reason != "preserve"]

        doc_class, doc_confidence = classify_document(ocr.text)

        assert len(ocr.text) > 0
        assert len(classified_entities) > 0
        assert len(redaction_boxes) > 0
        assert doc_class == "FIR"
        assert doc_confidence > 0.5
