"""Demo script for Backend AI service."""
import os
import sys
import time

os.environ["FLAGS_use_mkldnn"] = "0"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.loader import load_document
from app.services.ocr_service import ocr_image
from app.services.ner_service import extract_entities
from app.services.pii_patterns import scan_text_for_pii
from app.services.redaction_service import map_char_offsets_to_boxes, generate_entity_id
from app.services.pii_service import classify_pii_entities
from app.services.classifier import classify_document
from app.services.redaction_policy import apply_redaction_policy
from app.schemas import Entity


DEMO_DOCS = [
    ("fir_001.png", "FIR with full PII"),
    ("fir_002.png", "FIR with different names"),
    ("fir_006_poor_quality.png", "Poor quality scan"),
]


def run_demo():
    print("=" * 60)
    print("Backend AI Service - Demo")
    print("=" * 60)

    for doc_name, description in DEMO_DOCS:
        print(f"\n{'=' * 60}")
        print(f"Document: {doc_name}")
        print(f"Description: {description}")
        print(f"{'=' * 60}")

        start_time = time.time()

        try:
            imgs = load_document(doc_name)
            ocr = ocr_image(imgs[0])

            print(f"\nOCR Text (first 200 chars):")
            print(f"  {ocr.text[:200]}...")
            print(f"  Confidence: {ocr.confidence:.2f}")
            print(f"  Words detected: {len(ocr.words)}")

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

            redact_boxes = [box for box in redaction_boxes if box.reason != "preserve"]
            preserve_entities = [e for e in classified_entities if e.action == "preserve"]
            redact_entities = [e for e in classified_entities if e.action == "redact"]

            doc_class, doc_confidence = classify_document(ocr.text)

            elapsed = time.time() - start_time

            print(f"\nDocument Classification:")
            print(f"  Type: {doc_class} (confidence: {doc_confidence:.2f})")

            print(f"\nEntities Detected: {len(classified_entities)}")
            print(f"  Redact: {len(redact_entities)}")
            for e in redact_entities[:5]:
                print(f"    - {e.text} ({e.label}) [action: {e.action}]")
            print(f"  Preserve: {len(preserve_entities)}")
            for e in preserve_entities[:3]:
                print(f"    - {e.text} ({e.label}) [action: {e.action}]")

            print(f"\nRedaction Boxes: {len(redact_boxes)}")
            for box in redact_boxes[:3]:
                print(f"  - page={box.page} x={box.x:.3f} y={box.y:.3f} w={box.width:.3f} h={box.height:.3f} reason={box.reason}")

            print(f"\nProcessing Time: {elapsed:.2f}s")

        except Exception as e:
            print(f"\nError: {e}")

    print(f"\n{'=' * 60}")
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
