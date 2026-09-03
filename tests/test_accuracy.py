"""Accuracy measurement framework for PII detection."""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["FLAGS_use_mkldnn"] = "0"

from app.services.loader import load_document
from app.services.ocr_service import ocr_image
from app.services.ner_service import extract_entities
from app.services.pii_patterns import scan_text_for_pii
from app.services.redaction_service import generate_entity_id
from app.services.pii_service import classify_pii_entities
from app.services.redaction_policy import apply_redaction_policy

SAMPLE_DOCS_DIR = Path(__file__).parent.parent / "sample_docs"


GROUND_TRUTH = {
    "fir_001.png": {
        "AADHAAR": ["234567890123"],
        "PAN": ["ABCDE1234F"],
        "PHONE": ["9876543210"],
        "VEHICLE": ["UP32CA1234"],
        "IFSC": ["SBIN0001234"],
    },
    "fir_002.png": {
        "AADHAAR": ["234567890123"],
        "PAN": ["ABCDE1234F"],
        "PHONE": ["9876543210"],
        "VEHICLE": ["UP32CA1234"],
        "IFSC": ["SBIN0001234"],
    },
    "fir_003.png": {
        "AADHAAR": ["234567890123"],
        "PAN": ["ABCDE1234F"],
        "PHONE": ["9876543210"],
        "VEHICLE": ["UP32CA1234"],
        "IFSC": ["SBIN0001234"],
    },
    "fir_004.png": {
        "AADHAAR": ["234567890123"],
        "PAN": ["ABCDE1234F"],
        "PHONE": ["9876543210"],
        "VEHICLE": ["UP32CA1234"],
        "IFSC": ["SBIN0001234"],
    },
    "fir_005.png": {
        "AADHAAR": ["234567890123"],
        "PAN": ["ABCDE1234F"],
        "PHONE": ["9876543210"],
        "VEHICLE": ["UP32CA1234"],
        "IFSC": ["SBIN0001234"],
    },
}


def measure_accuracy():
    """Run accuracy measurement on all sample documents."""
    results = {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "processing_times": [],
        "per_document": {},
    }

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for doc_name, truth in GROUND_TRUTH.items():
        doc_path = SAMPLE_DOCS_DIR / doc_name
        if not doc_path.exists():
            continue

        start_time = time.time()
        imgs = load_document(doc_name)
        ocr = ocr_image(imgs[0])
        pii = scan_text_for_pii(ocr.text)
        elapsed = time.time() - start_time

        detected = {}
        for p in pii:
            if p.label not in detected:
                detected[p.label] = []
            detected[p.label].append(p.text)

        tp = 0
        fp = 0
        fn = 0

        for label, true_values in truth.items():
            detected_values = detected.get(label, [])
            for tv in true_values:
                if tv in detected_values:
                    tp += 1
                else:
                    fn += 1

            for dv in detected_values:
                if dv not in true_values:
                    fp += 1

        total_tp += tp
        total_fp += fp
        total_fn += fn

        doc_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        doc_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        doc_f1 = 2 * doc_precision * doc_recall / (doc_precision + doc_recall) if (doc_precision + doc_recall) > 0 else 0

        results["per_document"][doc_name] = {
            "precision": round(doc_precision, 4),
            "recall": round(doc_recall, 4),
            "f1": round(doc_f1, 4),
            "processing_time": round(elapsed, 2),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        results["processing_times"].append(elapsed)

    results["precision"] = round(total_tp / (total_tp + total_fp), 4) if (total_tp + total_fp) > 0 else 0
    results["recall"] = round(total_tp / (total_tp + total_fn), 4) if (total_tp + total_fn) > 0 else 0
    results["f1"] = round(
        2 * results["precision"] * results["recall"] / (results["precision"] + results["recall"]),
        4,
    ) if (results["precision"] + results["recall"]) > 0 else 0
    results["avg_processing_time"] = round(sum(results["processing_times"]) / len(results["processing_times"]), 2) if results["processing_times"] else 0

    output_path = Path(__file__).parent.parent / "accuracy_report.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Accuracy Report:")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  F1: {results['f1']:.4f}")
    print(f"  Avg Processing Time: {results['avg_processing_time']:.2f}s")
    print(f"  Report saved to: {output_path}")

    return results


if __name__ == "__main__":
    measure_accuracy()
