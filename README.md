# Backend AI Service

Stateless FastAPI microservice for scanned police document processing. Extracts text via OCR, detects PII (Aadhaar, PAN, phone numbers), classifies document type, and returns redaction box coordinates for the frontend to overlay.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI App                           │
│  ┌──────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │ /ai/health│  │  /ai/process     │  │  /ai/redact      │   │
│  └──────────┘  └────────┬─────────┘  └────────┬─────────┘   │
│                         │                      │             │
│  ┌──────────────────────┴──────────────────────┴──────────┐  │
│  │                    Pipeline                            │  │
│  │  ┌─────────┐  ┌───────┐  ┌─────┐  ┌─────┐  ┌──────┐  │  │
│  │  │ Loader  │→ │  OCR  │→ │ NER │→ │ PII │→ │Policy│  │  │
│  │  └─────────┘  └───────┘  └─────┘  └─────┘  └──────┘  │  │
│  │       ↓           ↓          ↓         ↓        ↓      │  │
│  │  ┌─────────┐  ┌───────┐  ┌─────┐  ┌─────┐  ┌──────┐  │  │
│  │  │Preproc. │  │Paddle │  │spaCy│  │Regex│  │Boxes │  │  │
│  │  │         │  │OCR    │  │ NER │  │Scan │  │Mapper│  │  │
│  │  └─────────┘  └───────┘  └─────┘  └─────┘  └──────┘  │  │
│  └────────────────────────────────────────────────────────┘  │
│                         │                                    │
│  ┌──────────────────────┴───────────────────────────────┐   │
│  │                   AIResult                            │   │
│  │  ocr_text, entities[], redaction_boxes[], doc_class   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Pipeline Flow

1. **Load** (`loader.py`): Read document from local filesystem or MinIO. Convert PDF pages to images.
2. **Preprocess** (`preprocessing.py`): Optional deskew, denoise, contrast normalization for poor scans.
3. **OCR** (`ocr_service.py`): Extract text + word bounding boxes. PaddleOCR primary, pytesseract fallback.
4. **NER** (`ner_service.py`): Named Entity Recognition via spaCy (PERSON, GPE, ORG, DATE, LAW).
5. **PII Scan** (`pii_patterns.py`): Regex detection for Indian PII (Aadhaar, PAN, Phone, Vehicle, IFSC).
6. **Classify** (`pii_service.py`): Confidence scoring + role detection (officer vs victim).
7. **Policy** (`redaction_policy.py`): Apply preserve/redact rules per entity type.
8. **Box Map** (`redaction_service.py`): Convert char offsets → normalized page coordinates (0.0–1.0).
9. **Classify Doc** (`classifier.py`): Detect document type (FIR, WITNESS, MEDICAL, OTHER).
10. **Redact** (`ai.py` → `/ai/redact`): Burn black rectangles onto PDF at specified coordinates.

## Project Structure

```
IndiaLex_Backend_AI/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app entry point, model pre-warming
│   ├── config.py               # Model paths, thresholds, storage config
│   ├── schemas.py              # Pydantic models for API request/response
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── health.py           # GET /ai/health endpoint
│   │   └── ai.py               # POST /ai/process, POST /ai/redact endpoints
│   └── services/
│       ├── __init__.py
│       ├── loader.py           # Document loading (local + MinIO)
│       ├── ocr_service.py      # PaddleOCR + pytesseract OCR
│       ├── ner_service.py      # spaCy NER extraction
│       ├── pii_patterns.py     # Indian PII regex patterns
│       ├── pii_service.py      # Confidence scoring + role detection
│       ├── classifier.py       # Document type classification
│       ├── redaction_service.py # Char offset → page coordinate mapping
│       ├── redaction_policy.py # Preserve/redact rules engine
│       ├── preprocessing.py    # Image preprocessing (deskew, denoise)
│       └── minio_client.py     # MinIO client wrapper
├── tests/
│   ├── test_integration.py     # Full pipeline integration tests
│   └── test_accuracy.py        # PII detection accuracy measurement
├── sample_docs/                # Sample FIR documents for testing
├── pyproject.toml              # Dependencies and project config
├── accuracy_report.json        # PII detection accuracy metrics
├── demo.py                     # Demo script showing full pipeline
└── README.md                   # This file
```

## API Endpoints

### `GET /ai/health`

Health check. Returns `{"status": "ok"}`.

### `POST /ai/process`

Process a document through the full OCR + NER + PII pipeline.

**Request:**
```json
{
  "document_id": "fir-001",
  "storage_key": "fir_001.png"
}
```

`storage_key` can be:
- Local file path: `"fir_001.png"` (looks in `sample_docs/`)
- Absolute path: `"C:/path/to/document.pdf"`
- MinIO key: `"minio://bucket/path/to/file.pdf"`

**Response:**
```json
{
  "document_id": "fir-001",
  "ocr_text": "FIRST INFORMATION REPORT...",
  "entities": [
    {
      "text": "234567890123",
      "label": "AADHAAR",
      "start_char": 150,
      "end_char": 162,
      "confidence": 0.95,
      "entity_id": "ent-abc12345",
      "role": null,
      "action": "redact"
    }
  ],
  "redaction_boxes": [
    {
      "page": 0,
      "x": 0.074,
      "y": 0.450,
      "width": 0.200,
      "height": 0.018,
      "reason": "AADHAAR",
      "confidence": 0.95,
      "entity_id": "ent-abc12345"
    }
  ],
  "doc_class": "FIR",
  "confidence": 0.85,
  "needs_review": false
}
```

**Coordinates are normalized (0.0–1.0)** relative to page dimensions, making them resolution-independent.

### `POST /ai/redact`

Generate a burned-in redacted PDF with black rectangles at specified coordinates.

**Request:**
```json
{
  "document_id": "fir-001",
  "storage_key": "fir_001.png",
  "boxes": [
    {
      "page": 0,
      "x": 0.074,
      "y": 0.450,
      "width": 0.200,
      "height": 0.018,
      "reason": "AADHAAR",
      "confidence": 0.95,
      "entity_id": "ent-abc12345"
    }
  ]
}
```

**Response:**
```json
{
  "redacted_storage_key": "redacted_fir-001.pdf"
}
```

## Entity Types Detected

| Type | Label | Regex/Method | Action |
|------|-------|--------------|--------|
| Aadhaar Number | `AADHAAR` | `[2-9][0-9]{11}` | Redact |
| PAN Card | `PAN` | `[A-Z]{5}[0-9]{4}[A-Z]` | Redact |
| Phone Number | `PHONE` | `(+91\|91)?[6-9][0-9]{9}` | Redact |
| Vehicle Reg. | `VEHICLE` | `[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}` | Redact |
| IFSC Code | `IFSC` | `[A-Z]{4}0[A-Z0-9]{6}` | Redact |
| Bank Account | `BANK_ACCOUNT` | `[0-9]{9,18}` | Redact |
| Person Name | `PERSON` | spaCy NER | Redact (victim) / Preserve (officer) |
| Location | `GPE` | spaCy NER | Redact |
| Organization | `ORG` | spaCy NER | Preserve |
| Date | `DATE` | spaCy NER | Preserve |
| Law Section | `LAW` | spaCy NER | Preserve |

## Setup

### Prerequisites

- Python 3.11+
- Tesseract OCR (for pytesseract fallback): https://github.com/tesseract-ocr/tesseract

### Installation

```bash
# Create virtual environment
py -3.11 -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -e ".[dev]"

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Running

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs at: http://localhost:8000/docs

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run accuracy measurement
python tests/test_accuracy.py

# Run demo
python demo.py
```

## Performance

- **Single document**: ~3.4 seconds average
- **OCR accuracy**: 98% confidence on clear scans
- **PII detection**: 80% precision, 80% recall on sample set

## Configuration

Edit `app/config.py` to adjust:

- `CONFIDENCE_THRESHOLD`: Review flagging threshold (default: 0.7)
- `TIMEOUT_SECONDS`: Processing timeout (default: 30)
- `PREPROCESSING_CONFIG`: Enable/disable preprocessing steps
- `STORAGE_CONFIG`: MinIO connection settings

## Sample Documents

The `sample_docs/` directory contains 6 synthetic FIR documents:

| File | Description |
|------|-------------|
| `fir_001.png` | Standard FIR with all PII types |
| `fir_002.png` | FIR with different names |
| `fir_003.png` | FIR with fraud case details |
| `fir_004.png` | FIR with criminal intimidation |
| `fir_005.png` | FIR with assault case |
| `fir_006_poor_quality.png` | Noisy scan for preprocessing test |

Each document contains test PII:
- Aadhaar: `234567890123`
- PAN: `ABCDE1234F`
- Phone: `9876543210`
- Vehicle: `UP32CA1234`
- IFSC: `SBIN0001234`
