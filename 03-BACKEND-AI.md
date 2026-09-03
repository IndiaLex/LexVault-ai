# Service 3 — Backend AI

**Stack:** FastAPI (Python), Tesseract/PaddleOCR, spaCy, PyMuPDF, Pillow
**Talks to:** Backend Core only (called by it, never calls it except for callbacks)
**Owns:** Turning a scanned police document into structured, searchable, PII-safe data

---

## 1. Responsibilities

| In scope | Out of scope |
|---|---|
| OCR — extract text from scans and PDFs | Storing documents permanently |
| NER — detect names, places, dates, IDs, phone numbers | Deciding who can see what (RBAC is Core's) |
| PII classification and redaction box generation | Rendering the redaction (frontend overlays it) |
| Document classification (FIR, witness statement, medical report) | Anchoring or hashing for the chain |
| Producing the redacted artifact when asked | User-facing anything |

**Design principle: stateless.** This service receives a pointer to a file,
does work, returns a result. It keeps no permanent database. That makes it
trivially restartable, independently scalable, and — importantly for the
hackathon — buildable with zero dependency on Backend Core.

---

## 2. Architecture

```
app/
├── main.py                FastAPI app
├── config.py              model paths, thresholds, storage config
├── schemas.py             ProcessRequest, AIResult, Entity, RedactionBox
├── routers/
│   ├── ai.py              /ai/process, /ai/redact, /ai/classify
│   └── health.py
├── services/
│   ├── loader.py          fetch file (local sample OR MinIO) + normalize to pages
│   ├── ocr_service.py     text + per-word bounding boxes
│   ├── ner_service.py     spaCy pipeline + regex patterns for Indian PII
│   ├── pii_service.py     entity → redaction decision, confidence scoring
│   ├── redaction_service.py  produce boxes, and optionally a burned-in redacted PDF
│   └── classifier.py      document type detection
└── models/                any fine-tuned model artifacts
```

**Pipeline:**
```
file → loader → page images
                    ↓
                 OCR (text + word boxes)
                    ↓
                 NER (spaCy entities + regex patterns)
                    ↓
                 PII classifier (which entities are sensitive, confidence)
                    ↓
                 box mapper (entity char offsets → page coordinates via OCR word boxes)
                    ↓
                 AIResult { ocr_text, entities, redaction_boxes, doc_class, confidence }
```

**The hard part is the box mapper.** NER gives character offsets in a text
string; the frontend needs page coordinates. OCR must therefore retain per-word
bounding boxes and a char-offset → word index mapping. Build this in Phase 2,
not Phase 4 — it's the piece most likely to eat a day.

---

## 3. PII Detection Design

**spaCy handles:** PERSON, GPE (locations), ORG, DATE, TIME.

**Regex patterns handle what spaCy won't** (these matter for Indian police documents):
- Aadhaar-style 12-digit numbers
- Indian mobile numbers (10 digits, starting 6–9)
- PAN card format
- Vehicle registration numbers
- FIR numbers and section references (these should be *preserved*, not redacted)
- Bank account and IFSC patterns

**Redaction policy** — encode as configuration, not hardcoded:

| Entity type | Default action | Rationale |
|---|---|---|
| PERSON (victim/witness role context) | Redact | Core privacy requirement of the problem statement |
| PERSON (officer) | Preserve | Accountability requires knowing who handled it |
| Aadhaar / PAN / bank | Redact always | Hard identifiers |
| Phone | Redact | |
| Address / GPE | Redact if residential context | Crime scene locations often must be preserved |
| DATE | Preserve | Case chronology |
| FIR number, IPC sections | Preserve | Case identity |

**Confidence handling:** every detection carries a confidence score. Low-confidence
detections are flagged for human review rather than silently redacted or silently
missed. Surface this to Backend Core so the frontend can show "3 items need review."
Never claim automated redaction is perfect — a false negative leaks a victim's name.

---

## 4. Coordinate Contract (agree with Frontend in Phase 0)

Redaction boxes use **normalized coordinates** (0.0–1.0) relative to page
dimensions, plus a page index:

```
{ page: 1, x: 0.12, y: 0.34, width: 0.20, height: 0.03,
  reason: "PERSON", confidence: 0.94, entity_id: "ent-17" }
```

Normalized coordinates survive zoom, DPI differences, and viewer scaling.
Pixel coordinates do not. This decision has bitten every team that skipped it.

---

## 5. Phases

### Phase 1 — Skeleton
- FastAPI boots, `/ai/health` responds.
- `/ai/process` returns a hardcoded `AIResult` matching the contract exactly.
- Sample document folder set up with 5–10 real-looking scanned FIRs.

**Exit:** Backend Core can integrate against a fake response immediately.

### Phase 2 — Core pipeline
- Real OCR working on the sample set, with per-word bounding boxes retained.
- spaCy NER extracting entities from OCR text.
- Char offset → page coordinate mapping working.
- Basic redaction box output.

**Exit:** upload a real sample FIR, get back real text, real entities, and boxes that land in the right place.

### Phase 3 — Integration
- `loader.py` switched from local sample files to MinIO fetch via `storage_key`.
- Handle Backend Core's real requests, real file types, real failures.
- Timeout and error contract agreed — what does Core do when AI fails?

**Exit:** end-to-end upload through Core produces real AI output.

### Phase 4 — Feature completion
- Regex pattern layer for Indian PII (Aadhaar, phone, PAN, vehicle).
- Redaction policy engine with the preserve/redact rules table.
- Confidence scoring and low-confidence review flagging.
- Document classification (FIR / witness statement / medical / other).
- Burned-in redacted PDF generation for sharing outside the system.

**Exit:** the redaction is genuinely useful, not a demo prop.

### Phase 5 — Quality
- Accuracy measurement on the sample set: precision/recall for PII detection. Write the numbers down — judges ask.
- Handle poor scans: deskew, denoise, contrast normalization before OCR.
- Multi-page PDF handling.
- Performance: keep a single document under ~10 seconds.

### Phase 6 — Demo prep
- Pre-warm models at startup so the first request isn't slow.
- Curated demo documents where the pipeline performs well and one where it flags for review — showing the review flow is more honest and more impressive than pretending it's perfect.

---

## 6. Feature Checklist

**Must have**
- [ ] OCR with per-word bounding boxes
- [ ] spaCy NER extraction
- [ ] Char offset → page coordinate mapping
- [ ] Redaction box generation
- [ ] `/ai/process` matching the contract

**Should have**
- [ ] Indian PII regex layer (Aadhaar, phone, PAN, vehicle)
- [ ] Redaction policy engine (preserve vs redact rules)
- [ ] Confidence scoring + review flagging
- [ ] Document classification
- [ ] Image preprocessing for poor scans

**Nice to have**
- [ ] Burned-in redacted PDF export
- [ ] Multilingual OCR (Hindi/Devanagari — high value for the actual problem statement)
- [ ] Fine-tuned NER on Indian names
- [ ] Handwriting recognition for handwritten FIR sections
- [ ] Summary generation per document
- [ ] Cross-document entity linking (same person across multiple files)

---

## 7. Interfaces Provided

```
POST /ai/process
  in:  { document_id, storage_key }
  out: { document_id, ocr_text, entities[], redaction_boxes[], doc_class, confidence, needs_review }

POST /ai/redact
  in:  { document_id, storage_key, boxes[] }
  out: { redacted_storage_key }   # burned-in redacted copy

GET  /ai/health
```

**Async option:** if processing exceeds a few seconds, switch to a job model —
Core posts a job, AI calls back `POST /internal/ai-callback` on Core when done.
Agree this in Phase 3; the endpoint shape above doesn't change, only who waits.

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| OCR accuracy poor on real scans | Curate the sample set in Phase 1; preprocessing in Phase 5; measure and report honestly |
| Box coordinate mapping is fiddly | Normalized coordinates agreed Day 0; build the mapper in Phase 2 while there's time |
| spaCy weak on Indian names | Regex + gazetteer layer; consider a fine-tune if time allows |
| Model load time makes first request slow | Pre-warm at startup |
| Over-claiming redaction accuracy | Show the review flow in the demo — flagged items are a feature, not a failure |
| Hindi/regional documents | Scope explicitly: state English-first, note multilingual as roadmap |
