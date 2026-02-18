# Doc Classifier Architecture v2.0

## Overview

A production-ready document classification API that combines evidence-based heuristics with optional LLM analysis (both text and vision models) using intelligent fusion algorithms.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CLIENT REQUEST                                   │
│  POST /classify with PDF + Basic Auth                                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      AUTHENTICATION MIDDLEWARE                           │
│  • Basic HTTP Auth with Docker-style usernames                          │
│  • Returns 401 with WWW-Authenticate header if missing/invalid          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      DOCUMENT PROCESSING PIPELINE                        │
│                                                                          │
│  1. PDF UPLOAD → TEMP STORAGE                                            │
│                                                                          │
│  2. TEXT EXTRACTION (Parallel paths)                                     │
│     ┌─────────────────────┐  ┌─────────────────────┐                    │
│     │   PyMuPDF (fast)    │  │  Tesseract OCR      │                    │
│     │   Primary method    │──│  Fallback if poor   │                    │
│     │   Extracts text     │  │  text quality       │                    │
│     └─────────────────────┘  └─────────────────────┘                    │
│                                                                          │
│  3. FEATURE EXTRACTION (24+ signals)                                     │
│     • Section detection (IMRaD, headings)                               │
│     • Citation patterns (numeric, author-year, DOI)                     │
│     • Keyword buckets (deliverable, educational, etc.)                  │
│     • Visual features (short lines, layout)                             │
│                                                                          │
│  4. EVIDENCE-BASED SCORING                                               │
│     • Score all 11 subcategories                                        │
│     • Weighted feature combination                                      │
│     • Confidence + probability calculation                              │
│     • Rationale generation with excerpts                                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
┌───────────────────────┐ ┌─────────────────┐ ┌─────────────────────────┐
│   EVIDENCE-BASED      │ │   TEXT LLM      │ │   VISION LLM            │
│   (Always runs)       │ │   (Qwen)        │ │   (InternVL)            │
│                       │ │   Optional      │ │   Optional              │
│ • Heuristic scoring   │ │                 │ │                         │
│ • Feature-based       │ │ • Analyzes      │ │ • Converts PDF to       │
│ • Measurable signals  │ │   extracted     │ │   images                │
│ • Fast (< 100ms)      │ │   text          │ │ • Sliding window        │
│                       │ │ • 11 categories │ │   (4 pages, overlap 2)  │
│                       │ │ • Rationale     │ │ • Sees layout/images    │
│                       │ │ • ~2-4s         │ │ • ~5-15s                │
└───────────────────────┘ └─────────────────┘ └─────────────────────────┘
                    │               │               │
                    └───────────────┴───────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      INTELLIGENT FUSION                                  │
│                                                                          │
│  When multiple sources enabled:                                         │
│                                                                          │
│  1. SOURCE RESULTS AGGREGATION                                           │
│     • Collect probabilities from each source                            │
│     • Track confidence scores                                           │
│                                                                          │
│  2. FUSION STRATEGY (configurable)                                       │
│     ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐         │
│     │   WEIGHTED      │ │   ADAPTIVE      │ │   AGREEMENT     │         │
│     │   Static alpha  │ │   Confidence    │ │   Bonus for     │         │
│     │   40%/60%       │ │   weighting     │ │   agreeing      │         │
│     │   (default)     │ │   (recommended) │ │   sources       │         │
│     └─────────────────┘ └─────────────────┘ └─────────────────┘         │
│                                                                          │
│  3. PROBABILITY FUSION                                                   │
│     • Normalize all source probabilities                                │
│     • Apply fusion weights                                              │
│     • Re-normalize to sum to 1.0                                        │
│                                                                          │
│  4. AGREEMENT CALCULATION                                                │
│     • Jaccard similarity of top-k predictions                           │
│     • 0.0 = complete disagreement, 1.0 = perfect agreement              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         API RESPONSE                                     │
│                                                                          │
│  {                                                                        │
│    "best_match": {         ← Fused or heuristics-only result            │
│      "subcategory_id": "...",                                           │
│      "subcategory_name": "...",                                         │
│      "confidence": 0.72,                                                │
│      "probability": 0.21,                                               │
│      "rationale": "..."                                                 │
│    },                                                                     │
│    "all_candidates": [...],  ← All 11 subcategories with scores         │
│    "fusion": {               ← If fusion occurred                       │
│      "fused": true,                                                     │
│      "strategy": "adaptive",                                            │
│      "weights": {"heuristics": 0.4, "text_llm": 0.6},                  │
│      "agreement_score": 0.85                                           │
│    },                                                                     │
│    "heuristics": {...},      ← Individual source results                │
│    "vision_llm": {...},                                                 │
│    "text_llm": {...},                                                   │
│    "processing_info": {                                                  │
│      "processing_time_ms": 4500,                                        │
│      "sources_used": [...],                                             │
│      "fusion_enabled": true                                             │
│    }                                                                      │
│  }                                                                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Data Model

### 11 Subcategories (from data_model.subcategories_document_consolidated.json)

| ID | Name | Parent Type | Key Features |
|----|------|-------------|--------------|
| Zyvdw7E2 | Journal article | scientific_research | IMRaD, peer review, citations |
| arQwir9z | Conference proceedings | scientific_research | Conference markers, IMRaD |
| P3nzEsdB | Book chapter | scientific_research | Book features, citations |
| k6VvsRTc | Thesis | scientific_research | Thesis markers, formal structure |
| NBq4fMG2 | Book | scientific_research | ISBN, chapters, publisher |
| CONSOLIDATED_TECH_REPORT | Technical Report | deliverable_report | Deliverable markers, version control |
| 4NLQdUhM | Tutorial | educational | Tutorial structure, learning objectives |
| CONSOLIDATED_GUIDE_MANUAL | Guide/Manual | practice_oriented | Procedure steps, materials, safety |
| CONSOLIDATED_PRESENTATION | Presentation | educational | Slide indicators, visual-heavy |
| CONSOLIDATED_NEWS_COMM | News & Communication | policy_guidance | News timeliness, press format |
| CONSOLIDATED_INFO_BOOKLET | Informational Booklet | practice_oriented | Short form, promotional |

## File Structure

```
doc_classifier/
│
├── app.py                          # FastAPI application, main entry point
├── start_server.sh                 # Bash startup script
├── start_server.py                 # Python startup script
├── test_api.py                     # API test suite
├── requirements.txt                # Python dependencies
├── .env                            # Environment configuration (not committed)
├── .env.example                    # Environment template
├── .gitignore                      # Git ignore rules
├── README.md                       # User documentation
├── ARCHITECTURE.md                 # This file
│
├── data_model/                     # Data model definitions
│   ├── data_model.subcategories_document_consolidated.json
│   └── document_subcategories_consolidation.md
│
├── docint/                         # Core library
│   ├── __init__.py
│   │
│   ├── extract/                    # Document extraction
│   │   ├── pdf_text.py            # PyMuPDF text extraction
│   │   ├── ocr.py                 # Tesseract OCR fallback
│   │   └── quality.py             # Text quality assessment
│   │
│   ├── features/                   # Feature extraction
│   │   ├── sections.py            # Section heading detection
│   │   ├── citations.py           # Citation pattern detection
│   │   ├── keywords.py            # Keyword bucket matching
│   │   └── doccontrol.py          # Document control features
│   │
│   ├── rubrics/                    # Scoring rubrics
│   │   ├── subcategories.py       # Subcategory definitions
│   │   ├── subcategory_scorer.py  # Evidence-based scoring engine
│   │   ├── imrad.py               # IMRaD structure scoring
│   │   ├── citations.py           # Citation strength scoring
│   │   ├── deliverable.py         # Deliverable rubric
│   │   ├── pedagogy.py            # Educational content scoring
│   │   └── procedure.py           # Procedure content scoring
│   │
│   ├── llm/                        # LLM integration
│   │   └── subcategory_classify.py # Vision & text LLM classification
│   │
│   └── fusion/                     # Fusion algorithms
│       └── intelligent_fusion.py  # Weighted/adaptive/agreement fusion
│
├── files/                          # Sample/test files (not committed)
│
└── old/                            # Deprecated files
    └── (moved old implementations)
```

## Configuration

### Environment Variables (.env)

```bash
# Basic Auth
DOCINT_AUTH_USERS=nifty_chandrasekhar,jolly_poincare,quirky_roentgen,dreamy_agnesi,festive_hypatia,zen_swartz
DOCINT_AUTH_PASSWORD=3C11TCYVnqXJ

# Text LLM (Qwen)
DOCINT_LLM_BASE_URL=https://your-qwen-server.com/v1
DOCINT_LLM_MODEL=qwen3-30b-a3b-awq
DOCINT_LLM_API_KEY=your-key

# Vision LLM (InternVL)
VISION_LLM_BASE_URL=https://your-internvl-server.com/v1
VISION_LLM_MODEL=internvl2-8b
VISION_LLM_API_KEY=your-key
```

## API Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/` | GET | Required | API info |
| `/health` | GET | Required | Health check with model status |
| `/docs` | GET | Required | Swagger UI documentation |
| `/redoc` | GET | Required | ReDoc documentation |
| `/subcategories` | GET | Required | List all subcategories |
| `/classify` | POST | Required | Main classification endpoint |

## Performance Characteristics

| Component | Typical Time | Notes |
|-----------|--------------|-------|
| Text extraction | 50-200ms | PyMuPDF, depends on PDF size |
| OCR fallback | 1-5s | Tesseract, only if needed |
| Heuristics scoring | 50-100ms | Fast, deterministic |
| Text LLM | 2-5s | Depends on text length |
| Vision LLM | 5-15s | Sliding window, parallel processing |
| Fusion | < 10ms | Fast computation |

## Security

- **Basic HTTP Auth**: All endpoints protected
- **Timing-attack safe**: `secrets.compare_digest()` for password comparison
- **Environment-based config**: No secrets in code
- **Input validation**: File type, size limits
- **Temp file cleanup**: Automatic removal after processing

## Deployment

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Start server
./start_server.sh
# or
python start_server.py
# or
uvicorn app:app --host 0.0.0.0 --port 8000
```
