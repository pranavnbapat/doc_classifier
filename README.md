# Doc Classifier API v2.0

Production-ready document subcategory classification API with evidence-based heuristics, dual LLM support (text and vision), and intelligent fusion.

## Features

- **Evidence-Based Classification**: 24+ measurable features, 11 subcategories, deterministic scoring
- **Text LLM (Qwen)**: Analyzes extracted text for semantic understanding
- **Vision LLM (InternVL)**: Analyzes PDF pages as images using sliding window (up to 50 pages)
- **Intelligent Fusion**: Combines multiple sources using confidence-adaptive weighting
- **Basic Auth Protection**: Secure access with Docker-style usernames
- **Full EU Language Support**: All Tesseract OCR languages

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.sample .env
```

Edit `.env`:

```bash
# Basic Auth (Docker-style usernames)
DOCINT_AUTH_USERS=
DOCINT_AUTH_PASSWORD=

# Text LLM (Qwen)
DOCINT_LLM_BASE_URL=https://your-qwen-server.com/v1
DOCINT_LLM_MODEL=qwen3-30b-a3b-awq
DOCINT_LLM_API_KEY=your-key

# Vision LLM (InternVL)
VISION_LLM_BASE_URL=https://your-internvl-server.com/v1
VISION_LLM_MODEL=internvl2-8b
VISION_LLM_API_KEY=your-key
```

### 3. Run

```bash
./start_server.sh
# Server starts at http://localhost:8000
```

### Browser Access

Visit `http://localhost:8000/docs` - browser will prompt for username/password.

## API Endpoints

### POST /classify

Classify a PDF document.

**Authentication**: Required (Basic Auth)

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | required | PDF file to classify |
| `use_vision` | boolean | `false` | Enable Vision LLM (InternVL) |
| `use_text_llm` | boolean | `false` | Enable Text LLM (Qwen) |
| `heuristics_alpha` | float | `0.4` | Weight for heuristics (0.0-1.0) |
| `fusion_strategy` | string | `adaptive` | `weighted`, `adaptive`, `agreement`, `cascade` |
| `vision_max_pages` | int | `20` | Max pages for vision analysis (1-50) |
| `ocr_lang` | string | all EU | Tesseract OCR languages |
| `ocr_max_pages` | int | `10` | Max pages for OCR fallback |

**Examples:**

```bash
# Heuristics only (fastest)
curl -X POST "http://localhost:8000/classify" \
  -F "file=@document.pdf"

# Heuristics + Text LLM with 40/60 split
curl -X POST "http://localhost:8000/classify?use_text_llm=true&heuristics_alpha=0.4" \
  -F "file=@document.pdf"

# Heuristics + Vision LLM (sliding window)
curl -X POST "http://localhost:8000/classify?use_vision=true&vision_max_pages=20" \
  -F "file=@document.pdf"

# All sources with adaptive fusion
curl -X POST "http://localhost:8000/classify?use_vision=true&use_text_llm=true&fusion_strategy=adaptive" \
  -F "file=@document.pdf"
```

**Response:**

```json
{
  "best_match": {
    "subcategory_id": "k6VvsRTc",
    "subcategory_name": "Thesis",
    "parent_type": "scientific_research",
    "confidence": 0.72,
    "probability": 0.21,
    "evidence_score": 0.42,
    "features_found": ["thesis_markers", "imrad_structure", "citation_density"],
    "rationale": "Thesis/Dissertation confirmed via thesis_markers, imrad_structure, citation_density..."
  },
  "all_candidates": [
    {
      "subcategory_id": "k6VvsRTc",
      "subcategory_name": "Thesis",
      "confidence": 0.72,
      "probability": 0.21,
      "rank": 1
    },
    {
      "subcategory_id": "NBq4fMG2",
      "subcategory_name": "Book",
      "confidence": 0.53,
      "probability": 0.15,
      "rank": 2
    }
  ],
  "fusion": {
    "fused": true,
    "strategy": "adaptive",
    "weights": {
      "heuristics": 0.35,
      "text_llm": 0.45,
      "vision_llm": 0.20
    },
    "agreement_score": 0.85,
    "rationale": "Sources agree on Thesis; confidence-adaptive weighting applied"
  },
  "heuristics": {
    "subcategory_name": "Thesis",
    "confidence": 0.60,
    "features_found": ["thesis_markers", "imrad_structure"]
  },
  "vision_llm": {
    "subcategory_name": "Thesis",
    "confidence": 0.95,
    "rationale": "Document shows academic structure..."
  },
  "text_llm": {
    "subcategory_name": "Book",
    "confidence": 0.70,
    "rationale": "Long-form academic content..."
  },
  "document_info": {
    "filename": "document.pdf",
    "pages": 45,
    "source": "pdf_text",
    "text_length": 45231
  },
  "processing_info": {
    "processing_time_ms": 8500,
    "sources_used": ["heuristics", "vision_llm", "text_llm"],
    "fusion_enabled": true
  }
}
```

### GET /health

Health check endpoint.

```bash
curl http://localhost:8000/health
```

### GET /subcategories

List all subcategory definitions.

```bash
curl http://localhost:8000/subcategories
```

### GET /docs

Interactive Swagger UI documentation.

Visit: `http://localhost:8000/docs`

## Fusion Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `weighted` | Static weights based on `heuristics_alpha` | Known trust levels |
| `adaptive` | Dynamic weights based on confidence | **Recommended default** |
| `agreement` | Bonus weight for sources that agree | Multiple LLMs available |
| `cascade` | Heuristics first, LLM fallback | Speed priority |

## Heuristics Alpha Guide

| Alpha | Heuristics | LLM | Use Case |
|-------|------------|-----|----------|
| 0.0 | 0% | 100% | Trust LLM completely |
| 0.3 | 30% | 70% | LLM preferred |
| **0.4** | **40%** | **60%** | **Balanced (default)** |
| 0.5 | 50% | 50% | Equal weight |
| 0.7 | 70% | 30% | Heuristics preferred |
| 1.0 | 100% | 0% | Heuristics only |

## 11 Subcategories

| ID | Name | Parent Type | Detected By |
|----|------|-------------|-------------|
| Zyvdw7E2 | Journal article | scientific_research | IMRaD structure, peer review markers, citations |
| arQwir9z | Conference proceedings | scientific_research | Conference markers, IMRaD structure |
| P3nzEsdB | Book chapter | scientific_research | Book features, citations |
| k6VvsRTc | Thesis | scientific_research | University markers, formal structure |
| NBq4fMG2 | Book | scientific_research | ISBN, publisher, chapters |
| CONSOLIDATED_TECH_REPORT | Technical Report | deliverable_report | Deliverable markers, version control |
| 4NLQdUhM | Tutorial | educational | Tutorial structure, learning objectives |
| CONSOLIDATED_GUIDE_MANUAL | Guide/Manual | practice_oriented | Procedure steps, materials, safety |
| CONSOLIDATED_PRESENTATION | Presentation | educational | Slide indicators, visual layout |
| CONSOLIDATED_NEWS_COMM | News & Communication | policy_guidance | News timeliness, press format |
| CONSOLIDATED_INFO_BOOKLET | Informational Booklet | practice_oriented | Short form, promotional content |

## Vision Model Sliding Window

For documents > 8 pages (InternVL limit), the vision model uses sliding windows:

```
Window 1: pages 1-4
Window 2: pages 3-6  (overlap 2)
Window 3: pages 5-8  (overlap 2)
Window 4: pages 7-10
...
```

Results from all windows are weighted-averaged by confidence.

## Testing

```bash
python test_api.py
```

Runs comprehensive tests including:
- Auth rejection without credentials
- Auth acceptance with valid credentials
- Heuristics-only classification
- Text LLM fusion
- Vision LLM fusion
- Adaptive fusion with multiple sources

## Performance

| Operation | Typical Time |
|-----------|--------------|
| Text extraction | 50-200ms |
| OCR fallback (if needed) | 1-5s |
| Heuristics scoring | 50-100ms |
| Text LLM | 2-5s |
| Vision LLM | 5-15s |
| Fusion | < 10ms |

## Project Structure

```
doc_classifier/
├── app.py                    # FastAPI application
├── requirements.txt          # Dependencies
├── .env                      # Configuration (not committed)
├── docint/                   # Core library
│   ├── extract/              # PDF/OCR extraction
│   ├── features/             # Feature extraction
│   ├── rubrics/              # Scoring rubrics
│   ├── llm/                  # LLM integration
│   └── fusion/               # Intelligent fusion
├── data_model/               # Subcategory definitions
└── files/                    # Test files
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system design.

## License

EU-FarmBook Project
