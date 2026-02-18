# Doc Classifier API v2.0

Evidence-based document subcategory classification with **intelligent LLM fusion**.

## Features

- **Evidence-Based Classification**: 24+ measurable features, 11 subcategories
- **Vision LLM (InternVL)**: Analyzes PDF pages as images (sliding window up to 50 pages)
- **Text LLM (Qwen)**: Analyzes extracted text
- **Intelligent Fusion**: Combines results using confidence-adaptive weighting
- **Configurable Alpha**: Control weight between heuristics (40%) and LLM (60%)

## Architecture

```
PDF Upload
    │
    ├──► PyMuPDF extracts text (or Tesseract OCR if needed)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│              EVIDENCE-BASED CLASSIFICATION                   │
│  • 24+ features extracted                                    │
│  • 11 subcategories scored                                   │
│  • Probabilities normalized                                  │
└─────────────────────────────────────────────────────────────┘
    │
    ├──► [Optional] Vision LLM (InternVL)
    │    • PDF → Images (sliding window: 4 pages, overlap 2)
    │    • Up to 50 pages analyzed
    │    • Returns subcategory probabilities
    │
    └──► [Optional] Text LLM (Qwen)
         • Extracted text analyzed
         • Returns subcategory probabilities
         • Returns: FusionResult
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│              INTELLIGENT FUSION                              │
│  • Weighted combination of all sources                       │
│  • Confidence-adaptive: higher confidence → higher weight    │
│  • Agreement-based: agreeing sources get bonus               │
│  • Configurable: heuristics_alpha parameter                  │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│              RESPONSE                                        │
│  • Fused best match (or heuristics if solo)                 │
│  • Individual source results                                 │
│  • Fusion metadata (weights, agreement, rationale)          │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Configure

```bash
# .env file

# Text LLM (Qwen)
DOCINT_LLM_BASE_URL=https://your-qwen-server.com/
DOCINT_LLM_MODEL=qwen3-30b-a3b-awq
DOCINT_LLM_API_KEY=your-key

# Vision LLM (InternVL)
VISION_LLM_BASE_URL=https://your-internvl-server.com/
VISION_LLM_MODEL=internvl2-8b
VISION_LLM_API_KEY=your-key
```

### 3. Run

```bash
./start_server.sh
# or
python start_server.py
```

## API Usage

### POST /classify

Classify a PDF document with optional LLM fusion.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | required | PDF file |
| `use_vision` | boolean | `false` | Enable Vision LLM |
| `use_text_llm` | boolean | `false` | Enable Text LLM |
| `heuristics_alpha` | float | `0.4` | Weight for heuristics (0.0-1.0) |
| `fusion_strategy` | string | `adaptive` | `weighted`, `adaptive`, `agreement`, `cascade` |
| `vision_max_pages` | int | `20` | Max pages for vision (up to 50) |
| `ocr_lang` | string | all EU | Tesseract languages |
| `ocr_max_pages` | int | `10` | Max pages for OCR fallback |

**Examples:**

```bash
# 1. Heuristics only (fastest)
curl -X POST "http://localhost:8000/classify" \
  -F "file=@document.pdf"

# 2. Heuristics + Text LLM (alpha=0.4: 40% heuristics, 60% LLM)
curl -X POST "http://localhost:8000/classify?use_text_llm=true&heuristics_alpha=0.4" \
  -F "file=@document.pdf"

# 3. All sources with adaptive fusion
curl -X POST "http://localhost:8000/classify?use_vision=true&use_text_llm=true&fusion_strategy=adaptive" \
  -F "file=@document.pdf"

# 4. Vision only (sliding window for long docs)
curl -X POST "http://localhost:8000/classify?use_vision=true&vision_max_pages=30" \
  -F "file=@document.pdf"
```

**Response:**

```json
{
  "best_match": {
    "subcategory_id": "k6VvsRTc",
    "subcategory_name": "Thesis",
    "confidence": 0.72,
    "probability": 0.21,
    "rationale": "Fused result based on..."
  },
  "all_candidates": [...],
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
  "processing_info": {
    "processing_time_ms": 4500,
    "sources_used": ["heuristics", "vision_llm", "text_llm"],
    "fusion_enabled": true
  }
}
```

## Fusion Strategies

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| `weighted` | Static weights (heuristics_alpha) | You trust one source more |
| `adaptive` | Dynamic based on confidence | **Recommended** - default |
| `agreement` | Bonus for agreeing sources | Multiple LLMs available |
| `cascade` | Heuristics first, fallback to LLM | Speed priority |

## Sliding Window Vision

For documents > 8 pages (InternVL limit):

```
Window 1: pages 1-4
Window 2: pages 3-6  (overlap 2)
Window 3: pages 5-8  (overlap 2)
...
```

Results from all windows are **weighted-averaged** by confidence.

## Subcategories (11)

| ID | Name | Parent Type |
|----|------|-------------|
| Zyvdw7E2 | Journal article | scientific_research |
| arQwir9z | Conference proceedings | scientific_research |
| P3nzEsdB | Book chapter | scientific_research |
| k6VvsRTc | Thesis | scientific_research |
| NBq4fMG2 | Book | scientific_research |
| CONSOLIDATED_TECH_REPORT | Technical Report | deliverable_report |
| 4NLQdUhM | Tutorial | educational |
| CONSOLIDATED_GUIDE_MANUAL | Guide/Manual | practice_oriented |
| CONSOLIDATED_PRESENTATION | Presentation | educational |
| CONSOLIDATED_NEWS_COMM | News & Communication | policy_guidance |
| CONSOLIDATED_INFO_BOOKLET | Informational Booklet | practice_oriented |

## Alpha Parameter Guide

| heuristics_alpha | Heuristics | LLM | Use Case |
|------------------|------------|-----|----------|
| 0.0 | 0% | 100% | Trust LLM completely |
| 0.3 | 30% | 70% | LLM preferred |
| **0.4** | **40%** | **60%** | **Balanced (default)** |
| 0.5 | 50% | 50% | Equal weight |
| 0.7 | 70% | 30% | Heuristics preferred |
| 1.0 | 100% | 0% | Heuristics only |

## Other Endpoints

```bash
# Health check
curl http://localhost:8000/health

# List subcategories
curl http://localhost:8000/subcategories

# Test LLM connections
curl "http://localhost:8000/test-llm?model_type=text"
curl "http://localhost:8000/test-llm?model_type=vision"
```

## Testing

```bash
python test_api.py
```

## License

EU-FarmBook Project
