# KO Classifier API

FastAPI service for explainable PDF document subcategory classification. The current runtime scope is PDF-based document classification into 11 consolidated document subcategories, with deterministic heuristics always available and optional text and vision LLM augmentation.

The broader category and KO-ingestion policy work is documented under [data_model/category_auto_selection_policy.md](/home/pranav/PyCharm/EU-FarmBook/ko_classifier/data_model/category_auto_selection_policy.md). That policy covers `Document`, `Video`, `Audio`, `Image`, `Dataset`, and `Software Application`, but the `/classify` endpoint in this repository currently classifies PDFs as document subcategories only.

## What The API Does

- Extracts text from PDFs with PyMuPDF.
- Falls back to OCR when extracted text quality is poor.
- Scores the document against 11 consolidated document subcategories using measurable heuristic signals.
- Optionally asks a text LLM and/or a vision LLM to classify the same document.
- Fuses heuristic and LLM probabilities with configurable strategies.
- Returns feature-level evidence, rationale text, and contrastive explanations for top candidates.

## Current Document Subcategories

- `Journal article`
- `Article in conference proceedings`
- `Chapter in edited volume`
- `Thesis`
- `Book`
- `Technical Report`
- `Tutorial`
- `Guide/Manual`
- `Presentation`
- `News & Communication`
- `Informational Booklet`

The consolidation rationale is documented in:

- [data_model/document_subcategories_consolidation.md](/home/pranav/PyCharm/EU-FarmBook/ko_classifier/data_model/document_subcategories_consolidation.md)
- [data_model/subcategories_consolidation_analysis.md](/home/pranav/PyCharm/EU-FarmBook/ko_classifier/data_model/subcategories_consolidation_analysis.md)

## Explainability Model

The heuristic layer uses 27 measurable signals, including:

- academic structure signals such as `imrad_structure`, `citation_density`, `abstract_quality`, `peer_review_markers`
- technical and deliverable signals such as `deliverable_markers`, `version_control`, `technical_specs`
- instructional signals such as `tutorial_structure`, `learning_objectives`, `procedure_steps`, `checklists`
- policy and communication signals such as `news_timeliness`, `press_release_format`, `regulatory_update_markers`, `compliance_language`, `governance_references`
- layout signals such as `slide_indicators`, `visual_heavy`, `short_form`

The source of truth for subcategory criteria is [docint/rubrics/subcategories.py](/home/pranav/PyCharm/EU-FarmBook/ko_classifier/docint/rubrics/subcategories.py). Each subcategory definition carries:

- detectable features
- positive signal hints
- negative signal hints
- close competitors
- minimum features required

Those same criteria are now used in three places:

- heuristic scoring
- contrastive API explanations
- LLM prompting guidance

## API Endpoints

### `POST /classify`

Classifies a PDF document.

Important runtime constraint:

- only `.pdf` files are currently accepted by the API

Query parameters:

- `use_vision`: enable InternVL-style vision classification
- `use_text_llm`: enable text LLM classification
- `heuristics_alpha`: heuristic weight used by weighted fusion
- `fusion_strategy`: `weighted`, `adaptive`, `agreement`, or `cascade`
- `vision_max_pages`: maximum number of pages passed to the vision model
- `ocr_lang`: Tesseract OCR language bundle
- `ocr_max_pages`: maximum pages sent through OCR fallback

Example:

```bash
curl -X POST "http://localhost:8000/classify?use_text_llm=true&fusion_strategy=adaptive" \
  -F "file=@document.pdf"
```

Representative response fields:

- `best_match`: top candidate after heuristics-only scoring or fusion
- `all_candidates`: full ranked list
- `feature_details`: feature-level evidence and excerpts
- `rationale`: direct explanation for the candidate
- `contrastive_rationale`: why the winner beat nearby alternatives
- `fusion`: weights, agreement score, and fusion rationale

## Recommended Fusion Defaults

Recommended production default:

- `fusion_strategy=adaptive`
- `heuristics_alpha=0.5`

Why:

- heuristics remain valuable because they are deterministic and auditable
- LLMs remain valuable because they are stronger on short-form and semantically ambiguous material
- `adaptive` lets the system reweight sources based on confidence instead of relying only on fixed static weights

When to use each strategy:

- `adaptive`: best general default for mixed production traffic
- `weighted`: best for controlled evaluation and reproducible comparisons
- `agreement`: useful when multiple model sources are active and consensus should matter more
- `cascade`: best when latency or inference cost matters and heuristics often resolve the easy cases

Practical guidance for `heuristics_alpha`:

- `0.6`: more conservative and heuristic-led
- `0.5`: balanced default
- `0.4`: more LLM-led, useful for short flyers, newsletters, and visually formatted material

### `GET /subcategories`

Returns the active document subcategories together with their criteria metadata. This is useful for UI configuration, documentation generation, and debugging explainability output.

### `GET /health`

Returns service and model configuration status.

Operational note:

- `/health` is intentionally left unauthenticated so Docker and other health checks can probe the service without Basic Auth credentials

### `GET /docs`

Swagger UI for interactive API testing.

## Local Setup

### Standalone

System packages required for full PDF and OCR support:

```bash
sudo apt-get update
sudo apt-get install -y poppler-utils tesseract-ocr tesseract-ocr-eng
```

Optional:

- install additional Tesseract language packs if multilingual OCR is required

Python setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.sample .env
```

Minimal `.env` for local heuristics-only use:

```bash
HOST=0.0.0.0
PORT=8000
WORKERS=1

DOCINT_AUTH_USERS=
DOCINT_AUTH_PASSWORD=
```

Add these variables only if text LLM classification should be enabled:

```bash
DOCINT_LLM_BASE_URL=https://your-qwen-server.com/v1
DOCINT_LLM_MODEL=qwen3-30b-a3b-awq
DOCINT_LLM_API_KEY=your-key
```

Add these variables only if vision classification should be enabled:

```bash
VISION_LLM_BASE_URL=https://your-internvl-server.com/v1
VISION_LLM_MODEL=internvl3-5-14b
VISION_LLM_API_KEY=your-key
```

Start the service with one of:

```bash
./start_server.sh
```

```bash
python start_server.py
```

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`.

### Docker Compose

The repository includes a local Compose file at [docker-compose.yml](/home/pranav/PyCharm/EU-FarmBook/ko_classifier/docker-compose.yml).

Basic flow:

```bash
cp .env.sample .env
docker compose up --build
```

The service will be exposed on `http://localhost:8000`.

For local testing, leaving `DOCINT_AUTH_USERS` and `DOCINT_AUTH_PASSWORD` empty is the simplest option. If Basic Auth is enabled, `/health` still remains public for container and load-balancer health checks.

### Docker Without Compose

Build:

```bash
docker build -t ko-classifier:local .
```

Run:

```bash
docker run --rm -p 8000:8000 --env-file .env ko-classifier:local
```

## Classifier Behavior Notes

- Heuristics are deterministic and comparatively fast.
- Most latency comes from remote LLM calls, not from local feature extraction.
- `cascade` fusion is the most practical speed-oriented option when heuristics are often decisive.
- The vision model uses a sliding window for longer PDFs.
- Text and vision prompts are aligned with the same criteria vocabulary used by heuristics.

## Project Structure

- [app.py](/home/pranav/PyCharm/EU-FarmBook/ko_classifier/app.py): FastAPI app and response shaping
- [docint/rubrics/subcategories.py](/home/pranav/PyCharm/EU-FarmBook/ko_classifier/docint/rubrics/subcategories.py): active subcategory source of truth
- [docint/rubrics/subcategory_scorer.py](/home/pranav/PyCharm/EU-FarmBook/ko_classifier/docint/rubrics/subcategory_scorer.py): heuristic scoring engine
- [docint/llm/subcategory_classify.py](/home/pranav/PyCharm/EU-FarmBook/ko_classifier/docint/llm/subcategory_classify.py): text and vision LLM classification
- [docint/fusion/intelligent_fusion.py](/home/pranav/PyCharm/EU-FarmBook/ko_classifier/docint/fusion/intelligent_fusion.py): fusion strategies
- [data_model/](/home/pranav/PyCharm/EU-FarmBook/ko_classifier/data_model): taxonomy, consolidation, and category policy documents

## Verification

Quick compile check:

```bash
python3 -m py_compile app.py docint/rubrics/subcategories.py docint/rubrics/subcategory_scorer.py docint/llm/subcategory_classify.py
```

Basic API test script:

```bash
python test_api.py
```
