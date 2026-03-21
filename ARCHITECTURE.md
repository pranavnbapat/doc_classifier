# KO Classifier Architecture

## Scope

This service currently implements PDF-based document subcategory classification. It does not yet enforce the broader KO category-selection policy in runtime. That policy work, including `Document`, `Video`, `Audio`, `Image`, `Dataset`, and `Software Application`, is documented separately in:

- [data_model/category_auto_selection_policy.md](/home/pranav/PyCharm/EU-FarmBook/ko_classifier/data_model/category_auto_selection_policy.md)
- [data_model/subcategories_consolidation_analysis.md](/home/pranav/PyCharm/EU-FarmBook/ko_classifier/data_model/subcategories_consolidation_analysis.md)

The live API path in this repository accepts PDFs and classifies them into 11 consolidated document subcategories.

## High-Level Flow

```text
Client PDF upload
  -> FastAPI request handling
  -> PDF text extraction
  -> OCR fallback if text quality is poor
  -> heuristic feature extraction and scoring
  -> optional text LLM classification
  -> optional vision LLM classification
  -> probability fusion
  -> ranked response with evidence and contrastive rationale
```

## Runtime Pipeline

### 1. Ingress

- Entry point: [app.py](/home/pranav/PyCharm/EU-FarmBook/ko_classifier/app.py)
- Endpoint: `POST /classify`
- Current file constraint: `.pdf` only
- Authentication: Basic Auth when `DOCINT_AUTH_USERS` and `DOCINT_AUTH_PASSWORD` are configured
- Exception: `GET /health` is intentionally unauthenticated so container and platform health probes can work without credentials

### 2. Text Extraction

- Primary path: [docint/extract/pdf_text.py](/home/pranav/PyCharm/EU-FarmBook/ko_classifier/docint/extract/pdf_text.py)
- Quality check: [docint/extract/quality.py](/home/pranav/PyCharm/EU-FarmBook/ko_classifier/docint/extract/quality.py)
- OCR fallback: [docint/extract/ocr.py](/home/pranav/PyCharm/EU-FarmBook/ko_classifier/docint/extract/ocr.py)

The service first tries direct PDF text extraction. If the resulting text does not meet the quality threshold, it falls back to OCR.

### 3. Feature Extraction

The heuristic layer is built around measurable signals rather than opaque labels. Core extraction helpers live under:

- [docint/features/sections.py](/home/pranav/PyCharm/EU-FarmBook/ko_classifier/docint/features/sections.py)
- [docint/features/citations.py](/home/pranav/PyCharm/EU-FarmBook/ko_classifier/docint/features/citations.py)
- [docint/features/keywords.py](/home/pranav/PyCharm/EU-FarmBook/ko_classifier/docint/features/keywords.py)

Rubric-level scoring modules include:

- [docint/rubrics/imrad.py](/home/pranav/PyCharm/EU-FarmBook/ko_classifier/docint/rubrics/imrad.py)
- [docint/rubrics/citations.py](/home/pranav/PyCharm/EU-FarmBook/ko_classifier/docint/rubrics/citations.py)
- [docint/rubrics/deliverable.py](/home/pranav/PyCharm/EU-FarmBook/ko_classifier/docint/rubrics/deliverable.py)
- [docint/rubrics/pedagogy.py](/home/pranav/PyCharm/EU-FarmBook/ko_classifier/docint/rubrics/pedagogy.py)
- [docint/rubrics/procedure.py](/home/pranav/PyCharm/EU-FarmBook/ko_classifier/docint/rubrics/procedure.py)

The active heuristic signal set currently covers 27 features, including:

- academic signals such as `imrad_structure`, `citation_density`, `abstract_quality`, `peer_review_markers`
- technical signals such as `deliverable_markers`, `version_control`, `technical_specs`, `formal_structure`
- instructional signals such as `tutorial_structure`, `learning_objectives`, `exercises_assessments`, `procedure_steps`
- communication and policy signals such as `news_timeliness`, `press_release_format`, `regulatory_update_markers`, `compliance_language`, `governance_references`
- layout signals such as `slide_indicators`, `visual_heavy`, `short_form`

### 4. Heuristic Scoring

- Scorer: [docint/rubrics/subcategory_scorer.py](/home/pranav/PyCharm/EU-FarmBook/ko_classifier/docint/rubrics/subcategory_scorer.py)
- Criteria source of truth: [docint/rubrics/subcategories.py](/home/pranav/PyCharm/EU-FarmBook/ko_classifier/docint/rubrics/subcategories.py)

Each subcategory definition carries:

- detectable features
- positive signal hints
- negative signal hints
- close competitors
- minimum features required

The scorer uses those definitions together with feature-level evidence to produce:

- evidence score
- confidence
- normalized candidate probabilities
- per-feature excerpts
- base rationale text

Recent explainability-oriented changes include:

- feature caching to avoid repeated extractor work
- stronger policy and regulatory-update signals
- reduced phantom evidence for thesis and conference detection
- subcategory-specific bonuses and penalties for close competitors

### 5. Optional Text LLM Classification

- Module: [docint/llm/subcategory_classify.py](/home/pranav/PyCharm/EU-FarmBook/ko_classifier/docint/llm/subcategory_classify.py)
- Intended model family: Qwen or any OpenAI-compatible text endpoint

The text LLM path receives extracted document text and is now prompted with the same criteria vocabulary used by heuristics:

- detectable features
- positive signals
- negative signals
- close competitors
- minimum evidence expectation

The prompt also asks the model to report:

- matched signals
- conflicting signals
- closest alternative

This keeps the LLM output closer to the auditable heuristic taxonomy instead of allowing purely free-form reasoning.

### 6. Optional Vision LLM Classification

- Module: [docint/llm/subcategory_classify.py](/home/pranav/PyCharm/EU-FarmBook/ko_classifier/docint/llm/subcategory_classify.py)
- Intended model family: InternVL or any OpenAI-compatible vision endpoint

PDF pages are converted to images and passed to the vision model. For longer documents, the service uses a sliding-window approach and then combines window-level probability distributions.

### 7. Fusion

- Module: [docint/fusion/intelligent_fusion.py](/home/pranav/PyCharm/EU-FarmBook/ko_classifier/docint/fusion/intelligent_fusion.py)

Supported strategies:

- `weighted`
- `adaptive`
- `agreement`
- `cascade`

Fusion operates on normalized source probability distributions and returns:

- fused best match
- fused ranking
- source weights
- agreement score
- fusion rationale

## Active Document Taxonomy

The runtime taxonomy currently contains 11 consolidated document subcategories:

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

The underlying consolidation rationale is documented in:

- [data_model/document_subcategories_consolidation.md](/home/pranav/PyCharm/EU-FarmBook/ko_classifier/data_model/document_subcategories_consolidation.md)
- [data_model/data_model.subcategories_document_consolidated.json](/home/pranav/PyCharm/EU-FarmBook/ko_classifier/data_model/data_model.subcategories_document_consolidated.json)

## Explainability Outputs

The API response includes both direct and contrastive explanation layers.

Candidate-level fields include:

- `features_found`
- `feature_details`
- `rationale`
- `contrastive_rationale`

`contrastive_rationale` is generated in [app.py](/home/pranav/PyCharm/EU-FarmBook/ko_classifier/app.py) by comparing the winner with nearby alternatives using the same criteria metadata from [docint/rubrics/subcategories.py](/home/pranav/PyCharm/EU-FarmBook/ko_classifier/docint/rubrics/subcategories.py).

The `/subcategories` endpoint also exposes criteria metadata so that documentation and UIs can stay aligned with runtime definitions.

## Current Architectural Boundaries

### What Is Implemented

- PDF upload and classification
- heuristic scoring with measurable signals
- criteria-backed contrastive explanations
- optional text LLM and vision LLM classification
- fusion across sources

### What Is Documented But Not Yet Enforced In Runtime

- category auto-selection by KO mode
- category filtering for file-based versus URL-based KOs
- `Software Application` availability only for URL-based KOs
- broader non-document category handling such as `Dataset`, `Video`, `Audio`, and `Image`

Those rules are part of the taxonomy and ingestion design work, not the current `/classify` implementation.

## Deployment Paths

### Standalone

- install Python dependencies from [requirements.txt](/home/pranav/PyCharm/EU-FarmBook/ko_classifier/requirements.txt)
- install `poppler-utils`, `tesseract-ocr`, and required language packs
- configure `.env`
- run via [start_server.sh](/home/pranav/PyCharm/EU-FarmBook/ko_classifier/start_server.sh), [start_server.py](/home/pranav/PyCharm/EU-FarmBook/ko_classifier/start_server.py), or `uvicorn`

### Docker

- build and run with [Dockerfile](/home/pranav/PyCharm/EU-FarmBook/ko_classifier/Dockerfile)
- local Compose path available in [docker-compose.yml](/home/pranav/PyCharm/EU-FarmBook/ko_classifier/docker-compose.yml)

## File Map

- [app.py](/home/pranav/PyCharm/EU-FarmBook/ko_classifier/app.py): API entry point and response shaping
- [docint/rubrics/subcategories.py](/home/pranav/PyCharm/EU-FarmBook/ko_classifier/docint/rubrics/subcategories.py): criteria source of truth
- [docint/rubrics/subcategory_scorer.py](/home/pranav/PyCharm/EU-FarmBook/ko_classifier/docint/rubrics/subcategory_scorer.py): heuristic scoring engine
- [docint/llm/subcategory_classify.py](/home/pranav/PyCharm/EU-FarmBook/ko_classifier/docint/llm/subcategory_classify.py): text and vision LLM integration
- [data_model/](/home/pranav/PyCharm/EU-FarmBook/ko_classifier/data_model): taxonomy consolidation and category policy documents
