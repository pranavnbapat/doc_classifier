# KO Classifier Architecture

## Scope

This service currently implements PDF-based document subcategory classification. It does not yet enforce the broader KO category-selection policy in runtime. That policy work, including `Document`, `Video`, `Audio`, `Image`, `Dataset`, and `Software Application`, is documented separately in:

- [category_auto_selection_policy.md](/home/pranav/PyCharm/EU-FarmBook/doc_classifier/data_model/category_auto_selection_policy.md)
- [subcategories_consolidation_analysis.md](/home/pranav/PyCharm/EU-FarmBook/doc_classifier/data_model/subcategories_consolidation_analysis.md)

The live API path in this repository accepts PDFs and classifies them into 11 consolidated document subcategories.

## High-Level Flow

```text
Client PDF upload
  -> FastAPI request handling
  -> PDF text extraction
  -> OCR fallback if text quality is poor
  -> heuristic feature extraction and scoring
  -> agriculture relevance gate
  -> heuristic subcategory scoring
  -> text LLM classification for agri docs when enabled
  -> selective vision classification when routing decides it is needed
  -> probability fusion
  -> ranked response with evidence and contrastive rationale
```

## Runtime Pipeline

### 1. Ingress

- Entry point: [app.py](/home/pranav/PyCharm/EU-FarmBook/doc_classifier/app.py)
- Endpoint: `POST /classify`
- Current file constraint: `.pdf` only
- Authentication: Basic Auth when `DOCINT_AUTH_USERS` and `DOCINT_AUTH_PASSWORD` are configured
- Exception: `GET /health` is intentionally unauthenticated so container and platform health probes can work without credentials

### 2. Text Extraction

- Primary path: [pdf_text.py](/home/pranav/PyCharm/EU-FarmBook/doc_classifier/docint/extract/pdf_text.py)
- Quality check: [quality.py](/home/pranav/PyCharm/EU-FarmBook/doc_classifier/docint/extract/quality.py)
- OCR fallback: [ocr.py](/home/pranav/PyCharm/EU-FarmBook/doc_classifier/docint/extract/ocr.py)

The service first tries direct PDF text extraction. If the resulting text does not meet the quality threshold, it falls back to OCR.

### 3. Agriculture Relevance Gate

- Module: [agriculture_pipeline.py](/home/pranav/PyCharm/EU-FarmBook/doc_classifier/docint/domain/agriculture_pipeline.py)
- Lexicon source: [agriculture_lexicon.json](/home/pranav/PyCharm/EU-FarmBook/doc_classifier/data_model/agriculture_lexicon.json)

The service now performs agriculture relevance assessment before document-subcategory classification:

- Stage 1: AGROVOC-style multilingual lexicon
- Stage 2: local multilingual embedding model when available
- Stage 3: text LLM fallback only for ambiguous cases

If `require_agriculture=true` and the file is assessed as non-agriculture, the API returns early and skips subcategory classification.

### 4. Feature Extraction

The heuristic layer is built around measurable signals rather than opaque labels. Core extraction helpers live under:

- [sections.py](/home/pranav/PyCharm/EU-FarmBook/doc_classifier/docint/features/sections.py)
- [citations.py](/home/pranav/PyCharm/EU-FarmBook/doc_classifier/docint/features/citations.py)
- [keywords.py](/home/pranav/PyCharm/EU-FarmBook/doc_classifier/docint/features/keywords.py)

Rubric-level scoring modules include:

- [imrad.py](/home/pranav/PyCharm/EU-FarmBook/doc_classifier/docint/rubrics/imrad.py)
- [citations.py](/home/pranav/PyCharm/EU-FarmBook/doc_classifier/docint/rubrics/citations.py)
- [deliverable.py](/home/pranav/PyCharm/EU-FarmBook/doc_classifier/docint/rubrics/deliverable.py)
- [pedagogy.py](/home/pranav/PyCharm/EU-FarmBook/doc_classifier/docint/rubrics/pedagogy.py)
- [procedure.py](/home/pranav/PyCharm/EU-FarmBook/doc_classifier/docint/rubrics/procedure.py)

The active heuristic signal set currently covers 27 features, including:

- academic signals such as `imrad_structure`, `citation_density`, `abstract_quality`, `peer_review_markers`
- technical signals such as `deliverable_markers`, `version_control`, `technical_specs`, `formal_structure`
- instructional signals such as `tutorial_structure`, `learning_objectives`, `exercises_assessments`, `procedure_steps`
- communication and policy signals such as `news_timeliness`, `press_release_format`, `regulatory_update_markers`, `compliance_language`, `governance_references`
- layout signals such as `slide_indicators`, `visual_heavy`, `short_form`

### 5. Heuristic Scoring

- Scorer: [subcategory_scorer.py](/home/pranav/PyCharm/EU-FarmBook/doc_classifier/docint/rubrics/subcategory_scorer.py)
- Criteria source of truth: [subcategories.py](/home/pranav/PyCharm/EU-FarmBook/doc_classifier/docint/rubrics/subcategories.py)

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

### 6. Text LLM Classification

- Module: [subcategory_classify.py](/home/pranav/PyCharm/EU-FarmBook/doc_classifier/docint/llm/subcategory_classify.py)
- Intended model family: Qwen or any OpenAI-compatible text endpoint

For agriculture-related documents, text LLM is allowed by default when `use_text_llm=true`. It receives extracted document text and is prompted with the same criteria vocabulary used by heuristics:

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

### 7. Selective Vision LLM Classification

- Module: [subcategory_classify.py](/home/pranav/PyCharm/EU-FarmBook/doc_classifier/docint/llm/subcategory_classify.py)
- Intended model family: InternVL or any OpenAI-compatible vision endpoint

Vision is no longer treated as a default whole-document pass. When `use_vision=true`, routing decides whether vision is needed based on signals such as:

- poor text quality
- OCR usage
- low subcategory confidence
- close top candidates
- visual/slide cues
- disagreement between heuristics and text LLM

When vision is used, the service sends deterministic representative sampled pages rather than overlapping sliding windows.

### 8. Fusion

- Module: [intelligent_fusion.py](/home/pranav/PyCharm/EU-FarmBook/doc_classifier/docint/fusion/intelligent_fusion.py)

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

- [document_subcategories_consolidation.md](/home/pranav/PyCharm/EU-FarmBook/doc_classifier/data_model/document_subcategories_consolidation.md)
- [data_model.subcategories_document_consolidated.json](/home/pranav/PyCharm/EU-FarmBook/doc_classifier/data_model/data_model.subcategories_document_consolidated.json)

## Explainability Outputs

The API response includes both direct and contrastive explanation layers.

Candidate-level fields include:

- `features_found`
- `feature_details`
- `rationale`
- `contrastive_rationale`

`contrastive_rationale` is generated in [app.py](/home/pranav/PyCharm/EU-FarmBook/doc_classifier/app.py) by comparing the winner with nearby alternatives using the same criteria metadata from [subcategories.py](/home/pranav/PyCharm/EU-FarmBook/doc_classifier/docint/rubrics/subcategories.py).

The `/subcategories` endpoint also exposes criteria metadata so that documentation and UIs can stay aligned with runtime definitions.

## Current Architectural Boundaries

### What Is Implemented

- PDF upload and classification
- agriculture-first early reject
- heuristic scoring with measurable signals
- criteria-backed contrastive explanations
- text LLM allowed by default for agriculture-related documents
- selective sampled-page vision classification
- fusion across sources

### What Is Documented But Not Yet Enforced In Runtime

- category auto-selection by KO mode
- category filtering for file-based versus URL-based KOs
- `Software Application` availability only for URL-based KOs
- broader non-document category handling such as `Dataset`, `Video`, `Audio`, and `Image`

Those rules are part of the taxonomy and ingestion design work, not the current `/classify` implementation.

## Deployment Paths

### Standalone

- install Python dependencies from [requirements.txt](/home/pranav/PyCharm/EU-FarmBook/doc_classifier/requirements.txt)
- install `poppler-utils`, `tesseract-ocr`, and required language packs
- configure `.env`
- run via [start_server.sh](/home/pranav/PyCharm/EU-FarmBook/doc_classifier/start_server.sh), [start_server.py](/home/pranav/PyCharm/EU-FarmBook/doc_classifier/start_server.py), or `uvicorn`

### Docker

- build and run with [Dockerfile](/home/pranav/PyCharm/EU-FarmBook/doc_classifier/Dockerfile)
- local Compose path available in [docker-compose.yml](/home/pranav/PyCharm/EU-FarmBook/doc_classifier/docker-compose.yml)

## File Map

- [app.py](/home/pranav/PyCharm/EU-FarmBook/doc_classifier/app.py): API entry point and response shaping
- [subcategories.py](/home/pranav/PyCharm/EU-FarmBook/doc_classifier/docint/rubrics/subcategories.py): criteria source of truth
- [subcategory_scorer.py](/home/pranav/PyCharm/EU-FarmBook/doc_classifier/docint/rubrics/subcategory_scorer.py): heuristic scoring engine
- [subcategory_classify.py](/home/pranav/PyCharm/EU-FarmBook/doc_classifier/docint/llm/subcategory_classify.py): text and vision LLM integration
- [data_model](/home/pranav/PyCharm/EU-FarmBook/doc_classifier/data_model): taxonomy consolidation and category policy documents
