# app.py
"""
KO Classifier API - FastAPI application for KO category and subtype classification.

This API provides evidence-based KO classification with optional LLM enhancement.
It supports text-based, vision-based, and category-specific hybrid routing with intelligent
fusion of results.

Run with: uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Depends, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import secrets

from docint.audio.transcribe import transcribe_audio_file
from docint.extract.quality import text_quality_ok
from docint.extract.ocr import ocr_pdf, ocr_image
from docint.category.infer import infer_category
from docint.category.audio_scorer import AUDIO_SUBTYPES, score_audio_subcategories
from docint.category.dataset_scorer import DATASET_SUBTYPES, score_dataset_subcategories
from docint.category.image_scorer import score_image_subcategories_from_text, IMAGE_SUBTYPES
from docint.category.video_scorer import VIDEO_SUBTYPES, score_video_subcategories
from docint.ingest.dispatcher import ingest_asset, SUPPORTED_DOCUMENT_EXTENSIONS
from docint.video.extract import media_duration_seconds, sample_video_frames, transcribe_video_audio
from docint.features.sections import count_sections
from docint.features.citations import detect_citations
from docint.features.keywords import count_keywords
from docint.domain.agriculture_pipeline import assess_agriculture_relevance_staged
from docint.rubrics.imrad import score_imrad
from docint.rubrics.citations import score_citations
from docint.rubrics.deliverable import score_deliverable
from docint.rubrics.pedagogy import score_pedagogy
from docint.rubrics.procedure import score_procedure
from docint.rubrics.subcategory_scorer import score_subcategories, SubcategoryScore
from docint.rubrics.subcategories import SUBCATEGORIES, get_subcategory_criteria
from docint.fusion.intelligent_fusion import (
    intelligent_fusion, 
    SourceResult, 
    FusionStrategy,
    convert_to_source_result
)

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

# Text LLM (Qwen) Configuration
LLM_BASE_URL = os.getenv("DOCINT_LLM_BASE_URL", "").rstrip("/")
if LLM_BASE_URL and not LLM_BASE_URL.endswith("/v1"):
    LLM_BASE_URL = f"{LLM_BASE_URL}/v1"
LLM_MODEL = os.getenv("DOCINT_LLM_MODEL", "qwen3-30b-a3b-awq")
LLM_API_KEY = os.getenv("DOCINT_LLM_API_KEY", "").strip()

# Vision LLM (InternVL) Configuration
VISION_LLM_BASE_URL = os.getenv("VISION_LLM_BASE_URL", "").rstrip("/")
if VISION_LLM_BASE_URL and not VISION_LLM_BASE_URL.endswith("/v1"):
    VISION_LLM_BASE_URL = f"{VISION_LLM_BASE_URL}/v1"
VISION_LLM_MODEL = os.getenv("VISION_LLM_MODEL", "internvl3-5-14b")
VISION_LLM_API_KEY = os.getenv("VISION_LLM_API_KEY", LLM_API_KEY).strip()

LLM_CONFIGURED = bool(LLM_BASE_URL and LLM_MODEL)

MEDIA_TRANSCRIBER_BASE_URL = os.getenv("MEDIA_TRANSCRIBER_BASE_URL", "").rstrip("/")
MEDIA_TRANSCRIBER_WHISPER_MODEL = os.getenv("MEDIA_TRANSCRIBER_WHISPER_MODEL", "").strip()
MEDIA_TRANSCRIBER_API_KEY = os.getenv("MEDIA_TRANSCRIBER_API_KEY", "").strip()
MEDIA_TRANSCRIBER_MODE = os.getenv("MEDIA_TRANSCRIBER_MODE", "auto").strip() or "auto"
MEDIA_TRANSCRIBER_BASIC_USER = os.getenv("MEDIA_TRANSCRIBER_BASIC_USER", "").strip()
MEDIA_TRANSCRIBER_BASIC_PASS = os.getenv("MEDIA_TRANSCRIBER_BASIC_PASS", "").strip()
MEDIA_TRANSCRIBER_ENABLED = os.getenv("MEDIA_TRANSCRIBER_ENABLED", "false").lower() == "true"
AUDIO_TRANSCRIPTION_CONFIGURED = bool(
    MEDIA_TRANSCRIBER_ENABLED and MEDIA_TRANSCRIBER_BASE_URL and MEDIA_TRANSCRIBER_WHISPER_MODEL
)

FFMPEG_AVAILABLE = bool(shutil.which("ffmpeg"))
FFPROBE_AVAILABLE = bool(shutil.which("ffprobe"))
MAX_AUDIO_DURATION_SEC = int(os.getenv("MAX_AUDIO_DURATION_SEC", "3000"))
MAX_VIDEO_DURATION_SEC = int(os.getenv("MAX_VIDEO_DURATION_SEC", "3000"))
MAX_AUDIO_UPLOAD_SIZE_MB = int(os.getenv("MAX_AUDIO_UPLOAD_SIZE_MB", "768"))
MAX_VIDEO_UPLOAD_SIZE_MB = int(os.getenv("MAX_VIDEO_UPLOAD_SIZE_MB", "1024"))
MAX_REQUEST_BODY_MB = int(os.getenv("MAX_REQUEST_BODY_MB", str(max(MAX_AUDIO_UPLOAD_SIZE_MB, MAX_VIDEO_UPLOAD_SIZE_MB))))

# All EU languages for Tesseract OCR
ALL_OCR_LANGS = "bul+ces+dan+deu+ell+eng+est+fin+fra+hrv+hun+ita+lav+lit+mlt+nld+pol+por+ron+slk+slv+spa+swe+gle"

# =============================================================================
# BASIC AUTH CONFIGURATION
# =============================================================================

# Initialize HTTPBasic security
security = HTTPBasic()

# Load authorized users from env
AUTH_USERS_STR = os.getenv("DOCINT_AUTH_USERS", "")
AUTH_PASSWORD = os.getenv("DOCINT_AUTH_PASSWORD", "")

# Parse comma-separated usernames
AUTHORIZED_USERS = {}
if AUTH_USERS_STR and AUTH_PASSWORD:
    for username in AUTH_USERS_STR.split(","):
        username = username.strip()
        if username:
            AUTHORIZED_USERS[username] = AUTH_PASSWORD

# Track if auth is enabled
AUTH_ENABLED = bool(AUTHORIZED_USERS)


def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """
    Verify Basic Auth credentials.
    
    Args:
        credentials: HTTP Basic auth credentials
        
    Returns:
        str: The authenticated username
        
    Raises:
        HTTPException: If credentials are invalid
    """
    if not AUTH_ENABLED:
        # No auth configured, allow all
        return "anonymous"
    
    # Check if username exists
    stored_password = AUTHORIZED_USERS.get(credentials.username)
    
    if stored_password is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": 'Basic realm="KO Classifier API"'},
        )
    
    # Verify password using constant-time comparison
    if not secrets.compare_digest(credentials.password, stored_password):
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": 'Basic realm="KO Classifier API"'},
        )
    
    return credentials.username


def require_auth():
    """
    Dependency that enforces authentication.
    Use this when auth should be required even if not globally enabled.
    """
    if not AUTH_ENABLED:
        return "anonymous"
    return Depends(verify_credentials)


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class FeatureEvidence(BaseModel):
    """Evidence for a single feature detection."""
    feature_name: str
    detected: bool
    score: float
    raw_value: Any
    excerpts: List[str]


class SubcategoryCandidate(BaseModel):
    """A single subcategory candidate with scoring."""
    subcategory_id: str
    subcategory_name: str
    parent_type: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    probability: float = Field(..., ge=0.0, le=1.0)
    evidence_score: float
    max_possible_evidence: float
    features_found: List[str]
    feature_details: Dict[str, FeatureEvidence]
    rationale: str
    contrastive_rationale: Optional[str] = None
    selection_basis: Optional[str] = None
    supporting_sources: Optional[List[str]] = None
    source_confidences: Optional[Dict[str, float]] = None
    source_rationales: Optional[Dict[str, str]] = None
    rank: int = Field(..., description="Rank by confidence (1 = best match)")


class FusionInfo(BaseModel):
    """Information about the fusion process."""
    fused: bool
    strategy: str
    weights: Dict[str, float]
    agreement_score: float
    rationale: str


class AgricultureRelevance(BaseModel):
    """Agriculture domain relevance assessment."""
    is_agriculture_related: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    score: float = Field(..., ge=0.0, le=1.0)
    method: str
    lexicon_version: str
    matched_terms: List[str]
    matched_buckets: List[str]
    matched_concepts: List[str]
    bucket_scores: Dict[str, float]
    rationale: str
    stages_used: List[str]
    stage_results: List[Dict[str, Any]]


class CategoryInference(BaseModel):
    """Inferred high-level category for the uploaded asset."""
    category: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str


class ClassificationResponse(BaseModel):
    """Complete classification response."""
    # Primary results
    best_match: Optional[SubcategoryCandidate] = None
    all_candidates: List[SubcategoryCandidate] = []
    
    # Fusion info (if multiple sources used)
    fusion: Optional[FusionInfo] = None
    
    # Individual source results
    heuristics: Optional[SubcategoryCandidate] = None
    vision_llm: Optional[Dict[str, Any]] = None
    text_llm: Optional[Dict[str, Any]] = None
    category_inference: CategoryInference
    agriculture_relevance: AgricultureRelevance
    classification_skipped: bool = False
    skip_reason: Optional[str] = None
    
    # Metadata
    total_candidates: int
    confidence_threshold_met: bool
    document_info: Dict[str, Any]
    processing_info: Dict[str, Any]


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="KO Classifier API",
    description="""
    Agriculture-gated KO classification with explainable category-specific subtype scoring.
    
    ## Features
    
    * **Agriculture Relevance Gate**: Rejects non-agriculture assets before subtype classification
    * **Multi-Category Routing**: Supports current `Document`, `Dataset`, `Image`, `Audio`, and `Video` branches
    * **Text LLM (Qwen)**: Allowed by default for agriculture-related text-rich assets
    * **Selective Vision LLM (InternVL)**: Triggered only when routing decides visual evidence is needed
    * **Intelligent Fusion**: Combines heuristics and model outputs using configurable strategies
    
    ## Runtime Flow
    
    1. Ingest the uploaded asset with a category-appropriate extractor
    2. Assess agriculture relevance
    3. Reject early if the file is non-agriculture
    4. Run category-specific heuristic subtype scoring
    5. Use text LLM for agri text-rich assets when enabled
    6. Trigger vision only for low-confidence, visually-driven, or weak-text cases
    7. Fuse available sources using the selected strategy
    """,
    version="2.0.0",
    docs_url="/docs",  # Enable docs - they'll be protected by middleware
    redoc_url="/redoc",
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def convert_to_candidate(
    score: SubcategoryScore, 
    rank: int,
    total_probability: float
) -> SubcategoryCandidate:
    """Convert SubcategoryScore to API response model."""
    if total_probability > 0:
        probability = score.confidence / total_probability
    else:
        probability = 1.0 / 11
    
    return SubcategoryCandidate(
        subcategory_id=score.subcategory_id,
        subcategory_name=score.subcategory_name,
        parent_type=score.parent_type,
        confidence=round(score.confidence, 4),
        probability=round(probability, 4),
        evidence_score=round(score.evidence_score, 4),
        max_possible_evidence=round(score.max_possible_evidence, 4),
        features_found=score.features_found,
        feature_details={
            k: FeatureEvidence(**v.to_dict())
            for k, v in score.feature_details.items()
        },
        rationale=score.rationale,
        rank=rank
    )


def build_probability_distribution(
    scores: List[SubcategoryScore]
) -> List[SubcategoryCandidate]:
    """Build probability distribution from all subcategory scores."""
    sorted_scores = sorted(scores, key=lambda x: x.confidence, reverse=True)
    total_confidence = sum(s.confidence for s in sorted_scores)
    
    candidates = []
    for rank, score in enumerate(sorted_scores, 1):
        candidate = convert_to_candidate(score, rank, total_confidence)
        candidates.append(candidate)
    
    # Renormalize probabilities
    total_prob = sum(c.probability for c in candidates)
    if total_prob > 0:
        for c in candidates:
            c.probability = round(c.probability / total_prob, 4)
    
    return candidates


def _candidate_missing_features(candidate: SubcategoryCandidate) -> List[str]:
    """List expected features for a candidate that were not detected."""
    missing: List[str] = []
    for feat_name, evidence in candidate.feature_details.items():
        if not evidence.detected:
            missing.append(feat_name)
    return missing


def _criteria_by_name() -> Dict[str, Dict[str, Any]]:
    criteria = {}
    for item in get_subcategory_criteria().values():
        criteria[item["name"]] = item
    return criteria


def _subcategory_key_by_name() -> Dict[str, str]:
    return {subcat.name: key for key, subcat in SUBCATEGORIES.items()}


def add_fusion_explanations(
    candidates: List[SubcategoryCandidate],
    fusion_result: Any,
    source_results: Dict[str, Any],
) -> None:
    """Annotate fused candidates with source-aware explanations."""
    if not candidates:
        return

    key_by_name = _subcategory_key_by_name()
    name_by_key = {key: subcat.name for key, subcat in SUBCATEGORIES.items()}

    for cand in candidates:
        cand.selection_basis = "fusion"
        subcat_key = key_by_name.get(cand.subcategory_name, cand.subcategory_name)

        source_confidences: Dict[str, float] = {}
        source_rationales: Dict[str, str] = {}
        supporting_sources: List[str] = []

        for source_name, source in source_results.items():
            if not source:
                continue
            source_confidences[source_name] = round(source.confidence, 4)
            if getattr(source, "rationale", None):
                source_rationales[source_name] = source.rationale

            source_top_key = getattr(source, "subcategory_key", "")
            source_prob = round(source.probs.get(subcat_key, 0.0), 4)
            if source_top_key == subcat_key or source_prob >= 0.2:
                supporting_sources.append(source_name)

        cand.supporting_sources = supporting_sources or None
        cand.source_confidences = source_confidences or None
        cand.source_rationales = source_rationales or None

        if cand.rank != 1:
            continue

        source_summaries: List[str] = []
        for source_name, source in source_results.items():
            if not source:
                continue
            source_top_key = getattr(source, "subcategory_key", "")
            source_top_name = name_by_key.get(source_top_key, source_top_key)
            source_summaries.append(
                f"{source_name} favored {source_top_name} ({source.confidence:.2f})"
            )

        fused_bits = [
            f"Fused result selected {cand.subcategory_name} with probability {cand.probability:.2f}.",
        ]
        if supporting_sources:
            fused_bits.append(
                "Supporting sources: " + ", ".join(supporting_sources) + "."
            )
        if source_summaries:
            fused_bits.append("Source summary: " + "; ".join(source_summaries) + ".")
        if fusion_result and getattr(fusion_result, "rationale", None):
            fused_bits.append(fusion_result.rationale)

        cand.rationale = " ".join(fused_bits)


def add_contrastive_rationales(candidates: List[SubcategoryCandidate], top_k: int = 3) -> None:
    """
    Add a short contrastive explanation for the top candidates so the response
    states not only what won, but what close alternatives were missing.
    """
    if len(candidates) < 2:
        return

    criteria_by_name = _criteria_by_name()
    focus = candidates[:max(2, min(top_k, len(candidates)))]
    winner = focus[0]
    winner_feats = set(winner.features_found)
    winner_criteria = criteria_by_name.get(winner.subcategory_name, {})

    contrasts: List[str] = []
    for alt in focus[1:]:
        alt_feats = set(alt.features_found)
        winner_adv = sorted(winner_feats - alt_feats)
        alt_missing = _candidate_missing_features(alt)
        alt_criteria = criteria_by_name.get(alt.subcategory_name, {})

        pieces: List[str] = []
        if winner_adv:
            pieces.append(f"stronger on {', '.join(winner_adv[:2])}")
        elif winner_criteria.get("positive_signal_hints"):
            pieces.append(
                "better matches "
                + ", ".join(winner_criteria["positive_signal_hints"][:2])
            )
        if alt_missing:
            pieces.append(f"{alt.subcategory_name} missing {', '.join(alt_missing[:2])}")
        elif alt_criteria.get("negative_signal_hints"):
            pieces.append(
                f"{alt.subcategory_name} conflicted with "
                + ", ".join(alt_criteria["negative_signal_hints"][:1])
            )
        if not pieces:
            pieces.append(f"higher overall evidence than {alt.subcategory_name}")

        contrasts.append(f"vs {alt.subcategory_name}: " + "; ".join(pieces))

    if contrasts:
        winner.contrastive_rationale = " | ".join(contrasts)

    # Add a compact note for near-miss alternatives too.
    for idx, cand in enumerate(focus[1:], start=1):
        missing = _candidate_missing_features(cand)
        cand_criteria = criteria_by_name.get(cand.subcategory_name, {})
        if missing:
            cand.contrastive_rationale = (
                f"Near miss behind {winner.subcategory_name}; missing "
                f"{', '.join(missing[:3])}"
            )
        elif cand_criteria.get("negative_signal_hints"):
            cand.contrastive_rationale = (
                f"Near miss behind {winner.subcategory_name}; conflicted with "
                f"{cand_criteria['negative_signal_hints'][0]}"
            )
        else:
            cand.contrastive_rationale = f"Near miss behind {winner.subcategory_name}"


def _top_probability_gap(candidates: List[SubcategoryCandidate]) -> float:
    if len(candidates) < 2:
        return 1.0
    return max(0.0, round(candidates[0].probability - candidates[1].probability, 4))


def _model_keys_disagree(*sources: Optional[SourceResult]) -> bool:
    tops = [s.subcategory_key for s in sources if s and getattr(s, "subcategory_key", "")]
    return len(set(tops)) > 1 if len(tops) >= 2 else False


def _candidate_visual_signals(candidate: Optional[SubcategoryCandidate]) -> bool:
    if not candidate:
        return False
    return bool({"visual_heavy", "slide_indicators"} & set(candidate.features_found))


def _should_run_text_llm(
    *,
    use_text_llm: bool,
    is_agriculture_related: bool,
) -> bool:
    return use_text_llm and is_agriculture_related and LLM_CONFIGURED


def _should_run_vision(
    *,
    use_vision: bool,
    ocr_used: bool,
    text_quality_ok_flag: bool,
    heuristics_source: Optional[SourceResult],
    best_candidate: Optional[SubcategoryCandidate],
    current_candidates: List[SubcategoryCandidate],
    text_source: Optional[SourceResult],
    confidence_threshold: float,
    gap_threshold: float,
) -> tuple[bool, List[str]]:
    reasons: List[str] = []
    if not use_vision or not VISION_LLM_BASE_URL:
        return False, reasons

    top_confidence = best_candidate.confidence if best_candidate else 0.0
    top_gap = _top_probability_gap(current_candidates)

    if not text_quality_ok_flag:
        reasons.append("text_quality_poor")
    if ocr_used:
        reasons.append("ocr_used")
    if top_confidence < confidence_threshold:
        reasons.append("low_subcategory_confidence")
    if top_gap < gap_threshold:
        reasons.append("close_top_candidates")
    if _candidate_visual_signals(best_candidate):
        reasons.append("visual_or_slide_signals")
    if _model_keys_disagree(heuristics_source, text_source):
        reasons.append("heuristics_text_disagreement")

    deduped_reasons = list(dict.fromkeys(reasons))
    return bool(deduped_reasons), deduped_reasons


def classify_document(
    file_path: str,
    filename: str,
    require_agriculture: bool = True,
    auto_route_models: bool = True,
    use_vision: bool = False,
    use_text_llm: bool = False,
    heuristics_alpha: float = 0.4,
    classification_confidence_threshold: float = 0.35,
    vision_trigger_threshold: float = 0.6,
    candidate_gap_threshold: float = 0.12,
    fusion_strategy: str = "adaptive",
    vision_max_pages: int = 20,
    ocr_lang: str = ALL_OCR_LANGS,
    ocr_max_pages: int = 10,
) -> ClassificationResponse:
    """
    Main classification function with optional LLM fusion.
    
    Args:
        file_path: Path to uploaded file
        filename: Original filename
        require_agriculture: Whether to stop early for non-agriculture assets
        auto_route_models: Whether to decide text/vision usage automatically
        use_vision: Whether to use Vision LLM
        use_text_llm: Whether to use Text LLM
        heuristics_alpha: Weight for heuristics (0.0-1.0), LLM gets (1-alpha)
        classification_confidence_threshold: Minimum confidence considered acceptable
        vision_trigger_threshold: Confidence below which vision may be triggered
        candidate_gap_threshold: Gap below which close candidates may trigger vision
        fusion_strategy: Fusion strategy (weighted, adaptive, agreement, cascade)
        vision_max_pages: Max pages for vision analysis
        ocr_lang: Tesseract OCR languages
        ocr_max_pages: Max pages for OCR fallback
    
    Returns:
        ClassificationResponse with results and fusion info
    """
    import time
    
    start_time = time.time()
    stage_timings_ms: Dict[str, float] = {}
    
    # 1) Extract text from PDF
    stage_start = time.time()
    asset = ingest_asset(file_path, filename)
    stage_timings_ms["text_extraction_ms"] = round((time.time() - stage_start) * 1000, 2)

    stage_start = time.time()
    category_result = infer_category(asset)
    category_response = CategoryInference(
        category=category_result.category,
        confidence=round(category_result.confidence, 4),
        rationale=category_result.rationale,
    )
    stage_timings_ms["category_inference_ms"] = round((time.time() - stage_start) * 1000, 2)
    
    # 2) OCR fallback or audio transcription if needed
    stage_start = time.time()
    quality = text_quality_ok(asset.text)
    if asset.ocr_supported and not quality.ok:
        if category_result.category == "Image":
            ocr_doc = ocr_image(file_path, lang=ocr_lang)
        else:
            ocr_doc = ocr_pdf(file_path, max_pages=ocr_max_pages, lang=ocr_lang)
        from dataclasses import replace
        asset = replace(
            asset,
            text=ocr_doc.text,
            lines=ocr_doc.lines,
            source="ocr",
            meta={**asset.meta, **ocr_doc.meta},
        )
    if category_result.category == "Audio":
        transcript = transcribe_audio_file(file_path)
        if transcript.text:
            from dataclasses import replace
            asset = replace(
                asset,
                text=transcript.text,
                lines=[line.strip() for line in transcript.text.splitlines() if line.strip()],
                source="audio_transcript",
                meta={
                    **asset.meta,
                    "transcription_available": transcript.available,
                    "transcription_used": transcript.used,
                    "transcription_method": transcript.method,
                    "transcription_model": transcript.model,
                    "transcription_rationale": transcript.rationale,
                },
            )
        else:
            asset.meta.update(
                {
                    "transcription_available": transcript.available,
                    "transcription_used": transcript.used,
                    "transcription_method": transcript.method,
                    "transcription_model": transcript.model,
                    "transcription_rationale": transcript.rationale,
                }
            )
        stage_timings_ms["audio_transcription_ms"] = round((time.time() - stage_start) * 1000, 2)
    if category_result.category == "Video":
        transcript = transcribe_video_audio(file_path)
        if transcript.text:
            from dataclasses import replace
            asset = replace(
                asset,
                text=transcript.text,
                lines=[line.strip() for line in transcript.text.splitlines() if line.strip()],
                source="video_transcript",
                meta={
                    **asset.meta,
                    "transcription_available": transcript.available,
                    "transcription_used": transcript.used,
                    "transcription_method": transcript.method,
                    "transcription_model": transcript.model,
                    "transcription_rationale": transcript.rationale,
                },
            )
        else:
            asset.meta.update(
                {
                    "transcription_available": transcript.available,
                    "transcription_used": transcript.used,
                    "transcription_method": transcript.method,
                    "transcription_model": transcript.model,
                    "transcription_rationale": transcript.rationale,
                }
            )
        stage_timings_ms["video_transcription_ms"] = round((time.time() - stage_start) * 1000, 2)
    quality = text_quality_ok(asset.text)
    stage_timings_ms["ocr_fallback_ms"] = round((time.time() - stage_start) * 1000, 2)
    
    # 3) Agriculture gate before the heavier classification pipeline
    stage_start = time.time()
    agri_relevance = assess_agriculture_relevance_staged(
        asset.text,
        lines=asset.lines,
        allow_llm_fallback=use_text_llm and LLM_CONFIGURED,
        llm_config={
            "base_url": LLM_BASE_URL,
            "api_key": LLM_API_KEY,
            "model": LLM_MODEL,
        } if use_text_llm and LLM_CONFIGURED else None,
    )
    stage_timings_ms["agriculture_pipeline_ms"] = round((time.time() - stage_start) * 1000, 2)
    agriculture_response = AgricultureRelevance(
        is_agriculture_related=agri_relevance.is_agriculture_related,
        confidence=round(agri_relevance.confidence, 4),
        score=round(agri_relevance.score, 4),
        method=agri_relevance.method,
        lexicon_version=agri_relevance.lexicon_version,
        matched_terms=agri_relevance.matched_terms,
        matched_buckets=agri_relevance.matched_buckets,
        matched_concepts=agri_relevance.matched_concepts,
        bucket_scores={k: round(v, 4) for k, v in agri_relevance.bucket_scores.items()},
        rationale=agri_relevance.rationale,
        stages_used=agri_relevance.stages_used,
        stage_results=[
            {
                "stage": item.stage,
                "available": item.available,
                "used": item.used,
                "is_agriculture_related": item.is_agriculture_related,
                "confidence": item.confidence,
                "rationale": item.rationale,
                "details": item.details,
            }
            for item in agri_relevance.stage_results
        ],
    )

    if category_result.category == "Image":
        stage_start = time.time()
        _, image_scores = score_image_subcategories_from_text(asset.text, asset.lines)
        image_candidates = build_probability_distribution(image_scores)
        add_contrastive_rationales(image_candidates)
        image_best = image_candidates[0] if image_candidates else None
        image_heuristics_best = image_best.model_copy(deep=True) if image_best else None
        stage_timings_ms["image_classification_ms"] = round((time.time() - stage_start) * 1000, 2)
        image_vision_result = None
        image_fusion = None

        if use_vision and VISION_LLM_BASE_URL:
            stage_start = time.time()
            try:
                from docint.llm.subcategory_classify import llm_classify_image_with_vision

                vlm_res = llm_classify_image_with_vision(
                    file_path,
                    base_url=VISION_LLM_BASE_URL,
                    api_key=VISION_LLM_API_KEY,
                    model=VISION_LLM_MODEL,
                    temperature=0.2,
                )
                image_vision_result = {
                    "subcategory_key": vlm_res.get("subcategory"),
                    "subcategory_name": IMAGE_SUBTYPES.get(vlm_res.get("subcategory"), next(iter(IMAGE_SUBTYPES.values()))).name if vlm_res.get("subcategory") else None,
                    "confidence": round(float(vlm_res.get("confidence", 0.0)), 4),
                    "rationale": vlm_res.get("rationale", ""),
                    "model": VISION_LLM_MODEL,
                    "agriculture_related": bool(vlm_res.get("is_agriculture_related", False)),
                    "agriculture_confidence": round(float(vlm_res.get("agriculture_confidence", 0.0)), 4),
                }
                agriculture_response.is_agriculture_related = bool(vlm_res.get("is_agriculture_related", agriculture_response.is_agriculture_related))
                agriculture_response.confidence = round(float(vlm_res.get("agriculture_confidence", agriculture_response.confidence)), 4)
                agriculture_response.score = agriculture_response.confidence
                agriculture_response.method = "image_vision_override"
                agriculture_response.rationale = str(vlm_res.get("rationale", agriculture_response.rationale))

                if image_best:
                    name_to_image_key = {subtype.name: key for key, subtype in IMAGE_SUBTYPES.items()}
                    heuristics_probs = {
                        name_to_image_key.get(candidate.subcategory_name, candidate.subcategory_name): candidate.probability
                        for candidate in image_candidates
                    }
                    heuristics_source = convert_to_source_result(
                        subcategory_key=name_to_image_key.get(image_best.subcategory_name, list(heuristics_probs.keys())[0]),
                        confidence=image_best.confidence,
                        probs=heuristics_probs,
                        source_name="heuristics",
                        evidence_score=image_best.evidence_score,
                        rationale=image_best.rationale,
                    )
                    vision_source = convert_to_source_result(
                        subcategory_key=vlm_res.get("subcategory") or list(IMAGE_SUBTYPES.keys())[0],
                        confidence=float(vlm_res.get("confidence", 0.0)),
                        probs=vlm_res.get("probs", {}),
                        source_name="vision_llm",
                        rationale=str(vlm_res.get("rationale", "")),
                    )
                    fusion_result = intelligent_fusion(
                        heuristics_result=heuristics_source,
                        vision_result=vision_source,
                        text_result=None,
                        strategy=FusionStrategy.CONFIDENCE_ADAPTIVE if fusion_strategy == "adaptive" else FusionStrategy.WEIGHTED,
                        heuristics_alpha=heuristics_alpha,
                        llm_alpha=1.0 - heuristics_alpha,
                    )
                    image_fusion = FusionInfo(
                        fused=True,
                        strategy=fusion_result.fusion_strategy,
                        weights=fusion_result.weights,
                        agreement_score=fusion_result.agreement_score,
                        rationale=fusion_result.rationale,
                    )
                    for candidate in image_candidates:
                        image_key = name_to_image_key.get(candidate.subcategory_name, candidate.subcategory_name)
                        fused_prob = round(fusion_result.probs.get(image_key, 0), 4)
                        candidate.probability = fused_prob
                        candidate.confidence = fused_prob
                    image_candidates.sort(key=lambda item: item.probability, reverse=True)
                    for idx, candidate in enumerate(image_candidates, start=1):
                        candidate.rank = idx
                    add_contrastive_rationales(image_candidates)
                    image_best = image_candidates[0] if image_candidates else None
            except Exception as e:
                image_vision_result = {"error": str(e), "model": VISION_LLM_MODEL}
            stage_timings_ms["image_vision_llm_ms"] = round((time.time() - stage_start) * 1000, 2)

        if require_agriculture and not agriculture_response.is_agriculture_related:
            processing_time_ms = (time.time() - start_time) * 1000
            return ClassificationResponse(
                best_match=None,
                all_candidates=[],
                fusion=None,
                heuristics=None,
                vision_llm=image_vision_result,
                text_llm=None,
                category_inference=category_response,
                agriculture_relevance=agriculture_response,
                classification_skipped=True,
                skip_reason="Image classified as non-agriculture; image subcategory classification skipped",
                total_candidates=0,
                confidence_threshold_met=False,
                document_info={
                    "filename": filename,
                    "pages": asset.units,
                    "unit_label": asset.unit_label,
                    "asset_type": asset.asset_type,
                    "inferred_category": category_result.category,
                    "source": asset.source,
                    "text_length": len(asset.text),
                    "text_quality": {
                        "chars": quality.metrics.get("chars"),
                        "letters": quality.metrics.get("letters"),
                        "letter_ratio": quality.metrics.get("letter_ratio"),
                        "ok": quality.ok,
                    } if hasattr(quality, 'metrics') else None,
                },
                processing_info={
                    "processing_time_ms": round(processing_time_ms, 2),
                    "ocr_used": asset.source == "ocr",
                    "sources_used": ["vision_llm"] if image_vision_result and "error" not in image_vision_result else [],
                    "fusion_enabled": False,
                    "require_agriculture": require_agriculture,
                    "auto_route_models": auto_route_models,
                    "stage_timings_ms": stage_timings_ms,
                },
            )

        processing_time_ms = (time.time() - start_time) * 1000
        return ClassificationResponse(
            best_match=image_best,
            all_candidates=image_candidates,
            fusion=image_fusion,
            heuristics=image_heuristics_best,
            vision_llm=image_vision_result,
            text_llm=None,
            category_inference=category_response,
            agriculture_relevance=agriculture_response,
            classification_skipped=False,
            skip_reason=None,
            total_candidates=len(image_candidates),
            confidence_threshold_met=bool(image_best and image_best.confidence >= classification_confidence_threshold),
            document_info={
                "filename": filename,
                "pages": asset.units,
                "unit_label": asset.unit_label,
                "asset_type": asset.asset_type,
                "inferred_category": category_result.category,
                "source": asset.source,
                "text_length": len(asset.text),
                "text_quality": {
                    "chars": quality.metrics.get("chars"),
                    "letters": quality.metrics.get("letters"),
                    "letter_ratio": quality.metrics.get("letter_ratio"),
                    "ok": quality.ok,
                } if hasattr(quality, 'metrics') else None,
            },
            processing_info={
                "processing_time_ms": round(processing_time_ms, 2),
                "ocr_used": asset.source == "ocr",
                "sources_used": ["heuristics"] + (["vision_llm"] if image_vision_result and "error" not in image_vision_result else []),
                "fusion_enabled": image_fusion is not None,
                "require_agriculture": require_agriculture,
                "auto_route_models": auto_route_models,
                "routing": {
                    "image_mode": True,
                    "text_llm": {"requested": use_text_llm, "used": False, "reason": "image_text_llm_not_enabled"},
                    "vision_llm": {"requested": use_vision, "used": image_vision_result is not None and "error" not in image_vision_result, "reasons": ["image_primary_classifier"]},
                },
                "classification_confidence_threshold": classification_confidence_threshold,
                "stage_timings_ms": stage_timings_ms,
            },
        )

    if category_result.category == "Video":
        stage_start = time.time()
        _, video_scores = score_video_subcategories(asset.text, asset.lines, filename=filename)
        video_candidates = build_probability_distribution(video_scores)
        add_contrastive_rationales(video_candidates)
        video_best = video_candidates[0] if video_candidates else None
        video_heuristics_best = video_best.model_copy(deep=True) if video_best else None
        stage_timings_ms["video_classification_ms"] = round((time.time() - stage_start) * 1000, 2)
        video_text_result = None
        video_text_source = None
        video_vision_result = None
        video_vision_source = None
        video_fusion = None

        if use_text_llm and LLM_CONFIGURED and asset.text.strip():
            stage_start = time.time()
            try:
                from docint.llm.subcategory_classify import llm_classify_video_subcategories_text

                llm_res = llm_classify_video_subcategories_text(
                    asset.text,
                    base_url=LLM_BASE_URL,
                    api_key=LLM_API_KEY,
                    model=LLM_MODEL,
                    max_chars=12000,
                    temperature=0.2,
                )
                video_text_source = convert_to_source_result(
                    subcategory_key=llm_res.subcategory_key,
                    confidence=llm_res.confidence,
                    probs=llm_res.probs,
                    source_name="text_llm",
                    rationale=llm_res.rationale,
                )
                video_text_result = {
                    "subcategory_key": llm_res.subcategory_key,
                    "subcategory_name": llm_res.subcategory_name,
                    "confidence": round(llm_res.confidence, 4),
                    "rationale": llm_res.rationale,
                    "model": LLM_MODEL,
                }
            except Exception as e:
                video_text_result = {"error": str(e), "model": LLM_MODEL}
            stage_timings_ms["video_text_llm_ms"] = round((time.time() - stage_start) * 1000, 2)

        if use_vision and VISION_LLM_BASE_URL:
            stage_start = time.time()
            try:
                from docint.llm.subcategory_classify import llm_classify_video_with_vision

                frame_sampling = sample_video_frames(file_path, max_frames=min(vision_max_pages, 8))
                stage_timings_ms["video_frame_sampling_ms"] = round((time.time() - stage_start) * 1000, 2)
                if frame_sampling.frame_paths:
                    vlm_res = llm_classify_video_with_vision(
                        frame_sampling.frame_paths,
                        asset.text,
                        base_url=VISION_LLM_BASE_URL,
                        api_key=VISION_LLM_API_KEY,
                        model=VISION_LLM_MODEL,
                        temperature=0.2,
                    )
                    video_vision_result = {
                        "subcategory_key": vlm_res.get("subcategory"),
                        "subcategory_name": VIDEO_SUBTYPES.get(vlm_res.get("subcategory"), next(iter(VIDEO_SUBTYPES.values()))).name if vlm_res.get("subcategory") else None,
                        "confidence": round(float(vlm_res.get("confidence", 0.0)), 4),
                        "rationale": vlm_res.get("rationale", ""),
                        "model": VISION_LLM_MODEL,
                        "agriculture_related": bool(vlm_res.get("is_agriculture_related", False)),
                        "agriculture_confidence": round(float(vlm_res.get("agriculture_confidence", 0.0)), 4),
                        "sampled_frame_count": int(vlm_res.get("sampled_frame_count", len(frame_sampling.frame_paths))),
                    }
                    agriculture_response.is_agriculture_related = bool(vlm_res.get("is_agriculture_related", agriculture_response.is_agriculture_related))
                    agriculture_response.confidence = round(float(vlm_res.get("agriculture_confidence", agriculture_response.confidence)), 4)
                    agriculture_response.score = agriculture_response.confidence
                    agriculture_response.method = "video_vision_override"
                    agriculture_response.rationale = str(vlm_res.get("rationale", agriculture_response.rationale))
                    video_vision_source = convert_to_source_result(
                        subcategory_key=vlm_res.get("subcategory") or list(VIDEO_SUBTYPES.keys())[0],
                        confidence=float(vlm_res.get("confidence", 0.0)),
                        probs=vlm_res.get("probs", {}),
                        source_name="vision_llm",
                        rationale=str(vlm_res.get("rationale", "")),
                    )
                else:
                    video_vision_result = {
                        "error": frame_sampling.rationale,
                        "model": VISION_LLM_MODEL,
                    }
            except Exception as e:
                video_vision_result = {"error": str(e), "model": VISION_LLM_MODEL}
            stage_timings_ms["video_vision_llm_ms"] = round((time.time() - stage_start) * 1000, 2)

        if require_agriculture and not agriculture_response.is_agriculture_related:
            processing_time_ms = (time.time() - start_time) * 1000
            return ClassificationResponse(
                best_match=None,
                all_candidates=[],
                fusion=None,
                heuristics=None,
                vision_llm=video_vision_result,
                text_llm=video_text_result,
                category_inference=category_response,
                agriculture_relevance=agriculture_response,
                classification_skipped=True,
                skip_reason="Video classified as non-agriculture; video subcategory classification skipped",
                total_candidates=0,
                confidence_threshold_met=False,
                document_info={
                    "filename": filename,
                    "pages": asset.units,
                    "unit_label": asset.unit_label,
                    "asset_type": asset.asset_type,
                    "inferred_category": category_result.category,
                    "source": asset.source,
                    "text_length": len(asset.text),
                    "text_quality": {
                        "chars": quality.metrics.get("chars"),
                        "letters": quality.metrics.get("letters"),
                        "letter_ratio": quality.metrics.get("letter_ratio"),
                        "ok": quality.ok,
                    } if hasattr(quality, 'metrics') else None,
                },
                processing_info={
                    "processing_time_ms": round(processing_time_ms, 2),
                    "ocr_used": False,
                    "sources_used": [src for src, val in [("text_llm", video_text_result), ("vision_llm", video_vision_result)] if val and "error" not in val],
                    "fusion_enabled": False,
                    "require_agriculture": require_agriculture,
                    "auto_route_models": auto_route_models,
                    "stage_timings_ms": stage_timings_ms,
                },
            )

        if not asset.text.strip() and not (video_vision_result and "error" not in video_vision_result):
            processing_time_ms = (time.time() - start_time) * 1000
            return ClassificationResponse(
                best_match=None,
                all_candidates=[],
                fusion=None,
                heuristics=None,
                vision_llm=video_vision_result,
                text_llm=video_text_result,
                category_inference=category_response,
                agriculture_relevance=agriculture_response,
                classification_skipped=True,
                skip_reason="Video transcript and sampled-frame evidence unavailable; video subtype classification skipped",
                total_candidates=0,
                confidence_threshold_met=False,
                document_info={
                    "filename": filename,
                    "pages": asset.units,
                    "unit_label": asset.unit_label,
                    "asset_type": asset.asset_type,
                    "inferred_category": category_result.category,
                    "source": asset.source,
                    "text_length": len(asset.text),
                    "text_quality": {
                        "chars": quality.metrics.get("chars"),
                        "letters": quality.metrics.get("letters"),
                        "letter_ratio": quality.metrics.get("letter_ratio"),
                        "ok": quality.ok,
                    } if hasattr(quality, 'metrics') else None,
                },
                processing_info={
                    "processing_time_ms": round(processing_time_ms, 2),
                    "ocr_used": False,
                    "sources_used": [],
                    "fusion_enabled": False,
                    "require_agriculture": require_agriculture,
                    "auto_route_models": auto_route_models,
                    "stage_timings_ms": stage_timings_ms,
                },
            )

        if video_best:
            name_to_video_key = {subtype.name: key for key, subtype in VIDEO_SUBTYPES.items()}
            video_heuristics_probs = {
                name_to_video_key.get(candidate.subcategory_name, candidate.subcategory_name): candidate.probability
                for candidate in video_candidates
            }
            video_heuristics_source = convert_to_source_result(
                subcategory_key=name_to_video_key.get(video_best.subcategory_name, list(video_heuristics_probs.keys())[0]),
                confidence=video_best.confidence,
                probs=video_heuristics_probs,
                source_name="heuristics",
                evidence_score=video_best.evidence_score,
                rationale=video_best.rationale,
            )
            if video_text_source or video_vision_source:
                fusion_result = intelligent_fusion(
                    heuristics_result=video_heuristics_source,
                    vision_result=video_vision_source,
                    text_result=video_text_source,
                    strategy=FusionStrategy.CONFIDENCE_ADAPTIVE if fusion_strategy == "adaptive" else FusionStrategy.WEIGHTED,
                    heuristics_alpha=heuristics_alpha,
                    llm_alpha=1.0 - heuristics_alpha,
                )
                video_fusion = FusionInfo(
                    fused=True,
                    strategy=fusion_result.fusion_strategy,
                    weights=fusion_result.weights,
                    agreement_score=fusion_result.agreement_score,
                    rationale=fusion_result.rationale,
                )
                for candidate in video_candidates:
                    video_key = name_to_video_key.get(candidate.subcategory_name, candidate.subcategory_name)
                    fused_prob = round(fusion_result.probs.get(video_key, 0), 4)
                    candidate.probability = fused_prob
                    candidate.confidence = fused_prob
                video_candidates.sort(key=lambda item: item.probability, reverse=True)
                for idx, candidate in enumerate(video_candidates, start=1):
                    candidate.rank = idx
                video_best = video_candidates[0] if video_candidates else None
                add_contrastive_rationales(video_candidates)

        processing_time_ms = (time.time() - start_time) * 1000
        return ClassificationResponse(
            best_match=video_best,
            all_candidates=video_candidates,
            fusion=video_fusion,
            heuristics=video_heuristics_best,
            vision_llm=video_vision_result,
            text_llm=video_text_result,
            category_inference=category_response,
            agriculture_relevance=agriculture_response,
            classification_skipped=False,
            skip_reason=None,
            total_candidates=len(video_candidates),
            confidence_threshold_met=bool(video_best and video_best.confidence >= classification_confidence_threshold),
            document_info={
                "filename": filename,
                "pages": asset.units,
                "unit_label": asset.unit_label,
                "asset_type": asset.asset_type,
                "inferred_category": category_result.category,
                "source": asset.source,
                "text_length": len(asset.text),
                "text_quality": {
                    "chars": quality.metrics.get("chars"),
                    "letters": quality.metrics.get("letters"),
                    "letter_ratio": quality.metrics.get("letter_ratio"),
                    "ok": quality.ok,
                } if hasattr(quality, 'metrics') else None,
            },
            processing_info={
                "processing_time_ms": round(processing_time_ms, 2),
                "ocr_used": False,
                "sources_used": ["heuristics"] + (["text_llm"] if video_text_result is not None and "error" not in video_text_result else []) + (["vision_llm"] if video_vision_result is not None and "error" not in video_vision_result else []),
                "fusion_enabled": video_fusion is not None,
                "require_agriculture": require_agriculture,
                "auto_route_models": auto_route_models,
                "routing": {
                    "video_mode": True,
                    "text_llm": {"requested": use_text_llm, "used": video_text_result is not None and "error" not in video_text_result, "reason": "video_text_llm_enabled" if use_text_llm else "disabled"},
                    "vision_llm": {"requested": use_vision, "used": video_vision_result is not None and "error" not in video_vision_result, "reasons": ["video_frame_sampling"] if video_vision_result and "error" not in video_vision_result else []},
                },
                "classification_confidence_threshold": classification_confidence_threshold,
                "stage_timings_ms": stage_timings_ms,
            },
        )

    if category_result.category == "Audio" and not asset.text.strip():
        processing_time_ms = (time.time() - start_time) * 1000
        agriculture_response = AgricultureRelevance(
            is_agriculture_related=False,
            confidence=0.0,
            score=0.0,
            method="audio_transcript_unavailable",
            lexicon_version="not_applicable",
            matched_terms=[],
            matched_buckets=[],
            matched_concepts=[],
            bucket_scores={},
            rationale="Audio transcription was unavailable or produced no usable transcript, so agriculture relevance could not be assessed reliably",
            stages_used=["audio_transcription"],
            stage_results=[
                {
                    "stage": "audio_transcription",
                    "available": bool(asset.meta.get("transcription_available", False)),
                    "used": bool(asset.meta.get("transcription_used", False)),
                    "is_agriculture_related": False,
                    "confidence": 0.0,
                    "rationale": asset.meta.get("transcription_rationale", "No usable transcript available"),
                    "details": {
                        "transcription_method": asset.meta.get("transcription_method"),
                        "transcription_model": asset.meta.get("transcription_model"),
                    },
                }
            ],
        )
        return ClassificationResponse(
            best_match=None,
            all_candidates=[],
            fusion=None,
            heuristics=None,
            vision_llm=None,
            text_llm=None,
            category_inference=category_response,
            agriculture_relevance=agriculture_response,
            classification_skipped=True,
            skip_reason="Audio transcription unavailable; agriculture and audio subtype classification skipped",
            total_candidates=0,
            confidence_threshold_met=False,
            document_info={
                "filename": filename,
                "pages": asset.units,
                "unit_label": asset.unit_label,
                "asset_type": asset.asset_type,
                "inferred_category": category_result.category,
                "source": asset.source,
                "text_length": len(asset.text),
                "text_quality": {
                    "chars": quality.metrics.get("chars"),
                    "letters": quality.metrics.get("letters"),
                    "letter_ratio": quality.metrics.get("letter_ratio"),
                    "ok": quality.ok,
                } if hasattr(quality, 'metrics') else None,
            },
            processing_info={
                "processing_time_ms": round(processing_time_ms, 2),
                "ocr_used": False,
                "sources_used": [],
                "fusion_enabled": False,
                "require_agriculture": require_agriculture,
                "auto_route_models": auto_route_models,
                "stage_timings_ms": stage_timings_ms,
            },
        )

    if require_agriculture and not agri_relevance.is_agriculture_related:
        processing_time_ms = (time.time() - start_time) * 1000
        return ClassificationResponse(
            best_match=None,
            all_candidates=[],
            fusion=None,
            heuristics=None,
            vision_llm=None,
            text_llm=None,
            category_inference=category_response,
            agriculture_relevance=agriculture_response,
            classification_skipped=True,
            skip_reason=f"{category_result.category} classified as non-agriculture; subcategory classification skipped",
            total_candidates=0,
            confidence_threshold_met=False,
            document_info={
                "filename": filename,
                "pages": asset.units,
                "unit_label": asset.unit_label,
                "asset_type": asset.asset_type,
                "inferred_category": category_result.category,
                "source": asset.source,
                "text_length": len(asset.text),
                "text_quality": {
                    "chars": quality.metrics.get("chars"),
                    "letters": quality.metrics.get("letters"),
                    "letter_ratio": quality.metrics.get("letter_ratio"),
                    "ok": quality.ok,
                } if hasattr(quality, 'metrics') else None,
            },
            processing_info={
                "processing_time_ms": round(processing_time_ms, 2),
                "ocr_used": asset.source == "ocr",
                "sources_used": [],
                "fusion_enabled": False,
                "require_agriculture": require_agriculture,
                "auto_route_models": auto_route_models,
                "stage_timings_ms": stage_timings_ms,
            },
        )

    if category_result.category == "Audio":
        stage_start = time.time()
        _, audio_scores = score_audio_subcategories(asset.text, asset.lines, filename=filename)
        audio_candidates = build_probability_distribution(audio_scores)
        add_contrastive_rationales(audio_candidates)
        audio_best = audio_candidates[0] if audio_candidates else None
        audio_heuristics_best = audio_best.model_copy(deep=True) if audio_best else None
        stage_timings_ms["audio_classification_ms"] = round((time.time() - stage_start) * 1000, 2)
        audio_text_result = None
        audio_text_source = None
        audio_fusion = None

        if use_text_llm and LLM_CONFIGURED:
            stage_start = time.time()
            try:
                from docint.llm.subcategory_classify import llm_classify_audio_subcategories_text

                llm_res = llm_classify_audio_subcategories_text(
                    asset.text,
                    base_url=LLM_BASE_URL,
                    api_key=LLM_API_KEY,
                    model=LLM_MODEL,
                    max_chars=12000,
                    temperature=0.2,
                )
                audio_text_source = convert_to_source_result(
                    subcategory_key=llm_res.subcategory_key,
                    confidence=llm_res.confidence,
                    probs=llm_res.probs,
                    source_name="text_llm",
                    rationale=llm_res.rationale,
                )
                audio_text_result = {
                    "subcategory_key": llm_res.subcategory_key,
                    "subcategory_name": llm_res.subcategory_name,
                    "confidence": round(llm_res.confidence, 4),
                    "rationale": llm_res.rationale,
                    "model": LLM_MODEL,
                }
            except Exception as e:
                audio_text_result = {"error": str(e), "model": LLM_MODEL}
            stage_timings_ms["audio_text_llm_ms"] = round((time.time() - stage_start) * 1000, 2)

        if audio_best and audio_text_source:
            name_to_audio_key = {subtype.name: key for key, subtype in AUDIO_SUBTYPES.items()}
            audio_heuristics_probs = {
                name_to_audio_key.get(candidate.subcategory_name, candidate.subcategory_name): candidate.probability
                for candidate in audio_candidates
            }
            audio_heuristics_source = convert_to_source_result(
                subcategory_key=name_to_audio_key.get(audio_best.subcategory_name, list(audio_heuristics_probs.keys())[0]),
                confidence=audio_best.confidence,
                probs=audio_heuristics_probs,
                source_name="heuristics",
                evidence_score=audio_best.evidence_score,
                rationale=audio_best.rationale,
            )
            fusion_result = intelligent_fusion(
                heuristics_result=audio_heuristics_source,
                vision_result=None,
                text_result=audio_text_source,
                strategy=FusionStrategy.CONFIDENCE_ADAPTIVE if fusion_strategy == "adaptive" else FusionStrategy.WEIGHTED,
                heuristics_alpha=heuristics_alpha,
                llm_alpha=1.0 - heuristics_alpha,
            )
            audio_fusion = FusionInfo(
                fused=True,
                strategy=fusion_result.fusion_strategy,
                weights=fusion_result.weights,
                agreement_score=fusion_result.agreement_score,
                rationale=fusion_result.rationale,
            )
            for candidate in audio_candidates:
                audio_key = name_to_audio_key.get(candidate.subcategory_name, candidate.subcategory_name)
                fused_prob = round(fusion_result.probs.get(audio_key, 0), 4)
                candidate.probability = fused_prob
                candidate.confidence = fused_prob
            audio_candidates.sort(key=lambda item: item.probability, reverse=True)
            for idx, candidate in enumerate(audio_candidates, start=1):
                candidate.rank = idx
            audio_best = audio_candidates[0] if audio_candidates else None
            add_contrastive_rationales(audio_candidates)

        processing_time_ms = (time.time() - start_time) * 1000
        return ClassificationResponse(
            best_match=audio_best,
            all_candidates=audio_candidates,
            fusion=audio_fusion,
            heuristics=audio_heuristics_best,
            vision_llm=None,
            text_llm=audio_text_result,
            category_inference=category_response,
            agriculture_relevance=agriculture_response,
            classification_skipped=False,
            skip_reason=None,
            total_candidates=len(audio_candidates),
            confidence_threshold_met=bool(audio_best and audio_best.confidence >= classification_confidence_threshold),
            document_info={
                "filename": filename,
                "pages": asset.units,
                "unit_label": asset.unit_label,
                "asset_type": asset.asset_type,
                "inferred_category": category_result.category,
                "source": asset.source,
                "text_length": len(asset.text),
                "text_quality": {
                    "chars": quality.metrics.get("chars"),
                    "letters": quality.metrics.get("letters"),
                    "letter_ratio": quality.metrics.get("letter_ratio"),
                    "ok": quality.ok,
                } if hasattr(quality, 'metrics') else None,
            },
            processing_info={
                "processing_time_ms": round(processing_time_ms, 2),
                "ocr_used": False,
                "sources_used": ["heuristics"] + (["text_llm"] if audio_text_result is not None and "error" not in audio_text_result else []),
                "fusion_enabled": audio_fusion is not None,
                "require_agriculture": require_agriculture,
                "auto_route_models": auto_route_models,
                "routing": {
                    "audio_mode": True,
                    "text_llm": {"requested": use_text_llm, "used": audio_text_result is not None and "error" not in audio_text_result, "reason": "audio_text_llm_enabled" if use_text_llm else "disabled"},
                    "vision_llm": {"requested": use_vision, "used": False, "reasons": ["audio_vision_not_applicable"]},
                },
                "classification_confidence_threshold": classification_confidence_threshold,
                "stage_timings_ms": stage_timings_ms,
            },
        )

    if category_result.category == "Dataset":
        stage_start = time.time()
        _, dataset_scores = score_dataset_subcategories(asset.text, asset.lines)
        dataset_candidates = build_probability_distribution(dataset_scores)
        add_contrastive_rationales(dataset_candidates)
        dataset_best = dataset_candidates[0] if dataset_candidates else None
        dataset_heuristics_best = dataset_best.model_copy(deep=True) if dataset_best else None
        stage_timings_ms["dataset_classification_ms"] = round((time.time() - stage_start) * 1000, 2)
        dataset_text_result = None
        dataset_text_source = None
        dataset_fusion = None

        if use_text_llm and LLM_CONFIGURED:
            stage_start = time.time()
            try:
                from docint.llm.subcategory_classify import llm_classify_dataset_subcategories_text

                llm_res = llm_classify_dataset_subcategories_text(
                    asset.text,
                    base_url=LLM_BASE_URL,
                    api_key=LLM_API_KEY,
                    model=LLM_MODEL,
                    max_chars=12000,
                    temperature=0.2,
                )
                dataset_text_source = convert_to_source_result(
                    subcategory_key=llm_res.subcategory_key,
                    confidence=llm_res.confidence,
                    probs=llm_res.probs,
                    source_name="text_llm",
                    rationale=llm_res.rationale,
                )
                dataset_text_result = {
                    "subcategory_key": llm_res.subcategory_key,
                    "subcategory_name": llm_res.subcategory_name,
                    "confidence": round(llm_res.confidence, 4),
                    "rationale": llm_res.rationale,
                    "model": LLM_MODEL,
                }
            except Exception as e:
                dataset_text_result = {"error": str(e), "model": LLM_MODEL}
            stage_timings_ms["dataset_text_llm_ms"] = round((time.time() - stage_start) * 1000, 2)

        if dataset_best and dataset_text_source:
            name_to_dataset_key = {subtype.name: key for key, subtype in DATASET_SUBTYPES.items()}
            dataset_heuristics_probs = {
                name_to_dataset_key.get(candidate.subcategory_name, candidate.subcategory_name): candidate.probability
                for candidate in dataset_candidates
            }
            dataset_heuristics_source = convert_to_source_result(
                subcategory_key=name_to_dataset_key.get(dataset_best.subcategory_name, list(dataset_heuristics_probs.keys())[0]),
                confidence=dataset_best.confidence,
                probs=dataset_heuristics_probs,
                source_name="heuristics",
                evidence_score=dataset_best.evidence_score,
                rationale=dataset_best.rationale,
            )
            fusion_result = intelligent_fusion(
                heuristics_result=dataset_heuristics_source,
                vision_result=None,
                text_result=dataset_text_source,
                strategy=FusionStrategy.CONFIDENCE_ADAPTIVE if fusion_strategy == "adaptive" else FusionStrategy.WEIGHTED,
                heuristics_alpha=heuristics_alpha,
                llm_alpha=1.0 - heuristics_alpha,
            )
            dataset_fusion = FusionInfo(
                fused=True,
                strategy=fusion_result.fusion_strategy,
                weights=fusion_result.weights,
                agreement_score=fusion_result.agreement_score,
                rationale=fusion_result.rationale,
            )
            for candidate in dataset_candidates:
                dataset_key = name_to_dataset_key.get(candidate.subcategory_name, candidate.subcategory_name)
                fused_prob = round(fusion_result.probs.get(dataset_key, 0), 4)
                candidate.probability = fused_prob
                candidate.confidence = fused_prob
            dataset_candidates.sort(key=lambda item: item.probability, reverse=True)
            for idx, candidate in enumerate(dataset_candidates, start=1):
                candidate.rank = idx
            dataset_best = dataset_candidates[0] if dataset_candidates else None
            add_contrastive_rationales(dataset_candidates)
        processing_time_ms = (time.time() - start_time) * 1000
        return ClassificationResponse(
            best_match=dataset_best,
            all_candidates=dataset_candidates,
            fusion=dataset_fusion,
            heuristics=dataset_heuristics_best,
            vision_llm=None,
            text_llm=dataset_text_result,
            category_inference=category_response,
            agriculture_relevance=agriculture_response,
            classification_skipped=False,
            skip_reason=None,
            total_candidates=len(dataset_candidates),
            confidence_threshold_met=bool(dataset_best and dataset_best.confidence >= classification_confidence_threshold),
            document_info={
                "filename": filename,
                "pages": asset.units,
                "unit_label": asset.unit_label,
                "asset_type": asset.asset_type,
                "inferred_category": category_result.category,
                "source": asset.source,
                "text_length": len(asset.text),
                "text_quality": {
                    "chars": quality.metrics.get("chars"),
                    "letters": quality.metrics.get("letters"),
                    "letter_ratio": quality.metrics.get("letter_ratio"),
                    "ok": quality.ok,
                } if hasattr(quality, 'metrics') else None,
            },
            processing_info={
                "processing_time_ms": round(processing_time_ms, 2),
                "ocr_used": asset.source == "ocr",
                "sources_used": ["heuristics"] + (["text_llm"] if dataset_text_result is not None and "error" not in dataset_text_result else []),
                "fusion_enabled": dataset_fusion is not None,
                "require_agriculture": require_agriculture,
                "auto_route_models": auto_route_models,
                "routing": {
                    "dataset_mode": True,
                    "text_llm": {"requested": use_text_llm, "used": dataset_text_result is not None and "error" not in dataset_text_result, "reason": "dataset_text_llm_enabled" if use_text_llm else "disabled"},
                    "vision_llm": {"requested": use_vision, "used": False, "reasons": ["dataset_vision_not_yet_enabled"]},
                },
                "classification_confidence_threshold": classification_confidence_threshold,
                "stage_timings_ms": stage_timings_ms,
            },
        )

    if category_result.category != "Document":
        processing_time_ms = (time.time() - start_time) * 1000
        return ClassificationResponse(
            best_match=None,
            all_candidates=[],
            fusion=None,
            heuristics=None,
            vision_llm=None,
            text_llm=None,
            category_inference=category_response,
            agriculture_relevance=agriculture_response,
            classification_skipped=True,
            skip_reason=f"Inferred category is {category_result.category}; document subcategory classification skipped",
            total_candidates=0,
            confidence_threshold_met=False,
            document_info={
                "filename": filename,
                "pages": asset.units,
                "unit_label": asset.unit_label,
                "asset_type": asset.asset_type,
                "inferred_category": category_result.category,
                "source": asset.source,
                "text_length": len(asset.text),
                "text_quality": {
                    "chars": quality.metrics.get("chars"),
                    "letters": quality.metrics.get("letters"),
                    "letter_ratio": quality.metrics.get("letter_ratio"),
                    "ok": quality.ok,
                } if hasattr(quality, 'metrics') else None,
            },
            processing_info={
                "processing_time_ms": round(processing_time_ms, 2),
                "ocr_used": asset.source == "ocr",
                "sources_used": [],
                "fusion_enabled": False,
                "require_agriculture": require_agriculture,
                "auto_route_models": auto_route_models,
                "stage_timings_ms": stage_timings_ms,
            },
        )

    # 4) Extract features and compute rubric scores
    stage_start = time.time()
    sections = count_sections(asset.lines)
    cites = detect_citations(asset.text, has_references_heading=sections.present.get("references", False))
    kw = count_keywords(asset.text)
    
    r_imrad = score_imrad(sections)
    r_cites = score_citations(cites, text_len=len(asset.text))
    r_deliv = score_deliverable(kw, sections=sections)
    r_ped = score_pedagogy(kw)
    r_proc = score_procedure(kw)
    rubric_scores = {
        "imrad": r_imrad.score,
        "citations": r_cites.score,
        "deliverable": r_deliv.score,
        "pedagogy": r_ped.score,
        "procedure": r_proc.score,
    }
    stage_timings_ms["feature_and_rubric_ms"] = round((time.time() - stage_start) * 1000, 2)
    
    # 4) Evidence-based classification
    stage_start = time.time()
    best_match, all_scores, _ = score_subcategories(
        text=asset.text,
        lines=asset.lines,
        page_count=asset.units,
        sections=sections,
        rubric_scores=rubric_scores,
        parent_type_filter=None,
    )
    
    candidates = build_probability_distribution(all_scores)
    add_contrastive_rationales(candidates)
    best_candidate = candidates[0] if candidates else None
    heuristics_best = best_candidate.model_copy(deep=True) if best_candidate else None
    final_candidates = [c.model_copy(deep=True) for c in candidates]
    key_by_name = _subcategory_key_by_name()
    
    # Convert heuristics to SourceResult for fusion
    heuristics_probs = {
        key_by_name.get(c.subcategory_name, c.subcategory_name): c.probability
        for c in candidates
    }
    heuristics_source = convert_to_source_result(
        subcategory_key=key_by_name.get(best_candidate.subcategory_name, "") if best_candidate else "",
        confidence=best_candidate.confidence if best_candidate else 0.0,
        probs=heuristics_probs,
        source_name="heuristics",
        evidence_score=best_candidate.evidence_score if best_candidate else 0.0,
        rationale=best_candidate.rationale if best_candidate else "",
    )
    stage_timings_ms["heuristics_classification_ms"] = round((time.time() - stage_start) * 1000, 2)
    
    # 5) Optional LLM analysis
    vision_source = None
    text_source = None
    llm_results = {}
    routing_info: Dict[str, Any] = {
        "auto_route_models": auto_route_models,
        "text_llm": {"requested": use_text_llm, "used": False, "reason": None},
        "vision_llm": {"requested": use_vision, "used": False, "reasons": []},
        "top_candidate_gap": _top_probability_gap(final_candidates),
    }

    should_use_text_llm = _should_run_text_llm(
        use_text_llm=use_text_llm,
        is_agriculture_related=agri_relevance.is_agriculture_related,
    ) if auto_route_models else (use_text_llm and LLM_CONFIGURED)
    routing_info["text_llm"]["reason"] = (
        "agriculture_related_default_text_stage"
        if should_use_text_llm and auto_route_models
        else ("manual_request" if should_use_text_llm else "disabled_or_not_configured")
    )

    if should_use_text_llm:
        stage_start = time.time()
        try:
            from docint.llm.subcategory_classify import llm_classify_subcategories_text
            
            llm_res = llm_classify_subcategories_text(
                asset.text,
                base_url=LLM_BASE_URL,
                api_key=LLM_API_KEY,
                model=LLM_MODEL,
                max_chars=15000,
                temperature=0.2,
            )
            
            text_source = convert_to_source_result(
                subcategory_key=llm_res.subcategory_key,
                confidence=llm_res.confidence,
                probs=llm_res.probs,
                source_name="text_llm",
                rationale=llm_res.rationale,
            )
            
            llm_results["text"] = {
                "subcategory_key": llm_res.subcategory_key,
                "subcategory_name": llm_res.subcategory_name,
                "confidence": round(llm_res.confidence, 4),
                "rationale": llm_res.rationale,
                "model": LLM_MODEL,
            }
            routing_info["text_llm"]["used"] = True
        except Exception as e:
            llm_results["text"] = {"error": str(e), "model": LLM_MODEL}
        stage_timings_ms["text_llm_ms"] = round((time.time() - stage_start) * 1000, 2)

    should_use_vision, vision_reasons = _should_run_vision(
        use_vision=use_vision and asset.asset_type == "pdf",
        ocr_used=asset.source == "ocr",
        text_quality_ok_flag=quality.ok,
        heuristics_source=heuristics_source,
        best_candidate=best_candidate,
        current_candidates=final_candidates,
        text_source=text_source,
        confidence_threshold=vision_trigger_threshold,
        gap_threshold=candidate_gap_threshold,
    ) if auto_route_models else ((use_vision and bool(VISION_LLM_BASE_URL)), ["manual_request"] if use_vision and VISION_LLM_BASE_URL else [])
    routing_info["vision_llm"]["reasons"] = vision_reasons

    if should_use_vision:
        stage_start = time.time()
        try:
            from docint.llm.subcategory_classify import llm_classify_subcategories_vision
            
            llm_res = llm_classify_subcategories_vision(
                file_path,
                base_url=VISION_LLM_BASE_URL,
                api_key=VISION_LLM_API_KEY,
                model=VISION_LLM_MODEL,
                max_total_pages=min(vision_max_pages, 8),
                temperature=0.2,
            )
            
            vision_source = convert_to_source_result(
                subcategory_key=llm_res.subcategory_key,
                confidence=llm_res.confidence,
                probs=llm_res.probs,
                source_name="vision_llm",
                rationale=llm_res.rationale,
            )
            
            llm_results["vision"] = {
                "subcategory_key": llm_res.subcategory_key,
                "subcategory_name": llm_res.subcategory_name,
                "confidence": round(llm_res.confidence, 4),
                "rationale": llm_res.rationale,
                "model": VISION_LLM_MODEL,
            }
            routing_info["vision_llm"]["used"] = True
        except Exception as e:
            llm_results["vision"] = {"error": str(e), "model": VISION_LLM_MODEL}
        stage_timings_ms["vision_llm_ms"] = round((time.time() - stage_start) * 1000, 2)
    
    # 6) Intelligent fusion if multiple sources
    fusion_info = None
    final_best = final_candidates[0] if final_candidates else None
    
    sources_for_fusion = [heuristics_source]
    if vision_source:
        sources_for_fusion.append(vision_source)
    if text_source:
        sources_for_fusion.append(text_source)
    
    if len(sources_for_fusion) > 1:
        stage_start = time.time()
        strategy_map = {
            "weighted": FusionStrategy.WEIGHTED,
            "adaptive": FusionStrategy.CONFIDENCE_ADAPTIVE,
            "agreement": FusionStrategy.AGREEMENT_BASED,
            "cascade": FusionStrategy.CASCADE,
        }
        strategy = strategy_map.get(fusion_strategy, FusionStrategy.CONFIDENCE_ADAPTIVE)
        
        fusion_result = intelligent_fusion(
            heuristics_result=heuristics_source,
            vision_result=vision_source,
            text_result=text_source,
            strategy=strategy,
            heuristics_alpha=heuristics_alpha,
            llm_alpha=1.0 - heuristics_alpha,
        )
        
        # Convert fusion result to candidate format
        fusion_info = FusionInfo(
            fused=True,
            strategy=fusion_result.fusion_strategy,
            weights=fusion_result.weights,
            agreement_score=fusion_result.agreement_score,
            rationale=fusion_result.rationale,
        )
        
        # Re-rank candidates based on fusion probabilities. Fusion stores
        # probabilities by internal subcategory key, not display name.
        for c in final_candidates:
            subcat_key = key_by_name.get(c.subcategory_name, c.subcategory_name)
            fused_prob = round(fusion_result.probs.get(subcat_key, 0), 4)
            c.probability = fused_prob
            c.confidence = fused_prob

        # Sort by new probabilities
        final_candidates.sort(key=lambda x: x.probability, reverse=True)
        for i, c in enumerate(final_candidates, 1):
            c.rank = i
        add_fusion_explanations(
            final_candidates,
            fusion_result,
            {
                "heuristics": heuristics_source,
                "vision_llm": vision_source,
                "text_llm": text_source,
            },
        )
        add_contrastive_rationales(final_candidates)

        final_best = final_candidates[0]
        stage_timings_ms["fusion_ms"] = round((time.time() - stage_start) * 1000, 2)
    
    threshold_met = final_best.confidence >= classification_confidence_threshold if final_best else False
    
    processing_time_ms = (time.time() - start_time) * 1000
    
    return ClassificationResponse(
        best_match=final_best,
        all_candidates=final_candidates,
        fusion=fusion_info,
        heuristics=heuristics_best,
        vision_llm=llm_results.get("vision"),
        text_llm=llm_results.get("text"),
        category_inference=category_response,
        agriculture_relevance=agriculture_response,
        classification_skipped=False,
        skip_reason=None,
        total_candidates=len(final_candidates),
        confidence_threshold_met=threshold_met,
        document_info={
            "filename": filename,
            "pages": asset.units,
            "unit_label": asset.unit_label,
            "asset_type": asset.asset_type,
            "inferred_category": category_result.category,
            "source": asset.source,
            "text_length": len(asset.text),
            "text_quality": {
                "chars": quality.metrics.get("chars"),
                "letters": quality.metrics.get("letters"),
                "letter_ratio": quality.metrics.get("letter_ratio"),
                "ok": quality.ok,
            } if hasattr(quality, 'metrics') else None,
        },
        processing_info={
            "processing_time_ms": round(processing_time_ms, 2),
            "ocr_used": asset.source == "ocr",
            "sources_used": [s.source_name for s in sources_for_fusion],
            "fusion_enabled": fusion_info is not None,
            "require_agriculture": require_agriculture,
            "auto_route_models": auto_route_models,
            "routing": routing_info,
            "classification_confidence_threshold": classification_confidence_threshold,
            "vision_trigger_threshold": vision_trigger_threshold,
            "candidate_gap_threshold": candidate_gap_threshold,
            "stage_timings_ms": stage_timings_ms,
        },
    )


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.middleware("http")
async def auth_middleware(request, call_next):
    """
    Global middleware to enforce Basic Auth on all requests.
    Skips auth for OPTIONS requests (CORS preflight) and health checks.
    """
    # Skip auth if not enabled
    if not AUTH_ENABLED:
        return await call_next(request)
    
    # Skip auth for OPTIONS requests (CORS preflight)
    if request.method == "OPTIONS":
        return await call_next(request)
    
    # Skip auth for health check endpoint
    if request.url.path == "/health":
        return await call_next(request)
    
    # Get Authorization header
    auth_header = request.headers.get("Authorization")
    
    if not auth_header or not auth_header.startswith("Basic "):
        return JSONResponse(
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="KO Classifier API"'},
            content={"detail": "Authentication required"},
        )
    
    # Decode credentials
    try:
        import base64
        encoded_credentials = auth_header.split(" ", 1)[1]
        decoded = base64.b64decode(encoded_credentials).decode("utf-8")
        username, password = decoded.split(":", 1)
        
        # Verify
        stored_password = AUTHORIZED_USERS.get(username)
        if stored_password is None or not secrets.compare_digest(password, stored_password):
            return JSONResponse(
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="KO Classifier API"'},
                content={"detail": "Invalid credentials"},
            )
        
        # Store username in request state for later use
        request.state.username = username
        
    except Exception:
        return JSONResponse(
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="KO Classifier API"'},
            content={"detail": "Invalid authentication format"},
        )
    
    return await call_next(request)


@app.get("/")
async def root(request: Request):
    """Root endpoint with API info."""
    username = getattr(request.state, 'username', 'anonymous')
    return {
        "name": "KO Classifier API",
        "version": "2.0.0",
        "authenticated_user": username,
        "auth_enabled": AUTH_ENABLED,
        "features": ["heuristics", "vision_llm", "text_llm", "intelligent_fusion"],
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health(request: Request):
    """Health check endpoint."""
    from urllib.parse import urlparse
    
    def mask_url(url: str) -> Optional[str]:
        if not url:
            return None
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"
    
    username = getattr(request.state, 'username', 'anonymous')
    
    return {
        "status": "ok",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "authenticated_user": username,
        "auth_enabled": AUTH_ENABLED,
        "models": {
            "heuristics": {"available": True},
            "text_llm": {
                "configured": LLM_CONFIGURED,
                "model": LLM_MODEL if LLM_CONFIGURED else None,
                "base_url": mask_url(LLM_BASE_URL),
            },
            "vision_llm": {
                "configured": bool(VISION_LLM_BASE_URL),
                "model": VISION_LLM_MODEL if VISION_LLM_BASE_URL else None,
                "base_url": mask_url(VISION_LLM_BASE_URL),
            },
            "audio_transcription": {
                "enabled": MEDIA_TRANSCRIBER_ENABLED,
                "configured": AUDIO_TRANSCRIPTION_CONFIGURED,
                "model": MEDIA_TRANSCRIBER_WHISPER_MODEL if AUDIO_TRANSCRIPTION_CONFIGURED else None,
                "mode": MEDIA_TRANSCRIBER_MODE if AUDIO_TRANSCRIPTION_CONFIGURED else None,
                "base_url": mask_url(MEDIA_TRANSCRIBER_BASE_URL),
                "basic_auth_configured": bool(MEDIA_TRANSCRIBER_BASIC_USER and MEDIA_TRANSCRIBER_BASIC_PASS),
                "api_key_configured": bool(MEDIA_TRANSCRIBER_API_KEY),
            },
            "video_tooling": {
                "ffmpeg_available": FFMPEG_AVAILABLE,
                "ffprobe_available": FFPROBE_AVAILABLE,
                "frame_sampling_ready": FFMPEG_AVAILABLE and FFPROBE_AVAILABLE,
                "audio_extract_ready": FFMPEG_AVAILABLE,
            },
        },
    }


@app.post("/classify", response_model=ClassificationResponse)
async def classify_endpoint(
    request: Request,
    file: UploadFile = File(..., description="Supported KO asset file to classify"),
    require_agriculture: bool = Query(True, description="Skip subtype classification for assets assessed as non-agriculture"),
    auto_route_models: bool = Query(True, description="Automatically decide whether text and vision models should be used"),
    use_vision: bool = Query(True, description="Allow Vision LLM (InternVL) when routing decides it is needed"),
    use_text_llm: bool = Query(True, description="Allow Text LLM (Qwen) for agriculture-related documents"),
    heuristics_alpha: float = Query(
        0.4,
        ge=0.0,
        le=1.0,
        description="Weight for heuristics (0.4 = 40% heuristics, 60% LLM)"
    ),
    classification_confidence_threshold: float = Query(0.35, ge=0.0, le=1.0, description="Confidence threshold used to mark subtype classification as strong enough"),
    vision_trigger_threshold: float = Query(0.6, ge=0.0, le=1.0, description="Subcategory confidence threshold below which vision may be triggered"),
    candidate_gap_threshold: float = Query(0.12, ge=0.0, le=1.0, description="Top-candidate probability gap below which vision may be triggered"),
    fusion_strategy: str = Query(
        "adaptive",
        description="Fusion strategy: weighted, adaptive, agreement, cascade"
    ),
    vision_max_pages: int = Query(
        8,
        ge=1,
        le=12,
        description="Maximum representative pages sampled for vision analysis; pages are sampled deterministically rather than scanning the whole document"
    ),
    ocr_lang: Optional[str] = Query(
        None,
        description="Optional Tesseract OCR language bundle used only when OCR fallback is triggered; if omitted, the configured multilingual default is used"
    ),
    ocr_max_pages: int = Query(
        5,
        ge=1,
        le=50,
        description="Maximum pages sent through OCR fallback when extracted text quality is poor"
    ),
):
    """Classify a supported KO asset file with optional model routing."""
    filename = file.filename or "unknown.pdf"
    suffix = os.path.splitext(filename)[1].lower()
    audio_suffixes = {".mp3", ".wav", ".m4a"}
    video_suffixes = {".mp4", ".avi", ".mov", ".wmv", ".mpeg", ".mpg", ".mkv", ".flv", ".webm", ".3gp", ".mts", ".m2ts", ".vob", ".rmvb"}
    content_length = request.headers.get("content-length")

    if suffix not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Supported types: {', '.join(sorted(SUPPORTED_DOCUMENT_EXTENSIONS))}",
        )

    if content_length:
        try:
            request_size_bytes = int(content_length)
            if request_size_bytes > MAX_REQUEST_BODY_MB * 1024 * 1024:
                raise HTTPException(
                    status_code=413,
                    detail=f"Request body too large ({round(request_size_bytes / (1024 * 1024), 2)} MB). Maximum allowed is {MAX_REQUEST_BODY_MB} MB.",
                )
        except ValueError:
            pass
    
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    file_size_bytes = len(contents)
    file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
    if suffix in audio_suffixes and file_size_bytes > MAX_AUDIO_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file too large ({file_size_mb} MB). Maximum allowed is {MAX_AUDIO_UPLOAD_SIZE_MB} MB.",
        )
    if suffix in video_suffixes and file_size_bytes > MAX_VIDEO_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"Video file too large ({file_size_mb} MB). Maximum allowed is {MAX_VIDEO_UPLOAD_SIZE_MB} MB.",
        )
    
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".bin") as tmp:
            tmp_path = tmp.name
            tmp.write(contents)

        if suffix in audio_suffixes | video_suffixes:
            duration_sec = media_duration_seconds(tmp_path)
            max_duration_sec = MAX_AUDIO_DURATION_SEC if suffix in audio_suffixes else MAX_VIDEO_DURATION_SEC
            if duration_sec and duration_sec > max_duration_sec:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"{'Audio' if suffix in audio_suffixes else 'Video'} duration too long "
                        f"({round(duration_sec, 1)} seconds). Maximum allowed is {max_duration_sec} seconds."
                    ),
                )
        
        result = classify_document(
            file_path=tmp_path,
            filename=filename,
            require_agriculture=require_agriculture,
            auto_route_models=auto_route_models,
            use_vision=use_vision,
            use_text_llm=use_text_llm,
            heuristics_alpha=heuristics_alpha,
            classification_confidence_threshold=classification_confidence_threshold,
            vision_trigger_threshold=vision_trigger_threshold,
            candidate_gap_threshold=candidate_gap_threshold,
            fusion_strategy=fusion_strategy,
            vision_max_pages=vision_max_pages,
            ocr_lang=ocr_lang or ALL_OCR_LANGS,
            ocr_max_pages=ocr_max_pages,
        )
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


@app.get("/subcategories")
async def list_subcategories(request: Request):
    """List all subcategory definitions."""
    from docint.rubrics.subcategories import get_all_detectable_features
    criteria = get_subcategory_criteria()

    subcats = {}
    for key, subcat in SUBCATEGORIES.items():
        subcats[key] = {
            "id": subcat.id,
            "name": subcat.name,
            "description": subcat.description,
            "parent_type": subcat.parent_type.value,
            "features": [f.name for f in subcat.detectable_features],
            "criteria": criteria.get(key),
        }
    
    return {
        "subcategories": subcats,
        "total": len(subcats),
        "all_features": get_all_detectable_features(),
    }


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
