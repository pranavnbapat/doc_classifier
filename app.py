# app.py
"""
KO Classifier API - FastAPI application for document subcategory classification.

This API provides evidence-based document classification with optional LLM enhancement.
It supports both text-based (Qwen) and vision-based (InternVL) LLMs with intelligent
fusion of results.

Run with: uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Optional
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Depends, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import secrets

from docint.extract.pdf_text import extract_pdf_text
from docint.extract.quality import text_quality_ok
from docint.extract.ocr import ocr_pdf
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
    Evidence-based document subcategory classification with intelligent LLM fusion.
    
    ## Features
    
    * **Evidence-Based Classification**: 24+ measurable features, 11 subcategories
    * **Vision LLM (InternVL)**: Analyzes PDF pages as images
    * **Text LLM (Qwen)**: Analyzes extracted text
    * **Intelligent Fusion**: Combines results using confidence-adaptive weighting
    
    ## Fusion Strategy
    
    When multiple sources are enabled (heuristics + LLM), results are fused using:
    - **Confidence-adaptive weighting**: Higher confidence sources get more weight
    - **Agreement-based boosting**: Sources that agree get bonus weight
    - **Configurable alpha**: Control base weight between heuristics and LLM
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


def classify_document(
    pdf_path: str,
    filename: str,
    require_agriculture: bool = True,
    use_vision: bool = False,
    use_text_llm: bool = False,
    heuristics_alpha: float = 0.4,
    fusion_strategy: str = "adaptive",
    vision_max_pages: int = 20,
    ocr_lang: str = ALL_OCR_LANGS,
    ocr_max_pages: int = 10,
) -> ClassificationResponse:
    """
    Main classification function with optional LLM fusion.
    
    Args:
        pdf_path: Path to PDF file
        filename: Original filename
        require_agriculture: Whether to stop early for confident non-agriculture PDFs
        use_vision: Whether to use Vision LLM
        use_text_llm: Whether to use Text LLM
        heuristics_alpha: Weight for heuristics (0.0-1.0), LLM gets (1-alpha)
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
    doc = extract_pdf_text(pdf_path, max_pages=None)
    stage_timings_ms["text_extraction_ms"] = round((time.time() - stage_start) * 1000, 2)
    
    # 2) OCR fallback if needed
    stage_start = time.time()
    quality = text_quality_ok(doc.text)
    if not quality.ok:
        ocr_doc = ocr_pdf(pdf_path, max_pages=ocr_max_pages, lang=ocr_lang)
        from dataclasses import replace
        doc = replace(
            doc,
            text=ocr_doc.text,
            lines=ocr_doc.lines,
            source="ocr",
        )
    stage_timings_ms["ocr_fallback_ms"] = round((time.time() - stage_start) * 1000, 2)
    
    # 3) Agriculture gate before the heavier classification pipeline
    stage_start = time.time()
    agri_relevance = assess_agriculture_relevance_staged(
        doc.text,
        lines=doc.lines,
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

    if require_agriculture and not agri_relevance.is_agriculture_related:
        processing_time_ms = (time.time() - start_time) * 1000
        return ClassificationResponse(
            best_match=None,
            all_candidates=[],
            fusion=None,
            heuristics=None,
            vision_llm=None,
            text_llm=None,
            agriculture_relevance=agriculture_response,
            classification_skipped=True,
            skip_reason="Document classified as non-agriculture; subcategory classification skipped",
            total_candidates=0,
            confidence_threshold_met=False,
            document_info={
                "filename": filename,
                "pages": doc.pages,
                "source": doc.source,
                "text_length": len(doc.text),
                "text_quality": {
                    "chars": quality.metrics.get("chars"),
                    "letters": quality.metrics.get("letters"),
                    "letter_ratio": quality.metrics.get("letter_ratio"),
                    "ok": quality.ok,
                } if hasattr(quality, 'metrics') else None,
            },
            processing_info={
                "processing_time_ms": round(processing_time_ms, 2),
                "ocr_used": doc.source == "ocr",
                "sources_used": [],
                "fusion_enabled": False,
                "require_agriculture": require_agriculture,
                "stage_timings_ms": stage_timings_ms,
            },
        )

    # 4) Extract features and compute rubric scores
    stage_start = time.time()
    sections = count_sections(doc.lines)
    cites = detect_citations(doc.text, has_references_heading=sections.present.get("references", False))
    kw = count_keywords(doc.text)
    
    r_imrad = score_imrad(sections)
    r_cites = score_citations(cites, text_len=len(doc.text))
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
        text=doc.text,
        lines=doc.lines,
        page_count=doc.pages,
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
    
    # Vision LLM
    if use_vision and VISION_LLM_BASE_URL:
        stage_start = time.time()
        try:
            from docint.llm.subcategory_classify import llm_classify_subcategories_vision
            
            llm_res = llm_classify_subcategories_vision(
                pdf_path,
                base_url=VISION_LLM_BASE_URL,
                api_key=VISION_LLM_API_KEY,
                model=VISION_LLM_MODEL,
                window_size=4,
                overlap=2,
                max_total_pages=min(vision_max_pages, 50),
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
        except Exception as e:
            llm_results["vision"] = {"error": str(e), "model": VISION_LLM_MODEL}
        stage_timings_ms["vision_llm_ms"] = round((time.time() - stage_start) * 1000, 2)
    
    # Text LLM
    if use_text_llm and LLM_CONFIGURED:
        stage_start = time.time()
        try:
            from docint.llm.subcategory_classify import llm_classify_subcategories_text
            
            llm_res = llm_classify_subcategories_text(
                doc.text,
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
        except Exception as e:
            llm_results["text"] = {"error": str(e), "model": LLM_MODEL}
        stage_timings_ms["text_llm_ms"] = round((time.time() - stage_start) * 1000, 2)
    
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
    
    threshold_met = final_best.confidence >= 0.35 if final_best else False
    
    processing_time_ms = (time.time() - start_time) * 1000
    
    return ClassificationResponse(
        best_match=final_best,
        all_candidates=final_candidates,
        fusion=fusion_info,
        heuristics=heuristics_best,
        vision_llm=llm_results.get("vision"),
        text_llm=llm_results.get("text"),
        agriculture_relevance=agriculture_response,
        classification_skipped=False,
        skip_reason=None,
        total_candidates=len(final_candidates),
        confidence_threshold_met=threshold_met,
        document_info={
            "filename": filename,
            "pages": doc.pages,
            "source": doc.source,
            "text_length": len(doc.text),
            "text_quality": {
                "chars": quality.metrics.get("chars"),
                "letters": quality.metrics.get("letters"),
                "letter_ratio": quality.metrics.get("letter_ratio"),
                "ok": quality.ok,
            } if hasattr(quality, 'metrics') else None,
        },
        processing_info={
            "processing_time_ms": round(processing_time_ms, 2),
            "ocr_used": doc.source == "ocr",
            "sources_used": [s.source_name for s in sources_for_fusion],
            "fusion_enabled": fusion_info is not None,
            "require_agriculture": require_agriculture,
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
        },
    }


@app.post("/classify", response_model=ClassificationResponse)
async def classify_endpoint(
    request: Request,
    file: UploadFile = File(..., description="PDF file to classify"),
    require_agriculture: bool = Query(True, description="Skip subcategory classification for confident non-agriculture PDFs"),
    use_vision: bool = Query(False, description="Use Vision LLM (InternVL)"),
    use_text_llm: bool = Query(False, description="Use Text LLM (Qwen)"),
    heuristics_alpha: float = Query(
        0.4,
        ge=0.0,
        le=1.0,
        description="Weight for heuristics (0.4 = 40% heuristics, 60% LLM)"
    ),
    fusion_strategy: str = Query(
        "adaptive",
        description="Fusion strategy: weighted, adaptive, agreement, cascade"
    ),
    vision_max_pages: int = Query(20, ge=1, le=50),
    ocr_lang: str = Query(ALL_OCR_LANGS),
    ocr_max_pages: int = Query(10, ge=1, le=200),
):
    """Classify a PDF document with optional LLM fusion."""
    filename = file.filename or "unknown.pdf"
    
    if not filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=415, detail="Only PDF files supported")
    
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")
    
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp_path = tmp.name
            tmp.write(contents)
        
        result = classify_document(
            pdf_path=tmp_path,
            filename=filename,
            require_agriculture=require_agriculture,
            use_vision=use_vision,
            use_text_llm=use_text_llm,
            heuristics_alpha=heuristics_alpha,
            fusion_strategy=fusion_strategy,
            vision_max_pages=vision_max_pages,
            ocr_lang=ocr_lang,
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
