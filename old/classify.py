# docint/pipeline/classify.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

from docint.extract.pdf_text import extract_pdf_text, ExtractedDoc
from docint.extract.quality import text_quality_ok
from docint.extract.ocr import ocr_pdf

from docint.features.citations import detect_citations
from docint.features.doccontrol import detect_doccontrol
from docint.features.sections import count_sections
from docint.features.keywords import count_keywords, bucket_score

from docint.rubrics.imrad import score_imrad
from docint.rubrics.citations import score_citations
from docint.rubrics.deliverable import score_deliverable
from docint.rubrics.pedagogy import score_pedagogy
from docint.rubrics.procedure import score_procedure
from docint.rubrics.subcategories import ParentType, get_subcategories_by_parent
from docint.rubrics.subcategory_scorer import (
    SubcategoryClassifier, SubcategoryScore,
    score_subcategories
)

from docint.llm.classify import llm_ensemble
from docint.fusion.combine import fuse_probs


@dataclass
class ClassificationResult:
    label: str
    confidence: float
    probs: Dict[str, float]
    reasons: Dict[str, Any]
    subcategory: Optional[SubcategoryScore] = None
    subcategory_scores: Optional[List[SubcategoryScore]] = None


def _heuristic_probs_from_rubrics(
    *,
    imrad: float,
    cites: float,
    deliverable: float,
    pedagogy: float,
    procedure: float,
    policy_hint: float = 0.0,
) -> Dict[str, float]:
    """
    Convert rubric scores (0..1) into a probability-like distribution.

    This is intentionally simple and tunable.
    You can later replace this with logistic regression on labelled data.
    """
    # Research tends to be structure + citations
    scientific = 0.55 * imrad + 0.45 * cites

    # Deliverables have their own cues; can also have exec summary/appendix
    deliver = deliverable

    # Educational signals are quite distinct
    edu = pedagogy

    # Practice oriented is procedural cues
    practice = procedure

    # Policy guidance: not fully implemented as a rubric yet; accept a hint input
    policy = max(0.0, policy_hint)

    raw = {
        "scientific_research": scientific,
        "deliverable_report": deliver,
        "educational": edu,
        "practice_oriented": practice,
        "policy_guidance": policy,
    }

    # Soft floor so nothing becomes exactly zero (helps fusion stability)
    eps = 1e-4
    raw = {k: float(v) + eps for k, v in raw.items()}

    s = sum(raw.values()) or 1.0
    return {k: v / s for k, v in raw.items()}


def _dampen_probs_by_evidence(
    probs: Dict[str, float],
    *,
    evidence_score: float,
    evidence_full_trust: float = 30.0,
) -> Dict[str, float]:
    """
    Blend heuristic probs towards uniform when evidence is weak.

    evidence_score:
      A non-negative scalar built from feature counts (keywords/headings/citations etc).

    evidence_full_trust:
      Evidence value at which we trust heuristics fully (weight=1.0).

    Returns:
      A probability distribution (sums to 1).
    """
    # Map evidence into [0, 1] trust weight
    w = min(1.0, max(0.0, evidence_score / max(1.0, evidence_full_trust)))

    # Uniform distribution over current keys
    keys = list(probs.keys())
    uniform = 1.0 / max(1, len(keys))

    dampened = {k: (w * float(probs.get(k, 0.0)) + (1.0 - w) * uniform) for k in keys}

    s = sum(dampened.values()) or 1.0
    return {k: v / s for k, v in dampened.items()}


def _top2(probs: Dict[str, float]) -> tuple[tuple[str, float], tuple[str, float]]:
    """
    Return (top1, top2) as ((label, prob), (label, prob)).
    """
    items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    top1 = items[0]
    top2 = items[1] if len(items) > 1 else ("", 0.0)
    return top1, top2


def _maybe_abstain(
    label: str,
    confidence: float,
    *,
    evidence_score: float,
    min_conf: float = 0.45,
    min_evidence: float = 3.0,
) -> tuple[str, float]:
    # Abstain if both are weak: low confidence AND low evidence.
    if confidence < min_conf and evidence_score < min_evidence:
        return "other_or_unknown", confidence
    return label, confidence


def _maybe_mixed_label(
    probs: Dict[str, float],
    *,
    mixed_gap: float = 0.08,
) -> tuple[str, float, Dict[str, float]]:
    """
    If the top two classes are close, return a mixed label.
    mixed_gap: if (top1 - top2) <= mixed_gap, consider it mixed.

    Returns:
      (label, confidence, probs)
    """
    (l1, p1), (l2, p2) = _top2(probs)
    if l2 and (p1 - p2) <= mixed_gap:
        return f"mixed:{l1}+{l2}", p1, probs
    return l1, p1, probs


def classify_pdf(
    pdf_path: str,
    *,
    max_pages_text: Optional[int] = None,
    ocr_if_poor_text: bool = True,
    ocr_max_pages: int = 10,
    ocr_lang: str = "eng",
    llm_enabled: bool = True,
    llm_base_url: str = "http://localhost:8000/v1",
    llm_model: str = "qwen3-30b-a3b-awq",
    llm_api_key: str = "EMPTY",
    llm_ensemble_n: int = 5,
    fusion_alpha: float = 0.6,
    mixed_gap: float = 0.08,
    llm_override_threshold: float = 3.0,
) -> ClassificationResult:
    """
    End-to-end classification for a PDF.

    Returns:
        ClassificationResult with final probs + detailed reasons for auditability.
    """
    # 1) Extract text
    doc = extract_pdf_text(pdf_path, max_pages=max_pages_text)

    # 2) OCR fallback if needed
    quality = text_quality_ok(doc.text)
    if ocr_if_poor_text and not quality.ok:
        ocr_doc = ocr_pdf(pdf_path, max_pages=ocr_max_pages, lang=ocr_lang)
        # Keep the original total page count from pdf_text extraction
        doc = ExtractedDoc(
            text=ocr_doc.text,
            lines=ocr_doc.lines,
            pages=doc.pages,
            source="ocr",
            meta={**doc.meta, "ocr": ocr_doc.meta, "text_quality": quality.metrics, "text_quality_reason": quality.reason},
        )
    else:
        # Attach quality metrics anyway (useful for debugging)
        doc.meta["text_quality"] = quality.metrics
        doc.meta["text_quality_reason"] = quality.reason

    # 3) Features
    sections = count_sections(doc.lines)
    cites = detect_citations(doc.text, has_references_heading=sections.present.get("references", False))
    kw = count_keywords(doc.text)

    # 4) Rubrics
    r_imrad = score_imrad(sections)
    r_cites = score_citations(cites, text_len=len(doc.text))
    r_deliv = score_deliverable(kw, sections=sections)
    r_ped = score_pedagogy(kw)
    r_proc = score_procedure(kw)

    # Simple policy hint from keyword bucket (until you add a proper policy rubric)
    policy_hint = bucket_score(kw.bucket_hits.get("policy_guidance", 0), saturation=25)

    heur_probs = _heuristic_probs_from_rubrics(
        imrad=r_imrad.score,
        cites=r_cites.score,
        deliverable=r_deliv.score,
        pedagogy=r_ped.score,
        procedure=r_proc.score,
        policy_hint=policy_hint,
    )

    # --- Evidence-strength dampening (prevents overconfidence on thin signals) ---
    # Evidence components:
    # - keyword hits across all buckets (raw counts)
    # - section heading matches (sum of all detected headings)
    # - citation markers (numeric + author-year + DOI), plus small weight for URLs
    text_len = max(1, len(doc.text))
    per_10k = text_len / 10000.0

    keyword_evidence_raw = sum(kw.bucket_hits.values())
    section_evidence_raw = sum(sections.counts.values())
    citation_evidence_raw = (
            cites.cite_numeric
            + cites.cite_authoryear
            + (2 * cites.doi_mentions)
            + int(0.5 * cites.url_mentions)
    )

    # Normalise counts by length (per 10k chars). Sections are sparse, so don’t over-normalise them.
    keyword_evidence = keyword_evidence_raw / per_10k
    citation_evidence = citation_evidence_raw / per_10k
    section_evidence = section_evidence_raw  # keep as-is (headings are countable and not length-proportional)

    evidence_score = (1.0 * keyword_evidence) + (2.0 * section_evidence) + (1.5 * citation_evidence)

    heur_probs = _dampen_probs_by_evidence(
        heur_probs,
        evidence_score=evidence_score,
        evidence_full_trust=20.0,  # lower now that evidence is length-normalised
    )

    # 5) LLM
    llm_res = None
    final = None
    reasons_llm_error: Optional[str] = None

    if llm_enabled:
        try:
            llm_res = llm_ensemble(
                doc.text,
                n=llm_ensemble_n,
                base_url=llm_base_url,
                api_key=llm_api_key,
                model=llm_model,
            )
        except Exception as e:
            # Fail soft: keep heuristics rather than crashing the whole pipeline
            llm_res = None
            reasons_llm_error = str(e)

        # Decide using LLM only if we actually have an LLM result
        if llm_res is not None:
            # --- LLM-only override when heuristics have almost no evidence ---
            if evidence_score < llm_override_threshold:
                final_probs = {k: round(v, 4) for k, v in llm_res.probs.items()}
                label, confidence, _ = _maybe_mixed_label(llm_res.probs, mixed_gap=mixed_gap)
                confidence = round(confidence, 3)
                final = None  # no fusion object in this path
            else:
                final = fuse_probs(heur_probs, llm_res.probs, alpha=fusion_alpha)
                final_probs = final.probs
                label, confidence, _ = _maybe_mixed_label(
                    {k: float(v) for k, v in final_probs.items()},
                    mixed_gap=mixed_gap,
                )
                confidence = round(confidence, 3)
        else:
            # LLM failed -> heuristic fallback
            final_probs = {k: round(v, 4) for k, v in heur_probs.items()}
            label, confidence, _ = _maybe_mixed_label(heur_probs, mixed_gap=mixed_gap)
            confidence = round(confidence, 3)
            final = None
    else:
        # Pure heuristic output
        final_probs = {k: round(v, 4) for k, v in heur_probs.items()}
        label, confidence, _ = _maybe_mixed_label(heur_probs, mixed_gap=mixed_gap)
        confidence = round(confidence, 3)
        final = None

    # Apply abstain/unknown rule (after final label/conf computed)
    label, confidence = _maybe_abstain(
        label,
        confidence,
        evidence_score=evidence_score,
    )

    # 6) Subcategory Classification (evidence-based from data_model)
    # Map main label to parent type for filtering
    parent_type_map = {
        "scientific_research": ParentType.SCIENTIFIC_RESEARCH,
        "deliverable_report": ParentType.DELIVERABLE_REPORT,
        "educational": ParentType.EDUCATIONAL,
        "practice_oriented": ParentType.PRACTICE_ORIENTED,
        "policy_guidance": ParentType.POLICY_GUIDANCE,
        "other_or_unknown": ParentType.OTHER,
    }
    parent_type_filter = parent_type_map.get(label.split(":")[0])  # Handle mixed labels
    
    # Build rubric scores dict for context
    rubric_scores = {
        "imrad": r_imrad.score,
        "citations": r_cites.score,
        "deliverable": r_deliv.score,
        "pedagogy": r_ped.score,
        "procedure": r_proc.score,
    }
    
    # Run subcategory classification
    subcat_best, subcat_all, subcat_report = score_subcategories(
        text=doc.text,
        lines=doc.lines,
        page_count=doc.pages,
        sections=sections,
        rubric_scores=rubric_scores,
        parent_type_filter=parent_type_filter,
    )

    # 7) Reasons payload (audit trail)
    reasons: Dict[str, Any] = {
        "doc": {
            "pdf_path": pdf_path,
            "pages": doc.pages,
            "source": doc.source,
            "meta": doc.meta,
        },
        "features": {
            "sections": {
                "counts": sections.counts,
                "present": sections.present,
                "matched_lines": sections.matched_lines,
            },
            "citations": cites.__dict__,
            "keywords": {
                "bucket_hits": kw.bucket_hits,
                "term_hits": kw.term_hits,
            },
        },
        "rubrics": {
            "imrad": r_imrad.__dict__,
            "citations": r_cites.__dict__,
            "deliverable": r_deliv.__dict__,
            "pedagogy": r_ped.__dict__,
            "procedure": r_proc.__dict__,
            "policy_hint": policy_hint,
        },
        "heuristic_probs": {k: round(v, 4) for k, v in heur_probs.items()},
        "evidence_score": evidence_score,
        "decision": {
            "mixed_gap": mixed_gap,
            "llm_override_threshold": llm_override_threshold,
            "used_llm_override": bool(llm_res is not None and (evidence_score < llm_override_threshold)),
            "fusion_alpha": fusion_alpha,
            "evidence_components": {
                "keyword_evidence_raw": keyword_evidence_raw,
                "section_evidence_raw": section_evidence_raw,
                "citation_evidence_raw": citation_evidence_raw,
                "keyword_evidence_norm": keyword_evidence,
                "citation_evidence_norm": citation_evidence,
            },
        },
        "final_decision": {"label": label, "confidence": confidence},
    }

    if llm_res is not None:
        reasons["llm"] = {
            "label": llm_res.label,
            "probs": {k: round(v, 4) for k, v in llm_res.probs.items()},
            "rationale": llm_res.rationale,
            "raw_json": llm_res.raw_json,  # keep for debugging; remove if too big
        }

    if reasons_llm_error is not None:
        reasons["llm_error"] = reasons_llm_error

    if final is not None:
        reasons["fusion"] = final.__dict__
    
    # Add subcategory information to reasons
    reasons["subcategory_classification"] = {
        "best_match": subcat_best.to_dict() if subcat_best else None,
        "all_scores": [s.to_dict() for s in subcat_all[:5]],  # Top 5
        "reproducibility": subcat_report.get("reproducibility"),
    }

    return ClassificationResult(
        label=label,
        confidence=confidence,
        probs=final_probs,
        reasons=reasons,
        subcategory=subcat_best,
        subcategory_scores=subcat_all,
    )
