# docint/rubrics/subcategories.py
"""
Subcategory classification based on data_model.subcategories_document_consolidated.json

Each subcategory has:
- detectable_features: measurable signals we can extract from PDF text
- parent_type: the main document type (scientific_research, deliverable_report, etc.)
- confidence_weights: how much each feature contributes to confidence
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from enum import Enum


class ParentType(str, Enum):
    SCIENTIFIC_RESEARCH = "scientific_research"
    DELIVERABLE_REPORT = "deliverable_report"
    EDUCATIONAL = "educational"
    PRACTICE_ORIENTED = "practice_oriented"
    POLICY_GUIDANCE = "policy_guidance"
    OTHER = "other_or_unknown"


@dataclass
class FeatureDefinition:
    """Definition of a detectable feature with scoring logic."""
    name: str
    description: str
    weight: float  # Contribution to confidence (0-1)
    # Function that extracts the feature value from context dict
    extractor_key: str  # Key to look up in extraction context


@dataclass
class SubcategoryDefinition:
    """Complete definition of a document subcategory."""
    id: str
    name: str
    description: str
    parent_type: ParentType
    detectable_features: List[FeatureDefinition]
    # Minimum features required for positive identification
    min_features_required: int = 2
    # Whether this subcategory can be auto-detected (some are too ambiguous)
    auto_detectable: bool = True
    # Rationale template for why this classification was chosen
    rationale_template: str = ""


# =============================================================================
# FEATURE DEFINITIONS (Measurable, Reproducible Signals)
# =============================================================================

FEATURES = {
    # Academic/Scientific features
    "imrad_structure": FeatureDefinition(
        name="imrad_structure",
        description="Presence of IMRaD sections (Introduction, Methods, Results, Discussion)",
        weight=0.25,
        extractor_key="imrad_score"
    ),
    "peer_review_markers": FeatureDefinition(
        name="peer_review_markers",
        description="Keywords like 'peer-reviewed', 'accepted for publication', journal names",
        weight=0.15,
        extractor_key="peer_review_signals"
    ),
    "citation_density": FeatureDefinition(
        name="citation_density",
        description="Academic citation patterns (numeric, author-year, DOI density)",
        weight=0.20,
        extractor_key="citation_score"
    ),
    "abstract_quality": FeatureDefinition(
        name="abstract_quality",
        description="Structured abstract with background, methods, results, conclusions",
        weight=0.15,
        extractor_key="abstract_features"
    ),
    "conference_markers": FeatureDefinition(
        name="conference_markers",
        description="Conference name, proceedings, presentation date, venue",
        weight=0.20,
        extractor_key="conference_signals"
    ),
    "thesis_markers": FeatureDefinition(
        name="thesis_markers",
        description="Thesis/dissertation keywords, university name, degree, supervisor",
        weight=0.25,
        extractor_key="thesis_signals"
    ),
    "book_features": FeatureDefinition(
        name="book_features",
        description="ISBN, publisher, chapter structure, table of contents",
        weight=0.20,
        extractor_key="book_signals"
    ),
    
    # Technical/Report features
    "deliverable_markers": FeatureDefinition(
        name="deliverable_markers",
        description="Work package, task number, milestone, grant agreement, Horizon Europe",
        weight=0.25,
        extractor_key="deliverable_signals"
    ),
    "version_control": FeatureDefinition(
        name="version_control",
        description="Version history, revision table, change log",
        weight=0.15,
        extractor_key="version_signals"
    ),
    "technical_specs": FeatureDefinition(
        name="technical_specs",
        description="Specifications, requirements, technical parameters",
        weight=0.20,
        extractor_key="technical_signals"
    ),
    "formal_structure": FeatureDefinition(
        name="formal_structure",
        description="Executive summary, appendices, structured sections",
        weight=0.15,
        extractor_key="formal_structure_score"
    ),
    
    # Educational features
    "learning_objectives": FeatureDefinition(
        name="learning_objectives",
        description="Learning goals, 'by the end of', 'you will be able to'",
        weight=0.25,
        extractor_key="pedagogy_score"
    ),
    "exercises_assessments": FeatureDefinition(
        name="exercises_assessments",
        description="Exercises, quizzes, assignments, self-check questions",
        weight=0.20,
        extractor_key="exercise_signals"
    ),
    "tutorial_structure": FeatureDefinition(
        name="tutorial_structure",
        description="Step-by-step instructions, examples, practice tasks",
        weight=0.20,
        extractor_key="tutorial_signals"
    ),
    
    # Practice/Guide features
    "procedure_steps": FeatureDefinition(
        name="procedure_steps",
        description="Numbered steps, instructions, 'how to' patterns",
        weight=0.20,
        extractor_key="procedure_score"
    ),
    "materials_tools": FeatureDefinition(
        name="materials_tools",
        description="Materials list, tools required, equipment specifications",
        weight=0.15,
        extractor_key="materials_signals"
    ),
    "safety_warnings": FeatureDefinition(
        name="safety_warnings",
        description="Safety notes, warnings, precautions, risk statements",
        weight=0.15,
        extractor_key="safety_signals"
    ),
    "checklists": FeatureDefinition(
        name="checklists",
        description="Checklist format, verification steps, quality control",
        weight=0.15,
        extractor_key="checklist_signals"
    ),
    
    # Communication features
    "news_timeliness": FeatureDefinition(
        name="news_timeliness",
        description="Date references, current events, time-sensitive language",
        weight=0.20,
        extractor_key="news_signals"
    ),
    "press_release_format": FeatureDefinition(
        name="press_release_format",
        description="Press release structure, contact info, boilerplate",
        weight=0.20,
        extractor_key="press_release_signals"
    ),
    "short_form": FeatureDefinition(
        name="short_form",
        description="1-8 pages, concise format, summary nature",
        weight=0.15,
        extractor_key="page_count_features"
    ),
    "promotional_content": FeatureDefinition(
        name="promotional_content",
        description="Promotional language, calls to action, benefits focus",
        weight=0.15,
        extractor_key="promotional_signals"
    ),
    
    # Presentation features
    "slide_indicators": FeatureDefinition(
        name="slide_indicators",
        description="Slide numbers, presentation software markers, 'slide' references",
        weight=0.25,
        extractor_key="slide_signals"
    ),
    "visual_heavy": FeatureDefinition(
        name="visual_heavy",
        description="High image-to-text ratio, minimal text per page",
        weight=0.15,
        extractor_key="visual_features"
    ),
    
    # Policy features
    "compliance_language": FeatureDefinition(
        name="compliance_language",
        description="Shall, must, should, compliance, regulation, directive",
        weight=0.25,
        extractor_key="policy_signals"
    ),
    "governance_references": FeatureDefinition(
        name="governance_references",
        description="Policy references, governance structures, legal framework",
        weight=0.20,
        extractor_key="governance_signals"
    ),
}


# =============================================================================
# SUBCATEGORY DEFINITIONS
# =============================================================================

SUBCATEGORIES: Dict[str, SubcategoryDefinition] = {
    # --- SCIENTIFIC RESEARCH SUBCATEGORIES ---
    "journal_article": SubcategoryDefinition(
        id="Zyvdw7E2",
        name="Journal article",
        description="Peer-reviewed academic article published in a scholarly journal",
        parent_type=ParentType.SCIENTIFIC_RESEARCH,
        detectable_features=[
            FEATURES["imrad_structure"],
            FEATURES["peer_review_markers"],
            FEATURES["citation_density"],
            FEATURES["abstract_quality"],
        ],
        min_features_required=3,
        rationale_template="Identified as journal article due to {features_found} (confidence: {confidence:.2f})",
    ),
    
    "conference_proceedings": SubcategoryDefinition(
        id="arQwir9z",
        name="Article in conference proceedings",
        description="Academic paper presented at a conference and published in proceedings",
        parent_type=ParentType.SCIENTIFIC_RESEARCH,
        detectable_features=[
            FEATURES["conference_markers"],
            FEATURES["imrad_structure"],
            FEATURES["citation_density"],
            FEATURES["peer_review_markers"],
        ],
        min_features_required=2,
        rationale_template="Conference paper detected based on {features_found} (confidence: {confidence:.2f})",
    ),
    
    "book_chapter": SubcategoryDefinition(
        id="P3nzEsdB",
        name="Chapter in edited volume",
        description="Individual chapter contributed to an edited book or collection",
        parent_type=ParentType.SCIENTIFIC_RESEARCH,
        detectable_features=[
            FEATURES["book_features"],
            FEATURES["citation_density"],
            FEATURES["imrad_structure"],
        ],
        min_features_required=2,
        rationale_template="Book chapter identified through {features_found} (confidence: {confidence:.2f})",
    ),
    
    "thesis": SubcategoryDefinition(
        id="k6VvsRTc",
        name="Thesis",
        description="Academic dissertation or thesis submitted for a degree",
        parent_type=ParentType.SCIENTIFIC_RESEARCH,
        detectable_features=[
            FEATURES["thesis_markers"],
            FEATURES["imrad_structure"],
            FEATURES["citation_density"],
            FEATURES["formal_structure"],
        ],
        min_features_required=2,
        rationale_template="Thesis/Dissertation confirmed via {features_found} (confidence: {confidence:.2f})",
    ),
    
    "book": SubcategoryDefinition(
        id="NBq4fMG2",
        name="Book",
        description="Full-length published book or monograph",
        parent_type=ParentType.SCIENTIFIC_RESEARCH,
        detectable_features=[
            FEATURES["book_features"],
            FEATURES["formal_structure"],
            FEATURES["citation_density"],
        ],
        min_features_required=2,
        rationale_template="Book/monograph detected based on {features_found} (confidence: {confidence:.2f})",
    ),
    
    # --- DELIVERABLE REPORT SUBCATEGORIES ---
    "technical_report": SubcategoryDefinition(
        id="CONSOLIDATED_TECH_REPORT",
        name="Technical Report",
        description="Formal technical documentation including project reports, reviews, specifications",
        parent_type=ParentType.DELIVERABLE_REPORT,
        detectable_features=[
            FEATURES["deliverable_markers"],
            FEATURES["version_control"],
            FEATURES["technical_specs"],
            FEATURES["formal_structure"],
        ],
        min_features_required=2,
        rationale_template="Technical report identified via {features_found} (confidence: {confidence:.2f})",
    ),
    
    # --- EDUCATIONAL SUBCATEGORIES ---
    "tutorial": SubcategoryDefinition(
        id="4NLQdUhM",
        name="Tutorial",
        description="Step-by-step instructional content for learning a specific skill",
        parent_type=ParentType.EDUCATIONAL,
        detectable_features=[
            FEATURES["tutorial_structure"],
            FEATURES["learning_objectives"],
            FEATURES["exercises_assessments"],
        ],
        min_features_required=2,
        rationale_template="Tutorial detected through {features_found} (confidence: {confidence:.2f})",
    ),
    
    # --- PRACTICE ORIENTED SUBCATEGORIES ---
    "guide_manual": SubcategoryDefinition(
        id="CONSOLIDATED_GUIDE_MANUAL",
        name="Guide/Manual",
        description="Comprehensive reference with instructions and detailed information",
        parent_type=ParentType.PRACTICE_ORIENTED,
        detectable_features=[
            FEATURES["procedure_steps"],
            FEATURES["materials_tools"],
            FEATURES["safety_warnings"],
            FEATURES["checklists"],
            FEATURES["formal_structure"],
        ],
        min_features_required=3,
        rationale_template="Guide/Manual confirmed via {features_found} (confidence: {confidence:.2f})",
    ),
    
    # --- PRESENTATION (can be educational or informative) ---
    "presentation": SubcategoryDefinition(
        id="CONSOLIDATED_PRESENTATION",
        name="Presentation",
        description="Slide-based presentations for any purpose",
        parent_type=ParentType.EDUCATIONAL,  # Default, can vary
        detectable_features=[
            FEATURES["slide_indicators"],
            FEATURES["visual_heavy"],
            FEATURES["short_form"],
        ],
        min_features_required=2,
        rationale_template="Presentation format detected via {features_found} (confidence: {confidence:.2f})",
    ),
    
    # --- NEWS & COMMUNICATION ---
    "news_communication": SubcategoryDefinition(
        id="CONSOLIDATED_NEWS_COMM",
        name="News & Communication",
        description="Newsletters, press releases, policy briefs - time-sensitive communications",
        parent_type=ParentType.POLICY_GUIDANCE,
        detectable_features=[
            FEATURES["news_timeliness"],
            FEATURES["press_release_format"],
            FEATURES["short_form"],
            FEATURES["promotional_content"],
        ],
        min_features_required=2,
        rationale_template="News/Communication document via {features_found} (confidence: {confidence:.2f})",
    ),
    
    # --- INFORMATIONAL BOOKLET ---
    "informational_booklet": SubcategoryDefinition(
        id="CONSOLIDATED_INFO_BOOKLET",
        name="Informational Booklet",
        description="Short printed materials (1-8 pages) with concise information",
        parent_type=ParentType.PRACTICE_ORIENTED,
        detectable_features=[
            FEATURES["short_form"],
            FEATURES["promotional_content"],
            FEATURES["visual_heavy"],
        ],
        min_features_required=2,
        rationale_template="Informational booklet detected via {features_found} (confidence: {confidence:.2f})",
    ),
}


def get_subcategories_by_parent(parent_type: ParentType) -> Dict[str, SubcategoryDefinition]:
    """Get all subcategories for a given parent type."""
    return {
        k: v for k, v in SUBCATEGORIES.items()
        if v.parent_type == parent_type
    }


def get_all_detectable_features() -> List[str]:
    """Get list of all unique feature names across all subcategories."""
    features = set()
    for subcat in SUBCATEGORIES.values():
        for feat in subcat.detectable_features:
            features.add(feat.name)
    return sorted(features)
