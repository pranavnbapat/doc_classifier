# docint/rubrics/subcategory_scorer.py
"""
Evidence-based subcategory classification with measurable, reproducible scoring.

This module provides:
1. Feature extraction context gathering
2. Subcategory scoring with confidence calculation
3. Rationale generation with specific evidence
4. Audit trail for all decisions
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

from docint.rubrics.subcategories import (
    SUBCATEGORIES, SubcategoryDefinition, FeatureDefinition,
    ParentType, get_subcategories_by_parent
)
from docint.features.sections import SectionSignals


@dataclass
class FeatureEvidence:
    """Evidence for a single feature detection."""
    feature_name: str
    detected: bool
    score: float  # 0.0 to 1.0
    raw_value: Any  # The actual extracted value
    excerpts: List[str] = field(default_factory=list)  # Text excerpts as proof
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "detected": self.detected,
            "score": round(self.score, 4),
            "raw_value": self.raw_value,
            "excerpts": self.excerpts[:3],  # Limit excerpts
        }


@dataclass
class SubcategoryScore:
    """Complete scoring result for a subcategory."""
    subcategory_id: str
    subcategory_name: str
    parent_type: str
    confidence: float  # 0.0 to 1.0
    evidence_score: float  # Raw evidence points
    max_possible_evidence: float
    features_found: List[str]  # Names of detected features
    feature_details: Dict[str, FeatureEvidence]
    rationale: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "subcategory_id": self.subcategory_id,
            "subcategory_name": self.subcategory_name,
            "parent_type": self.parent_type,
            "confidence": round(self.confidence, 4),
            "evidence_score": round(self.evidence_score, 4),
            "max_possible_evidence": round(self.max_possible_evidence, 4),
            "features_found": self.features_found,
            "feature_details": {k: v.to_dict() for k, v in self.feature_details.items()},
            "rationale": self.rationale,
        }


@dataclass
class ExtractionContext:
    """Container for all extracted features from a document."""
    # Core text info
    text: str
    text_lower: str
    lines: List[str]
    page_count: int
    
    # Section analysis
    sections: Optional[SectionSignals] = None
    
    # Rubric scores (from existing pipeline)
    imrad_score: float = 0.0
    citation_score: float = 0.0
    deliverable_score: float = 0.0
    pedagogy_score: float = 0.0
    procedure_score: float = 0.0
    
    # Extracted signals (populated by feature extractors)
    signals: Dict[str, Any] = field(default_factory=dict)
    excerpts: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    
    def get_signal(self, key: str, default: Any = None) -> Any:
        return self.signals.get(key, default)


# =============================================================================
# FEATURE EXTRACTORS (Measurable, Reproducible)
# =============================================================================

class FeatureExtractors:
    """Collection of feature extraction methods. Each returns (score, raw_value, excerpts)."""
    
    @staticmethod
    def extract_imrad_score(ctx: ExtractionContext) -> Tuple[float, Any, List[str]]:
        """Extract IMRaD structure score."""
        score = ctx.imrad_score
        excerpts = []
        if ctx.sections:
            for section_name, lines in ctx.sections.matched_lines.items():
                if lines:
                    excerpts.extend(lines[:2])
        return (score, score, excerpts)
    
    @staticmethod
    def extract_peer_review_signals(ctx: ExtractionContext) -> Tuple[float, Any, List[str]]:
        """Extract peer review and journal markers."""
        patterns = [
            r'\bpeer[- ]?review(ed)?\b',
            r'\baccepted for publication\b',
            r'\bjournal of\s+\w+',
            r'\bpublished in\s+\w+',
            r'\bdoi:\s*10\.\d+',
            r'\bcorresponding author\b',
            r'\breceived:\s*\d{1,2}',
            r'\brevised:\s*\d{1,2}',
        ]
        score = 0.0
        matches = []
        for pat in patterns:
            found = re.findall(pat, ctx.text_lower, re.I)
            matches.extend(found)
            score += len(found) * 0.15
        score = min(1.0, score)
        excerpts = matches[:3] if matches and isinstance(matches[0], str) else []
        return (score, len(matches), excerpts)
    
    @staticmethod
    def extract_citation_score(ctx: ExtractionContext) -> Tuple[float, Any, List[str]]:
        """Extract citation density score."""
        score = ctx.citation_score
        # Also look for reference section
        has_references = False
        if ctx.sections:
            has_references = ctx.sections.present.get('references', False)
        raw_value = {"score": score, "has_references_section": has_references}
        return (score, raw_value, [])
    
    @staticmethod
    def extract_abstract_features(ctx: ExtractionContext) -> Tuple[float, Any, List[str]]:
        """Extract structured abstract features."""
        if not ctx.sections:
            return (0.0, False, [])
        
        has_abstract = ctx.sections.present.get('abstract', False)
        score = 0.3 if has_abstract else 0.0
        
        # Check for structured abstract markers
        abstract_patterns = [
            r'\bbackground\s*:',
            r'\bobjective\s*:',
            r'\bmethods?\s*:',
            r'\bresults?\s*:',
            r'\bconclusions?\s*:',
        ]
        matches = 0
        for pat in abstract_patterns:
            if re.search(pat, ctx.text_lower):
                matches += 1
        score += min(0.7, matches * 0.15)
        
        return (score, {"has_abstract": has_abstract, "structured_markers": matches}, [])
    
    @staticmethod
    def extract_conference_signals(ctx: ExtractionContext) -> Tuple[float, Any, List[str]]:
        """Extract conference/proceedings markers."""
        patterns = [
            r'\bconference\b',
            r'\bproceedings\b',
            r'\bpresented at\b',
            r'\bworkshop\b',
            r'\bsymposium\b',
            r'\bannual meeting\b',
            r'\bcongress\b',
        ]
        score = 0.0
        matches = []
        for pat in patterns:
            found = re.findall(pat, ctx.text_lower, re.I)
            if found:
                matches.extend(found[:2])
                score += len(found) * 0.15
        score = min(1.0, score + 0.2)  # Base boost for any match
        return (score, len(matches), matches[:3])
    
    @staticmethod
    def extract_thesis_signals(ctx: ExtractionContext) -> Tuple[float, Any, List[str]]:
        """Extract thesis/dissertation markers."""
        patterns = [
            r'\b(thesis|dissertation)\b',
            r'\bsubmitted in partial fulfilment\b',
            r'\bsubmitted in partial fulfillment\b',
            r'\bdoctor of philosophy\b',
            r'\bph\.?d\b',
            r'\bmasters? degree\b',
            r'\bsupervisor\s*:',
            r'\bdoctoral\b',
            r'\buniversity of\s+\w+',
        ]
        score = 0.0
        matches = []
        for pat in patterns:
            found = re.findall(pat, ctx.text_lower, re.I)
            if found:
                matches.extend(found[:2])
                score += len(found) * 0.12
        score = min(1.0, score + 0.1)
        return (score, len(matches), matches[:3])
    
    @staticmethod
    def extract_book_signals(ctx: ExtractionContext) -> Tuple[float, Any, List[str]]:
        """Extract book/monograph markers."""
        patterns = [
            r'\bisbn\s*:?\s*\d',
            r'\bchapter\s+\d+\b',
            r'\bedited by\b',
            r'\bpublisher\s*:',
            r'\bfirst published\b',
            r'\bcopyright\s+©?\s*\d{4}',
        ]
        score = 0.0
        matches = []
        for pat in patterns:
            found = re.findall(pat, ctx.text_lower, re.I)
            if found:
                matches.extend(found[:2])
                score += len(found) * 0.15
        # Boost for page count (books are longer)
        if ctx.page_count > 50:
            score += 0.2
        score = min(1.0, score)
        return (score, {"matches": len(matches), "pages": ctx.page_count}, matches[:3])
    
    @staticmethod
    def extract_deliverable_signals(ctx: ExtractionContext) -> Tuple[float, Any, List[str]]:
        """Extract EU project deliverable markers."""
        patterns = [
            r'\bdeliverable\b',
            r'\bwork package\b',
            r'\btask\s+\d+\.?\d*\b',
            r'\bmilestone\b',
            r'\bgrant agreement\b',
            r'\bga number\b',
            r'\bhorizon europe\b',
            r'\bh2020\b',
            r'\bdissemination level\b',
            r'\bproject acronym\b',
            r'\bwp\d+\b',
        ]
        score = ctx.deliverable_score * 0.5  # Use existing score as base
        matches = []
        for pat in patterns:
            found = re.findall(pat, ctx.text_lower, re.I)
            if found:
                matches.extend(found[:2])
                score += len(found) * 0.08
        score = min(1.0, score)
        return (score, len(matches), matches[:3])
    
    @staticmethod
    def extract_version_signals(ctx: ExtractionContext) -> Tuple[float, Any, List[str]]:
        """Extract version control markers."""
        patterns = [
            r'\brevision history\b',
            r'\bversion\s*:?\s*\d+\.\d+',
            r'\bchange log\b',
            r'\bchangelog\b',
            r'\bdate\s+version\s+author',
            r'\bstatus\s*:?\s*(draft|final|approved)',
        ]
        score = 0.0
        matches = []
        for pat in patterns:
            found = re.findall(pat, ctx.text_lower, re.I)
            if found:
                matches.extend(found[:2])
                score += len(found) * 0.2
        score = min(1.0, score)
        return (score, len(matches), matches[:3])
    
    @staticmethod
    def extract_technical_signals(ctx: ExtractionContext) -> Tuple[float, Any, List[str]]:
        """Extract technical specification markers."""
        patterns = [
            r'\bspecifications?\b',
            r'\brequirements?\b',
            r'\btechnical parameters?\b',
            r'\bperformance criteria\b',
            r'\bfunctional requirements?\b',
        ]
        score = 0.0
        matches = []
        for pat in patterns:
            found = re.findall(pat, ctx.text_lower, re.I)
            if found:
                matches.extend(found[:2])
                score += len(found) * 0.15
        score = min(1.0, score)
        return (score, len(matches), matches[:3])
    
    @staticmethod
    def extract_formal_structure_score(ctx: ExtractionContext) -> Tuple[float, Any, List[str]]:
        """Extract formal document structure markers."""
        score = 0.0
        excerpts = []
        if ctx.sections:
            has_exec_summary = ctx.sections.present.get('executive_summary', False)
            has_appendix = ctx.sections.present.get('appendix', False)
            has_refs = ctx.sections.present.get('references', False)
            
            if has_exec_summary:
                score += 0.3
                excerpts.append("Executive Summary detected")
            if has_appendix:
                score += 0.2
                excerpts.append("Appendix detected")
            if has_refs:
                score += 0.2
            
            section_count = sum(1 for v in ctx.sections.present.values() if v)
            score += min(0.3, section_count * 0.05)
        
        return (score, score, excerpts)
    
    @staticmethod
    def extract_pedagogy_score(ctx: ExtractionContext) -> Tuple[float, Any, List[str]]:
        """Extract educational/pedagogy markers."""
        score = ctx.pedagogy_score * 0.6
        patterns = [
            r'\blearning objectives?\b',
            r'\bby the end of\b',
            r'\byou will be able to\b',
            r'\bprerequisites\b',
            r'\blearning outcomes?\b',
        ]
        matches = []
        for pat in patterns:
            found = re.findall(pat, ctx.text_lower, re.I)
            if found:
                matches.extend(found[:2])
                score += len(found) * 0.1
        score = min(1.0, score)
        return (score, len(matches), matches[:3])
    
    @staticmethod
    def extract_exercise_signals(ctx: ExtractionContext) -> Tuple[float, Any, List[str]]:
        """Extract exercise and assessment markers."""
        patterns = [
            r'\bexercise\s*\d*\b',
            r'\bquiz\b',
            r'\bassignment\b',
            r'\bself[- ]?check\b',
            r'\btest your knowledge\b',
            r'\breview questions?\b',
        ]
        score = 0.0
        matches = []
        for pat in patterns:
            found = re.findall(pat, ctx.text_lower, re.I)
            if found:
                matches.extend(found[:2])
                score += len(found) * 0.15
        score = min(1.0, score)
        return (score, len(matches), matches[:3])
    
    @staticmethod
    def extract_tutorial_signals(ctx: ExtractionContext) -> Tuple[float, Any, List[str]]:
        """Extract tutorial/step-by-step markers."""
        patterns = [
            r'\bstep\s*\d+\b',
            r'\bstep [-–] by [-–] step\b',
            r'\bfollow these steps\b',
            r'\bexample\s*\d*\b',
            r'\bpractice task\b',
            r'\btry it yourself\b',
        ]
        score = 0.0
        matches = []
        for pat in patterns:
            found = re.findall(pat, ctx.text_lower, re.I)
            if found:
                matches.extend(found[:2])
                score += len(found) * 0.15
        score = min(1.0, score)
        return (score, len(matches), matches[:3])
    
    @staticmethod
    def extract_procedure_score(ctx: ExtractionContext) -> Tuple[float, Any, List[str]]:
        """Extract procedure/step markers."""
        score = ctx.procedure_score * 0.6
        patterns = [
            r'\binstructions?\b',
            r'\bhow to\b',
            r'\bprocedure\b',
            r'^\s*\d+\.\s+\w+',  # Numbered steps
        ]
        matches = []
        for pat in patterns:
            found = re.findall(pat, ctx.text_lower, re.I)
            if found:
                matches.extend(found[:2])
                score += len(found) * 0.08
        score = min(1.0, score)
        return (score, len(matches), matches[:3])
    
    @staticmethod
    def extract_materials_signals(ctx: ExtractionContext) -> Tuple[float, Any, List[str]]:
        """Extract materials/tools list markers."""
        patterns = [
            r'\bmaterials needed\b',
            r'\btools required\b',
            r'\bequipment\b',
            r'\byou will need\b',
            r'\bmaterials list\b',
        ]
        score = 0.0
        matches = []
        for pat in patterns:
            found = re.findall(pat, ctx.text_lower, re.I)
            if found:
                matches.extend(found[:2])
                score += len(found) * 0.25
        score = min(1.0, score)
        return (score, len(matches), matches[:3])
    
    @staticmethod
    def extract_safety_signals(ctx: ExtractionContext) -> Tuple[float, Any, List[str]]:
        """Extract safety/warning markers."""
        patterns = [
            r'\bwarning\b',
            r'\bcaution\b',
            r'\bdanger\b',
            r'\bsafety\b',
            r'\bdo not\b',
            r'\bnever\b',
            r'\bimportant:\s*',
            r'\bnote:\s*',
        ]
        score = 0.0
        matches = []
        for pat in patterns:
            found = re.findall(pat, ctx.text_lower, re.I)
            if found:
                matches.extend(found[:2])
                score += len(found) * 0.08
        score = min(1.0, score)
        return (score, len(matches), matches[:3])
    
    @staticmethod
    def extract_checklist_signals(ctx: ExtractionContext) -> Tuple[float, Any, List[str]]:
        """Extract checklist format markers."""
        patterns = [
            r'[☐☑✓✔✗✘]',
            r'\[\s*[xX\u2713\u2714]?\s*\]',  # Checkbox [ ], [x], [✓]
            r'\bchecklist\b',
            r'\btick box\b',
        ]
        score = 0.0
        matches = []
        for pat in patterns:
            found = re.findall(pat, ctx.text_lower)
            if found:
                matches.extend(found[:2])
                score += len(found) * 0.15
        score = min(1.0, score)
        return (score, len(matches), matches[:3])
    
    @staticmethod
    def extract_news_signals(ctx: ExtractionContext) -> Tuple[float, Any, List[str]]:
        """Extract news/timeliness markers."""
        patterns = [
            r'\bpress release\b',
            r'\bfor immediate release\b',
            r'\bembargo\b',
            r'\bnewsletter\b',
            r'\bissue\s*\d+\b',
            r'\bthis month\b',
            r'\bthis week\b',
        ]
        score = 0.0
        matches = []
        for pat in patterns:
            found = re.findall(pat, ctx.text_lower, re.I)
            if found:
                matches.extend(found[:2])
                score += len(found) * 0.2
        score = min(1.0, score)
        return (score, len(matches), matches[:3])
    
    @staticmethod
    def extract_press_release_signals(ctx: ExtractionContext) -> Tuple[float, Any, List[str]]:
        """Extract press release format markers."""
        patterns = [
            r'\bfor more information\b',
            r'\bcontact\s*:',
            r'\bmedia contact\b',
            r'\babout\s+\w+\b',
            r'\b###\s*$',  # End marker
        ]
        score = 0.0
        matches = []
        for pat in patterns:
            found = re.findall(pat, ctx.text_lower, re.I)
            if found:
                matches.extend(found[:2])
                score += len(found) * 0.2
        score = min(1.0, score)
        return (score, len(matches), matches[:3])
    
    @staticmethod
    def extract_page_count_features(ctx: ExtractionContext) -> Tuple[float, Any, List[str]]:
        """Extract page count features."""
        pages = ctx.page_count
        # Short form: 1-8 pages
        if pages <= 8:
            score = 1.0
        elif pages <= 15:
            score = 0.5
        else:
            score = 0.0
        return (score, {"pages": pages}, [f"Document has {pages} pages"])
    
    @staticmethod
    def extract_promotional_signals(ctx: ExtractionContext) -> Tuple[float, Any, List[str]]:
        """Extract promotional content markers."""
        patterns = [
            r'\blearn more\b',
            r'\bcontact us\b',
            r'\bvisit\s+\w+\b',
            r'\bfree\b',
            r'\boffer\b',
            r'\bdiscover\b',
            r'\bdownload\b',
        ]
        score = 0.0
        matches = []
        for pat in patterns:
            found = re.findall(pat, ctx.text_lower, re.I)
            if found:
                matches.extend(found[:2])
                score += len(found) * 0.1
        score = min(1.0, score)
        return (score, len(matches), matches[:3])
    
    @staticmethod
    def extract_slide_signals(ctx: ExtractionContext) -> Tuple[float, Any, List[str]]:
        """Extract presentation/slide markers."""
        patterns = [
            r'\bslide\s*\d+\b',
            r'\bpage\s*\d+\s+of\s+\d+\b',
            r'\bpowerpoint\b',
            r'\bpresentation\b',
            r'\bclick to\b',
        ]
        score = 0.0
        matches = []
        for pat in patterns:
            found = re.findall(pat, ctx.text_lower, re.I)
            if found:
                matches.extend(found[:2])
                score += len(found) * 0.2
        # Check for high image-to-text ratio indicator (short lines)
        short_lines = sum(1 for ln in ctx.lines if len(ln) < 50 and len(ln) > 5)
        if len(ctx.lines) > 0 and short_lines / len(ctx.lines) > 0.5:
            score += 0.2
        score = min(1.0, score)
        return (score, len(matches), matches[:3])
    
    @staticmethod
    def extract_visual_features(ctx: ExtractionContext) -> Tuple[float, Any, List[str]]:
        """Extract visual-heavy document markers."""
        # Estimate based on short lines (often indicate bullet points/captions)
        if not ctx.lines:
            return (0.0, 0, [])
        short_lines = sum(1 for ln in ctx.lines if len(ln) < 40)
        ratio = short_lines / len(ctx.lines)
        score = min(1.0, ratio * 1.5)  # Scale up
        return (score, {"short_line_ratio": ratio}, [f"{ratio:.1%} short lines (visual indicator)"])
    
    @staticmethod
    def extract_policy_signals(ctx: ExtractionContext) -> Tuple[float, Any, List[str]]:
        """Extract policy/compliance language markers."""
        patterns = [
            r'\bshall\b',
            r'\bmust\b',
            r'\bshould\b',
            r'\bcompliance\b',
            r'\bregulation\b',
            r'\bdirective\b',
            r'\bguidelines\b',
            r'\bmandatory\b',
            r'\brecommended\b',
        ]
        score = 0.0
        matches = []
        for pat in patterns:
            found = re.findall(pat, ctx.text_lower, re.I)
            if found:
                matches.extend(found[:2])
                score += len(found) * 0.08
        score = min(1.0, score)
        return (score, len(matches), matches[:3])
    
    @staticmethod
    def extract_governance_signals(ctx: ExtractionContext) -> Tuple[float, Any, List[str]]:
        """Extract governance/policy framework markers."""
        patterns = [
            r'\bpolicy\b',
            r'\bgovernance\b',
            r'\blegal framework\b',
            r'\bregulatory\b',
            r'\bstandard\s*:',
            r'\biso\s*\d+',
        ]
        score = 0.0
        matches = []
        for pat in patterns:
            found = re.findall(pat, ctx.text_lower, re.I)
            if found:
                matches.extend(found[:2])
                score += len(found) * 0.15
        score = min(1.0, score)
        return (score, len(matches), matches[:3])


# Mapping from feature name to extractor method
EXTRACTOR_MAP: Dict[str, Callable] = {
    "imrad_score": FeatureExtractors.extract_imrad_score,
    "peer_review_signals": FeatureExtractors.extract_peer_review_signals,
    "citation_score": FeatureExtractors.extract_citation_score,
    "abstract_features": FeatureExtractors.extract_abstract_features,
    "conference_signals": FeatureExtractors.extract_conference_signals,
    "thesis_signals": FeatureExtractors.extract_thesis_signals,
    "book_signals": FeatureExtractors.extract_book_signals,
    "deliverable_signals": FeatureExtractors.extract_deliverable_signals,
    "version_signals": FeatureExtractors.extract_version_signals,
    "technical_signals": FeatureExtractors.extract_technical_signals,
    "formal_structure_score": FeatureExtractors.extract_formal_structure_score,
    "pedagogy_score": FeatureExtractors.extract_pedagogy_score,
    "exercise_signals": FeatureExtractors.extract_exercise_signals,
    "tutorial_signals": FeatureExtractors.extract_tutorial_signals,
    "procedure_score": FeatureExtractors.extract_procedure_score,
    "materials_signals": FeatureExtractors.extract_materials_signals,
    "safety_signals": FeatureExtractors.extract_safety_signals,
    "checklist_signals": FeatureExtractors.extract_checklist_signals,
    "news_signals": FeatureExtractors.extract_news_signals,
    "press_release_signals": FeatureExtractors.extract_press_release_signals,
    "page_count_features": FeatureExtractors.extract_page_count_features,
    "promotional_signals": FeatureExtractors.extract_promotional_signals,
    "slide_signals": FeatureExtractors.extract_slide_signals,
    "visual_features": FeatureExtractors.extract_visual_features,
    "policy_signals": FeatureExtractors.extract_policy_signals,
    "governance_signals": FeatureExtractors.extract_governance_signals,
}


# =============================================================================
# MAIN CLASSIFIER
# =============================================================================

class SubcategoryClassifier:
    """
    Evidence-based subcategory classifier with measurable, reproducible scoring.
    """
    
    def __init__(self):
        self.extractors = EXTRACTOR_MAP
        self.subcategories = SUBCATEGORIES
    
    def build_context(
        self,
        text: str,
        lines: List[str],
        page_count: int,
        sections: Optional[SectionSignals] = None,
        rubric_scores: Optional[Dict[str, float]] = None
    ) -> ExtractionContext:
        """Build extraction context from document data."""
        rubric_scores = rubric_scores or {}
        return ExtractionContext(
            text=text,
            text_lower=text.lower(),
            lines=lines,
            page_count=page_count,
            sections=sections,
            imrad_score=rubric_scores.get("imrad", 0.0),
            citation_score=rubric_scores.get("citations", 0.0),
            deliverable_score=rubric_scores.get("deliverable", 0.0),
            pedagogy_score=rubric_scores.get("pedagogy", 0.0),
            procedure_score=rubric_scores.get("procedure", 0.0),
        )
    
    def extract_feature(self, ctx: ExtractionContext, feature_name: str) -> FeatureEvidence:
        """Extract a single feature using the appropriate extractor."""
        extractor = self.extractors.get(feature_name)
        if not extractor:
            return FeatureEvidence(
                feature_name=feature_name,
                detected=False,
                score=0.0,
                raw_value=None,
                excerpts=[]
            )
        
        score, raw_value, excerpts = extractor(ctx)
        detected = score > 0.2  # Threshold for detection
        
        return FeatureEvidence(
            feature_name=feature_name,
            detected=detected,
            score=score,
            raw_value=raw_value,
            excerpts=excerpts
        )
    
    def score_subcategory(
        self,
        ctx: ExtractionContext,
        subcat_key: str
    ) -> SubcategoryScore:
        """Score a single subcategory based on all its features."""
        subcat = self.subcategories[subcat_key]
        
        feature_details: Dict[str, FeatureEvidence] = {}
        features_found: List[str] = []
        total_weighted_score = 0.0
        max_possible = 0.0
        
        for feat_def in subcat.detectable_features:
            evidence = self.extract_feature(ctx, feat_def.extractor_key)
            feature_details[feat_def.name] = evidence
            max_possible += feat_def.weight
            
            if evidence.detected:
                features_found.append(feat_def.name)
                total_weighted_score += evidence.score * feat_def.weight
        
        # Calculate confidence
        if max_possible > 0:
            base_confidence = total_weighted_score / max_possible
        else:
            base_confidence = 0.0
        
        # Boost confidence if minimum features met
        if len(features_found) >= subcat.min_features_required:
            confidence = min(1.0, base_confidence * 1.2)
        else:
            confidence = base_confidence * 0.7  # Penalty
        
        # Generate rationale
        features_str = ", ".join(features_found[:3]) if features_found else "minimal signals"
        rationale = subcat.rationale_template.format(
            features_found=features_str,
            confidence=confidence
        )
        
        return SubcategoryScore(
            subcategory_id=subcat.id,
            subcategory_name=subcat.name,
            parent_type=subcat.parent_type.value,
            confidence=round(confidence, 4),
            evidence_score=round(total_weighted_score, 4),
            max_possible_evidence=round(max_possible, 4),
            features_found=features_found,
            feature_details=feature_details,
            rationale=rationale
        )
    
    def classify(
        self,
        ctx: ExtractionContext,
        parent_type_filter: Optional[ParentType] = None
    ) -> Tuple[Optional[SubcategoryScore], List[SubcategoryScore]]:
        """
        Classify document into subcategories.
        
        Returns:
            (best_match, all_scores) - best_match may be None if no good match
        """
        all_scores: List[SubcategoryScore] = []
        
        for key, subcat in self.subcategories.items():
            # Filter by parent type if specified
            if parent_type_filter and subcat.parent_type != parent_type_filter:
                continue
            
            # Skip if not auto-detectable
            if not subcat.auto_detectable:
                continue
            
            score = self.score_subcategory(ctx, key)
            all_scores.append(score)
        
        # Sort by confidence
        all_scores.sort(key=lambda x: x.confidence, reverse=True)
        
        # Find best match above threshold
        best_match = None
        for score in all_scores:
            # Need minimum features AND confidence threshold
            subcat_def = self.subcategories.get(
                next((k for k, v in self.subcategories.items() 
                      if v.id == score.subcategory_id), None)
            )
            if subcat_def and len(score.features_found) >= subcat_def.min_features_required:
                if score.confidence >= 0.35:  # Minimum confidence threshold
                    best_match = score
                    break
        
        return best_match, all_scores
    
    def get_evidence_report(self, ctx: ExtractionContext) -> Dict[str, Any]:
        """Generate a comprehensive evidence report for audit."""
        best_match, all_scores = self.classify(ctx)
        
        return {
            "classification": {
                "best_match": best_match.to_dict() if best_match else None,
                "all_scores": [s.to_dict() for s in all_scores],
            },
            "document_metrics": {
                "page_count": ctx.page_count,
                "text_length": len(ctx.text),
                "line_count": len(ctx.lines),
            },
            "reproducibility": {
                "schema_version": "1.0",
                "scoring_method": "weighted_feature_sum",
                "confidence_formula": "weighted_score / max_possible * boost_factor",
                "minimum_confidence_threshold": 0.35,
                "minimum_features_required": "varies by subcategory",
            }
        }


def score_subcategories(
    text: str,
    lines: List[str],
    page_count: int,
    sections: Optional[SectionSignals] = None,
    rubric_scores: Optional[Dict[str, float]] = None,
    parent_type_filter: Optional[ParentType] = None
) -> Tuple[Optional[SubcategoryScore], List[SubcategoryScore], Dict[str, Any]]:
    """
    Convenience function to classify document into subcategories.
    
    Returns:
        (best_match, all_scores, evidence_report)
    """
    classifier = SubcategoryClassifier()
    ctx = classifier.build_context(text, lines, page_count, sections, rubric_scores)
    best_match, all_scores = classifier.classify(ctx, parent_type_filter)
    report = classifier.get_evidence_report(ctx)
    return best_match, all_scores, report
