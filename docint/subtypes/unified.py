from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

from docint.rubrics.subcategory_scorer import FeatureEvidence, SubcategoryScore


BASE_DIR = Path(__file__).resolve().parents[2]
MERGED_MODEL_PATH = BASE_DIR / "data_model" / "generated" / "v4_improved" / "cross_modal_feature_model_v4.json"
FALLBACK_MODEL_PATH = BASE_DIR / "data_model" / "generated" / "v4" / "subcategory_model_v4.json"


@dataclass(frozen=True)
class UnifiedSubtypeDefinition:
    key: str
    name: str
    definition: str
    scope_note: str
    user_label: str
    detailed_features: Tuple[str, ...]
    mapped_from_legacy: Tuple[str, ...]
    applicability_hints: Tuple[str, ...]


LEGACY_TO_UNIFIED: Dict[str, str] = {
    "journal_article": "technical_and_research_content",
    "Journal article": "technical_and_research_content",
    "article_in_conference_proceedings": "technical_and_research_content",
    "Article in conference proceedings": "technical_and_research_content",
    "chapter_in_edited_volume": "technical_and_research_content",
    "Chapter in edited volume": "technical_and_research_content",
    "book": "technical_and_research_content",
    "Book": "technical_and_research_content",
    "thesis": "technical_and_research_content",
    "Thesis": "technical_and_research_content",
    "technical_report": "technical_and_research_content",
    "Technical Report": "technical_and_research_content",
    "tutorial": "how_to_guides",
    "Tutorial": "how_to_guides",
    "guide_manual": "how_to_guides",
    "Guide/Manual": "how_to_guides",
    "presentation": "talks_and_lectures",
    "Presentation": "talks_and_lectures",
    "news_communication": "explainers",
    "News & Communication": "explainers",
    "informational_booklet": "explainers",
    "Informational Booklet": "explainers",
    "How-To / Instructional Documents": "how_to_guides",
    "Explanatory Documents": "explainers",
    "Technical & Scientific Documents": "technical_and_research_content",
    "Case Study / Practice Documents": "case_studies",
    "Project Reports": "technical_and_research_content",
    "Policy & Regulatory Documents": "technical_and_research_content",
    "Summaries & Factsheets": "explainers",
    "Templates & Reusable Documents": "templates",
    "Templates & Forms": "templates",
    "Presentation / Slide Documents": "talks_and_lectures",
    "educational_training_media": "explainers",
    "Educational/Training Media": "explainers",
    "recorded_session": "talks_and_lectures",
    "Recorded Session": "talks_and_lectures",
    "interview": "interviews",
    "Interview": "interviews",
    "qa_session": "q_and_a_sessions",
    "Q&A Session": "q_and_a_sessions",
    "audio_program": "talks_and_lectures",
    "Audio Program": "talks_and_lectures",
    "demonstration_field_recording": "field_demonstrations",
    "Demonstration/Field Recording": "field_demonstrations",
    "recorded_presentation_webinar": "talks_and_lectures",
    "Recorded presentation/webinar": "talks_and_lectures",
    "data_visualization": "charts_and_graphs",
    "Data Visualization": "charts_and_graphs",
    "figure_image": "photos",
    "Figure/Image": "photos",
    "map": "maps",
    "Map": "maps",
    "Chart/graph": "charts_and_graphs",
    "Infographic": "infographics",
    "Diagram/schematic": "diagrams",
    "Field/observational photograph": "field_demonstrations",
    "Diagnostic photograph": "diagnostic_images",
    "Equipment/infrastructure photograph": "photos",
    "Aerial/remote-sensing image": "maps",
    "geospatial_data": "maps",
    "Geospatial Data": "maps",
    "video_data": "datasets",
    "Video Data": "datasets",
    "audio_data": "datasets",
    "Audio Data": "datasets",
    "image_data": "datasets",
    "Image Data": "datasets",
    "text_data": "datasets",
    "Text Data": "datasets",
    "graph_network_data": "datasets",
    "Graph/Network Data": "datasets",
    "agricultural_production_data": "output_data",
    "Agricultural Production Data": "output_data",
    "environmental_temporal_data": "monitoring_data",
    "Environmental & Temporal Data": "monitoring_data",
    "Entity-focused dataset (Farm/Field Data)": "datasets",
    "Event/operations dataset (Activity Records)": "datasets",
    "Input-use dataset (Input Use Data)": "input_data",
    "Output/production dataset (Production Data)": "output_data",
    "Time-series dataset (Weather & Time Data)": "monitoring_data",
    "Geospatial dataset (Map-based or Geospatial Data)": "maps",
    "Analytical/derived dataset (Analysis & Insights Data)": "technical_and_research_content",
    "Survey/social dataset (Farmer & Survey Data)": "survey_data",
    "Machine/equipment dataset (Machinery & Sensor Data)": "monitoring_data",
    "Farm Management System (FMIS)": "software_tools",
    "Monitoring & Recording Tools": "software_tools",
    "Field Data Collection Apps": "software_tools",
    "Mapping & GIS Tools": "maps",
    "Data Analysis & Dashboard Tools": "software_tools",
    "Simulation & Forecasting Tools": "simulations",
    "Automation & Control Systems": "software_tools",
    "Training & Learning Applications": "how_to_guides",
    "Software Tools": "software_tools",
}


def _resolve_model_path() -> Path:
    return MERGED_MODEL_PATH if MERGED_MODEL_PATH.exists() else FALLBACK_MODEL_PATH


def _termify(text: str) -> List[str]:
    if not text:
        return []
    cleaned = re.sub(r"[\"'`]", "", text.lower())
    cleaned = cleaned.replace("->", " ").replace("_", " ")
    parts = [part.strip(" .:-") for part in re.split(r"[;,/()]+|\s{2,}", cleaned) if part.strip()]
    stop_terms = {
        "content",
        "data",
        "details",
        "topic",
        "format",
        "formats",
        "system",
        "systems",
        "application",
        "applications",
        "tool",
        "tools",
        "structure",
        "structured",
        "design",
        "process",
        "processes",
        "real-world",
        "real world",
    }
    terms: List[str] = []
    for part in parts:
        if len(part) < 4:
            continue
        if part in stop_terms:
            continue
        terms.append(part)
    return list(dict.fromkeys(terms))


def _match_terms(text_lower: str, terms: List[str], *, per_hit_weight: float) -> tuple[float, List[str]]:
    hits: List[str] = []
    for term in terms:
        if len(term) < 4:
            continue
        if " " in term or "-" in term or "/" in term:
            if term in text_lower:
                hits.append(term)
            continue
        pattern = rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])"
        if re.search(pattern, text_lower):
            hits.append(term)
    return min(1.0, len(hits) * per_hit_weight), hits[:12]


def _profile_name_terms(profile: Dict[str, Any]) -> List[str]:
    terms: List[str] = []
    for key in ("name", "definition", "scope_note"):
        terms.extend(_termify(str(profile.get(key, ""))))
    for item in profile.get("examples", []):
        terms.extend(_termify(str(item)))
    return list(dict.fromkeys(terms))


@lru_cache(maxsize=1)
def load_cross_modal_feature_model() -> Dict[str, Any]:
    return json.loads(_resolve_model_path().read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _load_profile_lookup() -> Dict[str, str]:
    payload = load_cross_modal_feature_model()
    lookup: Dict[str, str] = {}
    for category_payload in payload.get("categories", {}).values():
        for profile in category_payload.get("profiles", []):
            for link in profile.get("imports_to_unified", []):
                if link.get("relation") == "primary":
                    unified_id = link.get("unified_subcategory_id")
                    if unified_id:
                        lookup[profile.get("id", "")] = unified_id
                        lookup[profile.get("name", "")] = unified_id
    return lookup


@lru_cache(maxsize=1)
def load_unified_subtypes() -> Dict[str, UnifiedSubtypeDefinition]:
    payload = load_cross_modal_feature_model()
    result: Dict[str, UnifiedSubtypeDefinition] = {}
    for item in payload.get("unified_subcategories", []):
        result[item["id"]] = UnifiedSubtypeDefinition(
            key=item["id"],
            name=item["name"],
            definition=item["definition"],
            scope_note=item["scope_note"],
            user_label=item["user_label"],
            detailed_features=tuple(item.get("detailed_features", [])),
            mapped_from_legacy=tuple(item.get("mapped_from_legacy") or ()),
            applicability_hints=tuple(item.get("applicability_hints", [])),
        )
    return result


@lru_cache(maxsize=8)
def load_category_profiles(category: str) -> List[Dict[str, Any]]:
    payload = load_cross_modal_feature_model()
    return list((payload.get("categories", {}).get(category) or {}).get("profiles", []))


@lru_cache(maxsize=8)
def allowed_unified_keys_for_category(category: str) -> Tuple[str, ...]:
    defs = load_unified_subtypes()
    return tuple(
        key for key, subtype in defs.items()
        if not subtype.applicability_hints or category in subtype.applicability_hints
    )


@lru_cache(maxsize=1)
def load_unified_support_terms() -> Dict[str, Tuple[str, ...]]:
    payload = load_cross_modal_feature_model()
    support: Dict[str, List[str]] = {key: [] for key in load_unified_subtypes().keys()}
    for item in payload.get("unified_subcategories", []):
        subtype_id = item["id"]
        support[subtype_id].extend(_termify(item.get("name", "")))
        support[subtype_id].extend(_termify(item.get("user_label", "")))
        support[subtype_id].extend(_termify(item.get("definition", "")))
        support[subtype_id].extend(_termify(item.get("scope_note", "")))
        for feat in item.get("detailed_features", []):
            support[subtype_id].extend(_termify(str(feat)))
        for imported in item.get("imports_from_profiles", []):
            support[subtype_id].extend(_termify(str(imported.get("source_profile_name", ""))))
    for category_payload in payload.get("categories", {}).values():
        for profile in category_payload.get("profiles", []):
            for link in profile.get("imports_to_unified", []):
                unified_id = link.get("unified_subcategory_id")
                if unified_id not in support:
                    continue
                support[unified_id].extend(_profile_name_terms(profile))
                for feature_group in profile.get("feature_groups", []):
                    support[unified_id].extend(_termify(feature_group.get("definition", "")))
                    for indicator in feature_group.get("positive_indicators", []):
                        support[unified_id].extend(_termify(str(indicator)))
    return {key: tuple(dict.fromkeys(values)) for key, values in support.items()}


def _legacy_key_to_unified(key_or_name: str) -> str | None:
    if key_or_name in LEGACY_TO_UNIFIED:
        return LEGACY_TO_UNIFIED[key_or_name]
    return _load_profile_lookup().get(key_or_name)


def map_probs_to_unified(probs: Dict[str, float]) -> Dict[str, float]:
    aggregated: Dict[str, float] = {}
    for key, prob in probs.items():
        unified = _legacy_key_to_unified(key) or key
        aggregated[unified] = aggregated.get(unified, 0.0) + float(prob)
    total = sum(max(v, 0.0) for v in aggregated.values())
    if total > 0:
        return {k: v / total for k, v in aggregated.items()}
    return aggregated


def _score_profile_feature_group(text_lower: str, feature_group: Dict[str, Any]) -> tuple[float, float, List[str], List[str]]:
    weight = float(feature_group.get("weight", 0.0))
    positive_terms = [str(item).lower() for item in feature_group.get("positive_indicators", [])]
    negative_terms = [str(item).lower() for item in feature_group.get("negative_indicators", [])]
    pos_score, pos_hits = _match_terms(text_lower, positive_terms, per_hit_weight=0.22)
    neg_score, neg_hits = _match_terms(text_lower, negative_terms, per_hit_weight=0.12)
    if weight >= 0:
        contribution = max(0.0, pos_score - (0.55 * neg_score)) * weight
        penalty = 0.0
    else:
        contribution = 0.0
        penalty = pos_score * abs(weight)
    return contribution, penalty, pos_hits, neg_hits


def score_intermediate_profiles(*, category: str, text: str, filename: str = "") -> List[Dict[str, Any]]:
    profiles = load_category_profiles(category)
    text_lower = f"{filename}\n{text}".lower()
    results: List[Dict[str, Any]] = []

    for profile in profiles:
        lexical_terms = _profile_name_terms(profile)
        lexical_score, lexical_hits = _match_terms(text_lower, lexical_terms, per_hit_weight=0.10)

        positive_weight_total = 0.0
        contribution_total = 0.0
        penalty_total = 0.0
        matched_signals: List[str] = list(lexical_hits)
        conflicting_signals: List[str] = []

        for feature_group in profile.get("feature_groups", []):
            weight = float(feature_group.get("weight", 0.0))
            contribution, penalty, pos_hits, neg_hits = _score_profile_feature_group(text_lower, feature_group)
            if weight > 0:
                positive_weight_total += weight
                contribution_total += contribution
            penalty_total += penalty
            matched_signals.extend(pos_hits[:2])
            conflicting_signals.extend(neg_hits[:2])

        normalized_feature_score = (contribution_total / positive_weight_total) if positive_weight_total > 0 else 0.0
        profile_score = max(0.0, min(1.0, (0.62 * normalized_feature_score) + (0.38 * lexical_score) - min(0.42, penalty_total)))
        results.append(
            {
                "profile_id": profile.get("id"),
                "profile_name": profile.get("name"),
                "score": round(profile_score, 4),
                "matched_signals": list(dict.fromkeys(matched_signals))[:10],
                "conflicting_signals": list(dict.fromkeys(conflicting_signals))[:8],
                "imports_to_unified": profile.get("imports_to_unified", []),
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results


def score_unified_subcategories(
    *,
    text: str,
    category: str,
    legacy_probs: Dict[str, float] | None = None,
    filename: str = "",
) -> List[SubcategoryScore]:
    defs = load_unified_subtypes()
    support_terms = load_unified_support_terms()
    profile_scores = score_intermediate_profiles(category=category, text=text, filename=filename)
    profile_by_unified: Dict[str, List[Dict[str, Any]]] = {}
    for profile in profile_scores:
        for link in profile.get("imports_to_unified", []):
            unified_id = link.get("unified_subcategory_id")
            if not unified_id:
                continue
            scaled = profile["score"] * (1.0 if link.get("relation") == "primary" else 0.62)
            profile_by_unified.setdefault(unified_id, []).append(
                {
                    "profile_id": profile["profile_id"],
                    "profile_name": profile["profile_name"],
                    "score": round(scaled, 4),
                    "base_score": profile["score"],
                    "relation": link.get("relation"),
                    "why": link.get("why"),
                    "matched_signals": profile.get("matched_signals", []),
                }
            )

    text_lower = f"{filename}\n{text}".lower()
    unified_prior = map_probs_to_unified(legacy_probs or {})
    scores: List[SubcategoryScore] = []

    for key, subtype in defs.items():
        lexical_terms: List[str] = []
        lexical_terms.extend(_termify(subtype.name))
        lexical_terms.extend(_termify(subtype.user_label))
        lexical_terms.extend(_termify(subtype.definition))
        lexical_terms.extend(_termify(subtype.scope_note))
        lexical_terms.extend(_termify(" ".join(subtype.detailed_features)))
        lexical_terms.extend(list(support_terms.get(key, ())))
        lexical_terms = list(dict.fromkeys(lexical_terms))
        direct_score, direct_hits = _match_terms(text_lower, lexical_terms, per_hit_weight=0.09)

        supporting_profiles = sorted(profile_by_unified.get(key, []), key=lambda item: item["score"], reverse=True)
        profile_score = supporting_profiles[0]["score"] if supporting_profiles else 0.0
        profile_names = [item["profile_name"] for item in supporting_profiles[:3]]
        profile_hits: List[str] = []
        for item in supporting_profiles[:2]:
            profile_hits.extend(item.get("matched_signals", [])[:2])
        profile_hits = list(dict.fromkeys(profile_hits))[:6]

        prior_score = float(unified_prior.get(key, 0.0))
        applicable = not subtype.applicability_hints or category in subtype.applicability_hints
        agreement_bonus = 0.08 if ((direct_score >= 0.14 and profile_score >= 0.20) or (profile_score >= 0.20 and prior_score >= 0.18)) else 0.0

        fused_core = max(direct_score, profile_score)
        total_score = (0.48 * fused_core) + (0.22 * profile_score) + (0.16 * prior_score) + agreement_bonus
        if not applicable:
            total_score = 0.0
        elif fused_core < 0.08 and prior_score < 0.10:
            total_score = 0.0
        elif fused_core == 0.0:
            total_score = min(total_score, 0.32)
        total_score = min(1.0, max(0.0, total_score))

        details = {
            "unified_lexical_signals": FeatureEvidence(
                feature_name="unified_lexical_signals",
                detected=bool(direct_hits),
                score=round(direct_score, 4),
                raw_value={"matches": direct_hits[:8]},
                excerpts=direct_hits[:3],
            ),
            "intermediate_profile_support": FeatureEvidence(
                feature_name="intermediate_profile_support",
                detected=profile_score > 0.0,
                score=round(profile_score, 4),
                raw_value={
                    "top_profiles": [
                        {
                            "profile_id": item["profile_id"],
                            "profile_name": item["profile_name"],
                            "score": item["score"],
                            "relation": item["relation"],
                        }
                        for item in supporting_profiles[:3]
                    ]
                },
                excerpts=profile_names[:2],
            ),
            "category_type_prior": FeatureEvidence(
                feature_name="category_type_prior",
                detected=prior_score > 0.0,
                score=round(prior_score, 4),
                raw_value={"mapped_prior_probability": round(prior_score, 4)},
                excerpts=[],
            ),
            "category_applicability": FeatureEvidence(
                feature_name="category_applicability",
                detected=applicable,
                score=1.0 if applicable else 0.0,
                raw_value={"category": category, "applicability_hints": list(subtype.applicability_hints)},
                excerpts=[],
            ),
        }
        features_found = [name for name, ev in details.items() if ev.detected]
        rationale_bits: List[str] = []
        if direct_hits:
            rationale_bits.append(f"semantic cues: {', '.join(direct_hits[:3])}")
        if profile_names:
            rationale_bits.append(f"profile support: {', '.join(profile_names[:2])}")
        if profile_hits:
            rationale_bits.append(f"profile signals: {', '.join(profile_hits[:3])}")
        if prior_score > 0.0:
            rationale_bits.append(f"legacy prior: {prior_score:.2f}")
        if agreement_bonus > 0.0:
            rationale_bits.append("signal/profile agreement")
        if applicable and (direct_score > 0.0 or profile_score > 0.0 or prior_score > 0.0):
            rationale_bits.append(f"applicable to {category}")
        rationale = "; ".join(rationale_bits) if rationale_bits else "minimal unified subtype evidence"

        scores.append(
            SubcategoryScore(
                subcategory_id=key,
                subcategory_name=subtype.name,
                parent_type="unified",
                confidence=round(total_score, 4),
                evidence_score=round(total_score, 4),
                max_possible_evidence=1.0,
                features_found=features_found,
                feature_details=details,
                rationale=f"{subtype.name} selected via {rationale} (confidence: {total_score:.2f})",
            )
        )

    scores.sort(key=lambda item: item.confidence, reverse=True)
    return scores
