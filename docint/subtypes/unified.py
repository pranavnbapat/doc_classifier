from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

from docint.rubrics.subcategory_scorer import FeatureEvidence, SubcategoryScore


BASE_DIR = Path(__file__).resolve().parents[2]
V5_MODEL_PATH = BASE_DIR / "data_model" / "generated" / "v5" / "subcategories_v5_full_model.json"
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
    if V5_MODEL_PATH.exists():
        return V5_MODEL_PATH
    return MERGED_MODEL_PATH if MERGED_MODEL_PATH.exists() else FALLBACK_MODEL_PATH


def _split_indicator_text(text: str) -> List[str]:
    if not text:
        return []
    return [
        item.strip(" .:-")
        for item in re.split(r"[;,]+", text)
        if item.strip(" .:-")
    ]


def _normalize_runtime_category(category: str) -> str:
    if category == "Software Application":
        return "Software"
    return category


def _strength_rank(strength: str) -> int:
    order = {"Strong": 3, "Partial": 2, "Weak/Partial": 1}
    return order.get(strength, 0)


def _strength_to_relation(strength: str, *, is_first: bool) -> str:
    if strength == "Strong" and is_first:
        return "primary"
    return "related"


def _build_feature_groups_from_v5_profile(profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    dominant_rule = str(profile.get("dominant_rule", ""))
    dominant_feature = ""
    match = re.search(r"Must:\s*([a-zA-Z0-9_]+)", dominant_rule)
    if match:
        dominant_feature = match.group(1)

    for item in profile.get("feature_catalog_matches", []):
        feature_id = str(item.get("feature_id", "")).strip()
        if not feature_id:
            continue
        positive_indicators: List[str] = []
        positive_indicators.extend(_split_indicator_text(item.get("definition", "")))
        positive_indicators.extend(_termify(feature_id))
        positive_indicators.extend(_termify(profile.get("name", "")))
        examples = profile.get("examples", "")
        if isinstance(examples, str):
            positive_indicators.extend(_termify(examples))
        weight = 1.0 if feature_id == dominant_feature else 0.78
        groups.append(
            {
                "feature_id": feature_id,
                "weight": weight,
                "definition": item.get("definition", ""),
                "measurement": dominant_rule if feature_id == dominant_feature and dominant_rule else f"Detect evidence for {feature_id}",
                "positive_indicators": list(dict.fromkeys([v for v in positive_indicators if v]))[:12],
                "negative_indicators": [],
            }
        )

    if not groups and profile.get("key_features_text"):
        indicators = _split_indicator_text(str(profile.get("key_features_text", "")))
        if indicators:
            groups.append(
                {
                    "feature_id": f"{profile.get('id', 'profile')}_key_features",
                    "weight": 0.82,
                    "definition": str(profile.get("key_features_text", "")),
                    "measurement": dominant_rule or "Detect document key-feature evidence",
                    "positive_indicators": indicators[:12],
                    "negative_indicators": [],
                }
            )
    return groups


def _runtime_payload_from_v5(payload: Dict[str, Any]) -> Dict[str, Any]:
    unified_subcategories = []
    categories: Dict[str, Dict[str, Any]] = {}

    for item in payload.get("unified_subcategories", []):
        unified_subcategories.append(
            {
                "id": item["id"],
                "name": item["name"],
                "definition": item.get("definition", ""),
                "scope_note": item.get("scope_note", ""),
                "user_label": item.get("user_label", ""),
                "detailed_features": item.get("feature_basis", []),
                "mapped_from_legacy": [],
                "applicability_hints": [
                    _normalize_runtime_category(value)
                    for value in item.get("applicable_categories", [])
                ],
                "imports_from_profiles": [
                    {
                        "source_profile_id": source.get("profile_id"),
                        "source_profile_name": source.get("profile_name"),
                        "source_modality": source.get("modality"),
                        "relation": "primary" if source.get("strength") == "Strong" else "related",
                        "why": source.get("rationale", ""),
                    }
                    for source in item.get("source_profiles", [])
                ],
            }
        )

    for category_name, category_payload in payload.get("source_modalities", {}).items():
        runtime_category = _normalize_runtime_category(category_name)
        profiles: List[Dict[str, Any]] = []
        for profile in category_payload.get("profiles", []):
            mappings = list(profile.get("unified_mappings", []))
            mappings.sort(key=lambda row: _strength_rank(str(row.get("strength", ""))), reverse=True)
            imports_to_unified = [
                {
                    "unified_subcategory_id": mapping.get("unified_subcategory_id"),
                    "relation": _strength_to_relation(str(mapping.get("strength", "")), is_first=index == 0),
                    "why": f"{mapping.get('unified_subcategory_name', '')} via {mapping.get('strength', '')} mapping from v5 exhaustive model.",
                    "strength": mapping.get("strength"),
                }
                for index, mapping in enumerate(mappings)
            ]
            profiles.append(
                {
                    "id": profile.get("id"),
                    "name": profile.get("name"),
                    "definition": profile.get("definition", ""),
                    "scope_note": profile.get("scope_note", ""),
                    "examples": profile.get("examples", ""),
                    "feature_groups": _build_feature_groups_from_v5_profile(profile),
                    "imports_to_unified": imports_to_unified,
                }
            )
        categories[runtime_category] = {"profiles": profiles}

    return {
        "unified_subcategories": unified_subcategories,
        "categories": categories,
    }


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
        tokens = [part]
        if " " in part or "-" in part:
            tokens.extend(re.split(r"[\s-]+", part))
        for token in tokens:
            token = token.strip(" .:-")
            if len(token) < 4:
                continue
            if token in stop_terms:
                continue
            terms.append(token)
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


def _extract_prefixed_values(text: str, prefixes: Tuple[str, ...]) -> List[str]:
    values: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        lower = line.lower()
        for prefix in prefixes:
            if lower.startswith(prefix):
                _, _, rest = line.partition(":")
                if rest:
                    values.extend([part.strip().lower() for part in rest.split("|") if part.strip()])
                break
    return list(dict.fromkeys(values))


def _extract_dataset_context(text: str, filename: str) -> Dict[str, Any]:
    columns = _extract_prefixed_values(text, ("columns",))
    row_values = _extract_prefixed_values(text, tuple(f"row {idx}" for idx in range(0, 31)))
    text_lower = f"{filename}\n{text}".lower()
    return {
        "text_lower": text_lower,
        "columns": columns,
        "row_values": row_values[:120],
        "has_tabular_preview": bool(columns),
        "sheet_mentions": text_lower.count("sheet:"),
        "row_mentions": len(re.findall(r"(?m)^row\s+\d+:", text_lower)),
    }


def _score_dataset_signal(context: Dict[str, Any], feature_id: str) -> tuple[float, List[str]]:
    text_lower = context["text_lower"]
    columns = context["columns"]
    row_values = context["row_values"]
    header_space = " ".join(columns)

    def _hits(terms: List[str], *, in_columns_weight: float = 0.26, in_text_weight: float = 0.14) -> tuple[float, List[str]]:
        hits: List[str] = []
        score = 0.0
        for term in terms:
            matched = False
            if any(term == col or term in col for col in columns):
                hits.append(term)
                score += in_columns_weight
                matched = True
            elif term in header_space or term in " ".join(row_values):
                hits.append(term)
                score += in_text_weight
                matched = True
            elif " " in term and term in text_lower:
                hits.append(term)
                score += in_text_weight
                matched = True
            elif re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text_lower):
                hits.append(term)
                score += in_text_weight
                matched = True
            if matched and len(hits) >= 10:
                break
        return min(1.0, score), hits

    signals: Dict[str, List[str]] = {
        "entity_focus": ["field", "plot", "farm", "parcel", "crop", "animal", "livestock", "soil", "water", "orchard", "block_id", "field_id"],
        "event_log_structure": ["event", "activity", "operation", "task", "schedule", "start_time", "end_time", "status", "operator", "performed_by"],
        "input_application_records": ["fertilizer", "fertiliser", "pesticide", "herbicide", "fungicide", "seed", "dose", "rate", "application", "input"],
        "output_measurement_structure": ["yield", "production", "harvest", "output", "quantity", "weight", "biomass", "tons", "kg", "productivity"],
        "temporal_series_structure": ["date", "time", "timestamp", "datetime", "day", "month", "year", "hour", "minute", "daily", "hourly"],
        "spatial_geometry_structure": ["latitude", "longitude", "lat", "lon", "lng", "geometry", "geojson", "polygon", "wkt", "epsg", "bbox", "x", "y"],
        "derived_aggregation_structure": ["average", "mean", "median", "sum", "total", "index", "score", "kpi", "aggregated", "forecast", "estimate", "derived"],
        "social_survey_structure": ["survey", "respondent", "questionnaire", "likert", "demographic", "gender", "age", "household", "attitude", "response"],
        "machine_telemetry_structure": ["sensor", "telemetry", "tractor", "machine", "equipment", "engine", "rpm", "fuel", "speed", "canbus", "gps", "implement"],
    }

    terms = signals.get(feature_id, [])
    score, hits = _hits(terms)

    if feature_id == "event_log_structure":
        if context["row_mentions"] >= 3 and any(term in text_lower for term in ("operation", "event", "activity", "task")):
            score = min(1.0, score + 0.18)
    elif feature_id == "temporal_series_structure":
        if context["row_mentions"] >= 3 and any(term in columns for term in ("date", "time", "timestamp")):
            score = min(1.0, score + 0.2)
    elif feature_id == "spatial_geometry_structure":
        if any(term in columns for term in ("lat", "lon", "lng", "geometry", "wkt", "epsg")):
            score = min(1.0, score + 0.22)
    elif feature_id == "derived_aggregation_structure":
        if any(term in text_lower for term in ("average", "forecast", "index", "score", "aggregated")):
            score = min(1.0, score + 0.14)

    if context["has_tabular_preview"] and feature_id in signals:
        score = min(1.0, score + 0.08)

    return score, hits[:10]


def _score_software_signal(text: str, filename: str, feature_id: str) -> tuple[float, List[str]]:
    text_lower = f"{filename}\n{text}".lower()
    signals: Dict[str, List[str]] = {
        "workflow_role_and_scope": ["workflow", "planning", "records", "manage", "management", "operations", "module", "user", "platform", "dashboard"],
        "integration_and_interoperability_connectivity": ["api", "integration", "interoperability", "connector", "sync", "import", "export", "plugin", "webhook"],
        "temporal_recording_orientation": ["track", "tracking", "record", "logging", "history", "timeline", "over time", "time series", "monitor"],
        "input_modality_and_capture_mode": ["mobile", "tablet", "offline", "form", "capture", "camera", "gps", "scan", "entry", "input"],
        "field_capture_and_observation_structure": ["field", "scouting", "inspection", "observation", "geo-tagged", "in-field", "capture", "survey in field"],
        "spatial_interaction_and_georeferenced_analysis": ["map", "mapping", "gis", "geospatial", "parcel", "layer", "location", "georeferenced", "spatial"],
        "analysis_visualisation_and_insight_generation": ["analysis", "analytics", "dashboard", "visualisation", "insight", "kpi", "reporting", "chart"],
        "model_prediction_and_scenario_logic": ["simulate", "simulation", "forecast", "prediction", "scenario", "optimisation", "model"],
        "automation_control_and_triggering": ["automation", "automatic", "control", "trigger", "alert", "schedule action", "irrigation control", "actuator"],
        "learning_mechanics_and_training_design": ["training", "learning", "lesson", "quiz", "tutorial", "guided", "practice", "assessment"],
    }

    terms = signals.get(feature_id, [])
    score, hits = _match_terms(text_lower, terms, per_hit_weight=0.18)

    if feature_id == "workflow_role_and_scope" and any(term in text_lower for term in ("platform", "dashboard", "management")):
        score = min(1.0, score + 0.16)
    elif feature_id == "integration_and_interoperability_connectivity" and any(term in text_lower for term in ("api", "integration", "import", "export")):
        score = min(1.0, score + 0.16)
    elif feature_id == "field_capture_and_observation_structure" and any(term in text_lower for term in ("mobile app", "field app", "scouting")):
        score = min(1.0, score + 0.14)
    elif feature_id == "analysis_visualisation_and_insight_generation" and any(term in text_lower for term in ("dashboard", "chart", "analytics")):
        score = min(1.0, score + 0.14)
    elif feature_id == "automation_control_and_triggering" and any(term in text_lower for term in ("control", "automation", "alert")):
        score = min(1.0, score + 0.14)

    return score, hits[:10]


def _feature_specific_signal(text: str, filename: str, category: str, feature_id: str) -> tuple[float, List[str]]:
    runtime_category = _normalize_runtime_category(category)
    if runtime_category == "Dataset":
        return _score_dataset_signal(_extract_dataset_context(text, filename), feature_id)
    if runtime_category == "Software":
        return _score_software_signal(text, filename, feature_id)
    return 0.0, []


def _profile_name_terms(profile: Dict[str, Any]) -> List[str]:
    terms: List[str] = []
    for key in ("name", "definition", "scope_note"):
        terms.extend(_termify(str(profile.get(key, ""))))
    for item in profile.get("examples", []):
        terms.extend(_termify(str(item)))
    return list(dict.fromkeys(terms))


@lru_cache(maxsize=1)
def load_cross_modal_feature_model() -> Dict[str, Any]:
    payload = json.loads(_resolve_model_path().read_text(encoding="utf-8"))
    if payload.get("model_version") == "v5" and "source_modalities" in payload:
        return _runtime_payload_from_v5(payload)
    return payload


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
    category = _normalize_runtime_category(category)
    return list((payload.get("categories", {}).get(category) or {}).get("profiles", []))


@lru_cache(maxsize=8)
def allowed_unified_keys_for_category(category: str) -> Tuple[str, ...]:
    category = _normalize_runtime_category(category)
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


def _score_profile_feature_group(text: str, text_lower: str, feature_group: Dict[str, Any], *, category: str, filename: str) -> tuple[float, float, List[str], List[str]]:
    weight = float(feature_group.get("weight", 0.0))
    positive_terms = [str(item).lower() for item in feature_group.get("positive_indicators", [])]
    negative_terms = [str(item).lower() for item in feature_group.get("negative_indicators", [])]
    pos_score, pos_hits = _match_terms(text_lower, positive_terms, per_hit_weight=0.22)
    feature_id = str(feature_group.get("feature_id", "")).strip()
    feature_score, feature_hits = _feature_specific_signal(text, filename, category, feature_id)
    if feature_score > pos_score:
        pos_score = feature_score
        pos_hits = list(dict.fromkeys(feature_hits + pos_hits))[:12]
    else:
        pos_hits = list(dict.fromkeys(pos_hits + feature_hits))[:12]
    neg_score, neg_hits = _match_terms(text_lower, negative_terms, per_hit_weight=0.12)
    if weight >= 0:
        contribution = max(0.0, pos_score - (0.55 * neg_score)) * weight
        penalty = 0.0
    else:
        contribution = 0.0
        penalty = pos_score * abs(weight)
    return contribution, penalty, pos_hits, neg_hits


def score_intermediate_profiles(*, category: str, text: str, filename: str = "") -> List[Dict[str, Any]]:
    category = _normalize_runtime_category(category)
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
            contribution, penalty, pos_hits, neg_hits = _score_profile_feature_group(
                text,
                text_lower,
                feature_group,
                category=category,
                filename=filename,
            )
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
    runtime_category = _normalize_runtime_category(category)
    defs = load_unified_subtypes()
    support_terms = load_unified_support_terms()
    profile_scores = score_intermediate_profiles(category=runtime_category, text=text, filename=filename)
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
        applicable = not subtype.applicability_hints or runtime_category in subtype.applicability_hints
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
                raw_value={"category": runtime_category, "applicability_hints": list(subtype.applicability_hints)},
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
            rationale_bits.append(f"applicable to {runtime_category}")
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
