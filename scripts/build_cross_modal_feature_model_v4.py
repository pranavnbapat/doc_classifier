from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


BASE_DIR = Path(__file__).resolve().parents[1]
V4_DIR = BASE_DIR / "data_model" / "generated" / "v4"
IMPROVED_DIR = BASE_DIR / "data_model" / "generated" / "v4_improved"
OUT_PATH = IMPROVED_DIR / "cross_modal_feature_model_v4.json"


PROFILE_IMPORTS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "Document": {
        "how_to_instructional_documents": {
            "primary_unified_id": "how_to_guides",
            "why": "Definition, scope, and feature groups all emphasise ordered steps, imperative action language, prerequisites, and guided execution.",
        },
        "explanatory_documents": {
            "primary_unified_id": "explainers",
            "why": "This profile is concept-focused and designed to improve understanding rather than drive a procedure or narrate a case.",
        },
        "technical_scientific_documents": {
            "primary_unified_id": "technical_and_research_content",
            "why": "Formal structure, evidence density, technical terminology, and analytical framing align directly with technical and research-oriented content.",
        },
        "case_study_practice_documents": {
            "primary_unified_id": "case_studies",
            "why": "The profile is defined by a context-to-action-to-result arc anchored in a concrete real-world implementation.",
        },
        "project_reports": {
            "primary_unified_id": "technical_and_research_content",
            "related_unified_ids": ["case_studies"],
            "why": "Project reports usually present structured evidence, deliverables, methods, progress, and outcomes; that makes them primarily technical/research content, with some overlap into case-study reporting.",
        },
        "policy_regulatory_documents": {
            "primary_unified_id": "technical_and_research_content",
            "related_unified_ids": ["explainers"],
            "why": "These documents are formal, structured, and evidence- or rule-driven. They often explain obligations, but their primary character remains formal technical/regulatory content.",
        },
        "summaries_and_factsheets": {
            "primary_unified_id": "explainers",
            "why": "Summaries, factsheets, brochures, and flyers are condensed explanatory assets whose main purpose is rapid understanding.",
        },
        "informational_communication_documents": {
            "primary_unified_id": "explainers",
            "related_unified_ids": ["talks_and_lectures"],
            "why": "This profile focuses on communicating and explaining information to a broad audience, which aligns best with explainers rather than procedural or research-heavy content.",
        },
        "templates_reusable_documents": {
            "primary_unified_id": "templates",
            "why": "The profile is explicitly about predefined reusable structure, placeholders, and fill-in formats.",
        },
    },
    "Audio": {
        "interview_practitioner_perspective": {
            "primary_unified_id": "interviews",
            "why": "The audio is centred on one person's experience, interviewer-led prompts, and testimonial reflection.",
        },
        "expert_commentary_explainer": {
            "primary_unified_id": "explainers",
            "related_unified_ids": ["talks_and_lectures"],
            "why": "The profile is a sustained explanatory monologue intended to clarify a concept or method rather than present a formal event recording.",
        },
        "case_study_practice_story": {
            "primary_unified_id": "case_studies",
            "why": "The profile is explicitly defined by a practical context-action-outcome story arc and reflective lessons learned.",
        },
        "panel_discussion": {
            "primary_unified_id": "panel_discussions",
            "why": "Multiple substantive speakers and moderator-led viewpoint exchange align directly with the panel discussion unified subtype.",
        },
        "q_and_a": {
            "primary_unified_id": "q_and_a_sessions",
            "why": "The dominant rule and feature set are explicit question-answer sequencing and prompt-driven topic changes.",
        },
        "talk_lecture": {
            "primary_unified_id": "talks_and_lectures",
            "why": "This profile is a structured presentation by a dominant speaker with formal topic progression.",
        },
        "how_to_procedure_guide": {
            "primary_unified_id": "how_to_guides",
            "why": "The profile is built around stepwise procedural instruction, imperative actions, and task completion cues.",
        },
    },
    "Video": {
        "field_demonstration_walkthrough": {
            "primary_unified_id": "field_demonstrations",
            "why": "On-site observational context and practice-in-place are the core traits of the field demonstration unified subtype.",
        },
        "how_to_procedure_demonstration": {
            "primary_unified_id": "how_to_guides",
            "related_unified_ids": ["field_demonstrations"],
            "why": "The video is instructional and stepwise first, even though it may occur in a real-world context.",
        },
        "case_study": {
            "primary_unified_id": "case_studies",
            "why": "The defining features are a single-case problem-solution-outcome narrative and explicit lessons learned.",
        },
        "explainer_documentary": {
            "primary_unified_id": "explainers",
            "related_unified_ids": ["talks_and_lectures"],
            "why": "The primary goal is conceptual explanation of a system or topic, not an event recording, tool operation, or procedural guide.",
        },
        "interview_practitioner_perspective": {
            "primary_unified_id": "interviews",
            "why": "The video is centred on one speaker's experience and testimonial reflections.",
        },
        "expert_q_and_a_session": {
            "primary_unified_id": "q_and_a_sessions",
            "why": "Question-led turn-taking is the dominant structural signal for this profile.",
        },
        "panel_discussion": {
            "primary_unified_id": "panel_discussions",
            "why": "Moderator-led multi-speaker exchange and contrasting viewpoints align directly with panel discussions.",
        },
        "recorded_presentation_webinar": {
            "primary_unified_id": "talks_and_lectures",
            "why": "The dominant rule is slide- or screen-led formal presentation, which corresponds to talks and lectures.",
        },
        "tool_machinery_software_walkthrough": {
            "primary_unified_id": "tool_walkthroughs",
            "related_unified_ids": ["how_to_guides"],
            "why": "The central focus is operating and navigating a tool, machine, or interface rather than giving a generic procedure.",
        },
        "simulation_animation_model_visualisation": {
            "primary_unified_id": "simulations",
            "why": "Model-based representation, scenarios, and animation-driven explanation are the defining cues of the simulations unified subtype.",
        },
    },
    "Image": {
        "chart_graph": {
            "primary_unified_id": "charts_and_graphs",
            "why": "Numeric data encoded visually with axes, legends, and comparison structure aligns exactly with charts and graphs.",
        },
        "infographic": {
            "primary_unified_id": "infographics",
            "why": "Multi-panel visual summary composition with icons and short text blocks maps directly to infographics.",
        },
        "diagram_schematic": {
            "primary_unified_id": "diagrams",
            "why": "Abstract explanatory rendering, labelled relationships, and schematic structure align directly with diagrams.",
        },
        "map": {
            "primary_unified_id": "maps",
            "why": "Geospatial reference structure, cartographic support elements, and spatial distribution are the defining map signals.",
        },
        "field_observational_photograph": {
            "primary_unified_id": "field_demonstrations",
            "related_unified_ids": ["photos"],
            "why": "This profile is more specific than generic photos: it shows real-world practices or conditions in situ, which aligns with observational field demonstration content.",
        },
        "diagnostic_photograph": {
            "primary_unified_id": "diagnostic_images",
            "why": "Close-up problem-identification intent and detailed defect/symptom visibility align directly with diagnostic images.",
        },
        "equipment_infrastructure_photograph": {
            "primary_unified_id": "photos",
            "related_unified_ids": ["tool_walkthroughs"],
            "why": "The asset is still a real-world photo first, but with equipment as the central subject rather than a field scene or diagram.",
        },
        "aerial_remote_sensing_image": {
            "primary_unified_id": "maps",
            "related_unified_ids": ["photos"],
            "why": "Remote-sensing and aerial imagery are primarily spatial surface representations; that makes them closer to the maps subtype than to generic photos.",
        },
    },
    "Dataset": {
        "entity_focused_dataset_farm_field_data": {
            "primary_unified_id": "datasets",
            "why": "The core characteristic is structured machine-readable data centred on domain entities rather than a more specialised dataset subtype.",
        },
        "event_operations_dataset_activity_records": {
            "primary_unified_id": "datasets",
            "related_unified_ids": ["monitoring_data"],
            "why": "These are structured operational records first; they may be time-ordered, but they are not purely monitoring time series.",
        },
        "input_use_dataset_input_use_data": {
            "primary_unified_id": "input_data",
            "why": "The entire profile is defined by application of inputs, quantities, rates, and timing of use.",
        },
        "output_production_dataset_production_data": {
            "primary_unified_id": "output_data",
            "why": "The main variables measure yield, production, or realised outputs and performance.",
        },
        "time_series_dataset_weather_time_data": {
            "primary_unified_id": "monitoring_data",
            "why": "Repeated time-indexed measurements and temporal patterns align directly with monitoring data.",
        },
        "geospatial_dataset_map_based_or_geospatial_data": {
            "primary_unified_id": "maps",
            "related_unified_ids": ["datasets"],
            "why": "Spatial geometry and georeferenced structure are primary, making this best represented as a map-oriented unified subtype.",
        },
        "analytical_derived_dataset_analysis_insights_data": {
            "primary_unified_id": "technical_and_research_content",
            "related_unified_ids": ["datasets"],
            "why": "Processed, derived, modelled, and insight-oriented datasets are closer to technical and analytical content than to raw downloadable data alone.",
        },
        "survey_social_dataset_farmer_survey_data": {
            "primary_unified_id": "survey_data",
            "why": "Questionnaire structure, human-response orientation, and demographic variables align directly with survey data.",
        },
        "machine_equipment_dataset_machinery_sensor_data": {
            "primary_unified_id": "monitoring_data",
            "related_unified_ids": ["datasets"],
            "why": "Telemetry and repeated sensor readings behave most like monitoring data, even when the monitored subject is a machine.",
        },
    },
    "Software": {
        "farm_management_system_fmis": {
            "primary_unified_id": "software_tools",
            "why": "This is an interactive multi-workflow application that performs operational tasks across a defined farm-management workflow.",
        },
        "monitoring_recording_tools": {
            "primary_unified_id": "software_tools",
            "related_unified_ids": ["monitoring_data"],
            "why": "The artifact is software first, even though its function is time-oriented monitoring and recording.",
        },
        "field_data_collection_apps": {
            "primary_unified_id": "software_tools",
            "related_unified_ids": ["how_to_guides"],
            "why": "These are still interactive applications with defined workflows; they are not themselves instructional content.",
        },
        "mapping_gis_tools": {
            "primary_unified_id": "maps",
            "related_unified_ids": ["software_tools"],
            "why": "Spatial interaction and georeferenced analysis are the core user-facing mode, so the map-oriented unified subtype is the best semantic fit.",
        },
        "data_analysis_dashboard_tools": {
            "primary_unified_id": "software_tools",
            "related_unified_ids": ["technical_and_research_content"],
            "why": "The artifact remains software, although its outputs may surface technical and analytical content.",
        },
        "simulation_forecasting_tools": {
            "primary_unified_id": "simulations",
            "related_unified_ids": ["software_tools"],
            "why": "The defining behaviour is model-based prediction and scenario logic, which aligns directly with simulations.",
        },
        "automation_control_systems": {
            "primary_unified_id": "software_tools",
            "related_unified_ids": ["tool_walkthroughs"],
            "why": "These are still software systems, but with stronger operational-control behaviour than a generic app.",
        },
        "training_learning_applications": {
            "primary_unified_id": "how_to_guides",
            "related_unified_ids": ["software_tools"],
            "why": "The primary semantic function is guided learning and skill acquisition, which aligns best with how-to guidance in the current unified taxonomy.",
        },
    },
}


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_feature_ids(profile: Dict[str, Any]) -> List[str]:
    return [item.get("feature_id", "") for item in profile.get("feature_groups", []) if item.get("feature_id")]


def _infer_profile_key(profile: Dict[str, Any]) -> str:
    return profile.get("id") or profile.get("name") or ""


def _build_category_entry(path: Path) -> Dict[str, Any]:
    payload = _load_json(path)
    category = payload["category"]
    mappings = PROFILE_IMPORTS[category]
    profiles: List[Dict[str, Any]] = []
    imports_flat: List[Dict[str, Any]] = []

    for profile in payload.get("profiles", []):
        profile_key = _infer_profile_key(profile)
        if profile_key not in mappings:
            raise KeyError(f"Missing unified mapping for {category}:{profile_key}")
        mapping = mappings[profile_key]
        imported_feature_ids = _iter_feature_ids(profile)
        imports_to_unified = [
            {
                "unified_subcategory_id": mapping["primary_unified_id"],
                "relation": "primary",
                "why": mapping["why"],
                "imported_feature_ids": imported_feature_ids,
                "imported_from_profile_id": profile_key,
                "imported_from_profile_name": profile["name"],
            }
        ]
        for related in mapping.get("related_unified_ids", []):
            imports_to_unified.append(
                {
                    "unified_subcategory_id": related,
                    "relation": "related",
                    "why": f"Secondary semantic overlap with {related} based on the same source profile evidence, but weaker than the primary mapping.",
                    "imported_feature_ids": imported_feature_ids,
                    "imported_from_profile_id": profile_key,
                    "imported_from_profile_name": profile["name"],
                }
            )

        enriched_profile = dict(profile)
        enriched_profile["imports_to_unified"] = imports_to_unified
        profiles.append(enriched_profile)

        for item in imports_to_unified:
            imports_flat.append(
                {
                    "category": category,
                    "source_profile_id": profile_key,
                    "source_profile_name": profile["name"],
                    "target_unified_subcategory_id": item["unified_subcategory_id"],
                    "relation": item["relation"],
                    "why": item["why"],
                    "imported_feature_ids": item["imported_feature_ids"],
                    "source_profile_file": str(path.relative_to(BASE_DIR)),
                }
            )

    return {
        "category": category,
        "source_profile_file": payload["source_profile_file"],
        "status": payload.get("status"),
        "design_notes": payload.get("design_notes", {}),
        "profiles": profiles,
        "imports_flat": imports_flat,
    }


def main() -> None:
    base_model = _load_json(V4_DIR / "subcategory_model_v4.json")
    improved_files = {
        "Document": IMPROVED_DIR / "document_profile_features_v4.json",
        "Audio": IMPROVED_DIR / "audio_profile_features_v4.json",
        "Video": IMPROVED_DIR / "video_profile_features_v4.json",
        "Image": IMPROVED_DIR / "image_profile_features_v4.json",
        "Dataset": IMPROVED_DIR / "dataset_profile_features_v4.json",
        "Software": IMPROVED_DIR / "software_profile_features_v4.json",
    }

    categories: Dict[str, Any] = {}
    imports_flat: List[Dict[str, Any]] = []
    unified_rollups: Dict[str, List[Dict[str, Any]]] = {item["id"]: [] for item in base_model.get("unified_subcategories", [])}

    for category, path in improved_files.items():
        entry = _build_category_entry(path)
        categories[category] = {
            "source_profile_file": entry["source_profile_file"],
            "status": entry["status"],
            "design_notes": entry["design_notes"],
            "profiles": entry["profiles"],
        }
        imports_flat.extend(entry["imports_flat"])
        for imported in entry["imports_flat"]:
            unified_rollups[imported["target_unified_subcategory_id"]].append(imported)

    unified_enriched: List[Dict[str, Any]] = []
    for subtype in base_model.get("unified_subcategories", []):
        enriched = dict(subtype)
        enriched["imports_from_profiles"] = unified_rollups.get(subtype["id"], [])
        unified_enriched.append(enriched)

    merged_payload = {
        "schema_version": "v4_cross_modal_feature_model",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "design_status": "merged_source_of_truth_for_intermediate_profile_scoring",
        "design_notes": {
            "purpose": "Merged cross-modal feature model built from the v4 unified subtype taxonomy plus improved category-specific intermediate profile definitions.",
            "runtime_intent": "Use category-specific profiles as first-class intermediate scoring layers, then aggregate into unified subcategories for final API outputs.",
            "provenance_policy": "Every imported profile records which unified subcategory it feeds, which feature groups were imported, and why that mapping was chosen.",
        },
        "source_files": {
            "base_unified_model": str((V4_DIR / "subcategory_model_v4.json").relative_to(BASE_DIR)),
            "category_profile_files": {key: str(path.relative_to(BASE_DIR)) for key, path in improved_files.items()},
        },
        "unified_subcategories": unified_enriched,
        "categories": categories,
        "profile_to_unified_imports": imports_flat,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(merged_payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"[OK] Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
