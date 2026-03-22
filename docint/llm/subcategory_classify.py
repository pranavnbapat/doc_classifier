# docint/llm/subcategory_classify.py
"""
LLM classification for subcategories - aligns with evidence-based scoring.
Supports both text models (Qwen) and vision models (InternVL).
"""

from __future__ import annotations

import json
import os
import base64
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from pathlib import Path

from openai import OpenAI

# Import subcategory definitions
from docint.rubrics.subcategories import SUBCATEGORIES


# Build subcategory list from data model
SUBCAT_TYPES = list(SUBCATEGORIES.keys())


def build_subcategory_prompt() -> str:
    """Build LLM prompt using actual subcategory definitions."""
    prompt_lines = [
        "You are a document subcategory classifier.",
        "Classify the document into ONE subcategory based on observable evidence in the document.",
        "Use the same measurable criteria vocabulary as the heuristic classifier.",
        "",
    ]
    
    for key, subcat in SUBCATEGORIES.items():
        features_desc = ", ".join([f.name for f in subcat.detectable_features])
        prompt_lines.append(f"- {key} ({subcat.name})")
        prompt_lines.append(f"  parent_type: {subcat.parent_type.value}")
        prompt_lines.append(f"  description: {subcat.description}")
        prompt_lines.append(f"  detectable_features: {features_desc}")
        if subcat.positive_signal_hints:
            prompt_lines.append(
                "  positive_signals: "
                + ", ".join(subcat.positive_signal_hints)
            )
        if subcat.negative_signal_hints:
            prompt_lines.append(
                "  negative_signals: "
                + ", ".join(subcat.negative_signal_hints)
            )
        if subcat.close_competitors:
            prompt_lines.append(
                "  close_competitors: "
                + ", ".join(subcat.close_competitors)
            )
        prompt_lines.append(
            f"  minimum_features_required: {subcat.min_features_required}"
        )
        prompt_lines.append("")
    
    prompt_lines.extend([
        "Return ONLY valid JSON with:",
        "1. 'subcategory': the key of the best matching subcategory",
        "2. 'confidence': your confidence 0.0-1.0",
        "3. 'rationale': brief explanation citing specific evidence from the text or pages",
        "4. 'matched_signals': short list of criteria-consistent signals that support the chosen class",
        "5. 'conflicting_signals': short list of signals that create ambiguity or point elsewhere",
        "6. 'closest_alternative': the next most plausible subcategory key",
        "7. 'probs': object with probability for EACH subcategory (should sum to 1.0)",
        "",
        "Be honest about uncertainty. If multiple categories seem possible, distribute probability accordingly.",
        "Prefer signals grounded in the supplied taxonomy, such as IMRaD structure, governance references, slide indicators, regulatory update markers, or tutorial structure.",
    ])
    
    return "\n".join(prompt_lines)


# Pre-built system prompt
SYSTEM_PROMPT = build_subcategory_prompt()


def build_schema() -> str:
    """Build JSON schema for response."""
    probs_template = "\n".join([f'    "{k}": 0.0,' for k in SUBCAT_TYPES])
    
    return f"""Return ONLY valid JSON:
{{
  "subcategory": "one_of_the_keys_below",
  "confidence": 0.0,
  "rationale": "explanation with evidence",
  "matched_signals": ["signal_1", "signal_2"],
  "conflicting_signals": ["signal_1"],
  "closest_alternative": "one_of_the_keys_below",
  "probs": {{
{probs_template}
  }}
}}

Available subcategory keys:
""" + "\n".join([f"- {k}" for k in SUBCAT_TYPES])


@dataclass
class SubcategoryLlmResult:
    """Result from LLM subcategory classification."""
    subcategory_key: str
    subcategory_name: str
    parent_type: str
    confidence: float
    rationale: str
    probs: Dict[str, float]  # Probabilities for all subcategories
    raw_json: Dict[str, Any]


def normalize_subcategory_probs(probs: Dict[str, float]) -> Dict[str, float]:
    """Normalize probabilities to sum to 1.0."""
    # Ensure all keys exist
    out = {k: float(probs.get(k, 0.0)) for k in SUBCAT_TYPES}
    
    s = sum(out.values())
    if s <= 0:
        # Uniform fallback
        return {k: 1.0 / len(SUBCAT_TYPES) for k in SUBCAT_TYPES}
    
    return {k: round(v / s, 4) for k, v in out.items()}


def llm_classify_subcategories_text(
    text: str,
    *,
    base_url: str,
    api_key: str,
    model: str,
    max_chars: int = 15000,
    temperature: float = 0.2,
    timeout: float = 60.0,
) -> SubcategoryLlmResult:
    """
    Classify document text into subcategories using text-based LLM.
    
    Args:
        text: Document text content
        base_url: vLLM/OpenAI compatible endpoint
        api_key: API key
        model: Model name
        max_chars: Max text length to send
        temperature: Sampling temperature
        timeout: Request timeout
    
    Returns:
        SubcategoryLlmResult with classification
    """
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    
    # Truncate text if needed
    if len(text) > max_chars:
        head_len = int(max_chars * 0.7)
        tail_len = int(max_chars * 0.3)
        text = text[:head_len] + "\n\n[...TRUNCATED...]\n\n" + text[-tail_len:]
    
    schema = build_schema()
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": schema + "\n\nDOCUMENT TEXT:\n" + text},
    ]
    
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    
    raw = resp.choices[0].message.content or ""
    
    # Parse JSON response
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown/code blocks
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(raw[start:end + 1])
        else:
            raise ValueError(f"Could not parse LLM response: {raw[:200]}")
    
    # Extract and normalize
    subcat_key = data.get("subcategory", "")
    if subcat_key not in SUBCATEGORIES:
        # Try to find closest match or use highest prob
        probs = data.get("probs", {})
        subcat_key = max(probs.items(), key=lambda x: x[1])[0] if probs else SUBCAT_TYPES[0]
    
    subcat_def = SUBCATEGORIES[subcat_key]
    probs = normalize_subcategory_probs(data.get("probs", {}))
    
    return SubcategoryLlmResult(
        subcategory_key=subcat_key,
        subcategory_name=subcat_def.name,
        parent_type=subcat_def.parent_type.value,
        confidence=float(data.get("confidence", probs.get(subcat_key, 0))),
        rationale=str(data.get("rationale", "")).strip(),
        probs=probs,
        raw_json=data,
    )


def llm_classify_subcategories_vision_batch(
    pdf_path: str,
    page_numbers: List[int],
    *,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float = 0.2,
    timeout: float = 120.0,
) -> SubcategoryLlmResult:
    """
    Classify specific pages of a PDF using vision-language model.
    
    Args:
        pdf_path: Path to PDF file
        page_numbers: List of page numbers to analyze (1-indexed)
        base_url: VLM endpoint
        api_key: API key
        model: VLM model name
        temperature: Sampling temperature
        timeout: Request timeout
    
    Returns:
        SubcategoryLlmResult with classification
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        raise ImportError("pdf2image required for vision classification. Install: pip install pdf2image")
    
    # Convert specific pages to images
    images = convert_from_path(pdf_path, first_page=min(page_numbers), last_page=max(page_numbers))
    
    # Map images to page numbers
    page_images = {}
    for i, page_num in enumerate(range(min(page_numbers), max(page_numbers) + 1)):
        if page_num in page_numbers and i < len(images):
            page_images[page_num] = images[i]
    
    if not page_images:
        raise ValueError("Could not convert PDF pages to images")
    
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    
    # Build message with images
    content = [
        {"type": "text", "text": SYSTEM_PROMPT + "\n\n" + build_schema()},
    ]
    
    # Add images in order with page labels
    for page_num in sorted(page_images.keys()):
        img = page_images[page_num]
        import io
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        content.append({
            "type": "text",
            "text": f"\n[Page {page_num}]\n"
        })
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_base64}",
                "detail": "high"
            }
        })
    
    messages = [
        {"role": "user", "content": content}
    ]
    
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=2000,
    )
    
    raw = resp.choices[0].message.content or ""
    
    # Parse JSON
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(raw[start:end + 1])
        else:
            raise ValueError(f"Could not parse VLM response: {raw[:200]}")
    
    # Extract results
    subcat_key = data.get("subcategory", "")
    if subcat_key not in SUBCATEGORIES:
        probs = data.get("probs", {})
        subcat_key = max(probs.items(), key=lambda x: x[1])[0] if probs else SUBCAT_TYPES[0]
    
    subcat_def = SUBCATEGORIES[subcat_key]
    probs = normalize_subcategory_probs(data.get("probs", {}))
    
    return SubcategoryLlmResult(
        subcategory_key=subcat_key,
        subcategory_name=subcat_def.name,
        parent_type=subcat_def.parent_type.value,
        confidence=float(data.get("confidence", probs.get(subcat_key, 0))),
        rationale=str(data.get("rationale", "")).strip(),
        probs=probs,
        raw_json={**data, "vision_model": True, "pages_analyzed": page_numbers},
    )


def _sample_vision_pages(total_pages: int, max_total_pages: int) -> List[int]:
    """Deterministically sample representative pages from the document."""
    pages_to_process = min(total_pages, max_total_pages)
    if pages_to_process <= 3:
        return list(range(1, pages_to_process + 1))

    if pages_to_process <= 6:
        return sorted(set([1, 2, pages_to_process // 2 or 1, pages_to_process - 1, pages_to_process]))

    sample_size = min(8, pages_to_process)
    if sample_size <= 1:
        return [1]

    positions = [0.0, 0.12, 0.28, 0.5, 0.72, 0.88, 1.0]
    if sample_size < len(positions):
        positions = positions[:sample_size]
        positions[-1] = 1.0

    sampled_pages = []
    for pos in positions:
        idx = round((pages_to_process - 1) * pos)
        sampled_pages.append(idx + 1)

    return sorted(set(max(1, min(pages_to_process, p)) for p in sampled_pages))


def llm_classify_subcategories_vision_sampled(
    pdf_path: str,
    *,
    base_url: str,
    api_key: str,
    model: str,
    max_total_pages: int = 8,
    temperature: float = 0.2,
) -> SubcategoryLlmResult:
    """
    Classify PDF using vision model on deterministic sampled pages.
    
    Args:
        pdf_path: Path to PDF file
        base_url: VLM endpoint
        api_key: API key
        model: VLM model name
        max_total_pages: Maximum sampled pages to process
        temperature: Sampling temperature
    
    Returns:
        SubcategoryLlmResult from sampled pages
    """
    from PyPDF2 import PdfReader
    
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    page_numbers = _sample_vision_pages(total_pages, max_total_pages)
    result = llm_classify_subcategories_vision_batch(
        pdf_path,
        page_numbers,
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=temperature,
    )
    result.raw_json = {
        **result.raw_json,
        "vision_sampling": "deterministic_stratified",
        "total_pages_in_document": total_pages,
        "sampled_page_count": len(page_numbers),
    }
    return result


# Alias for backward compatibility
llm_classify_subcategories_vision = llm_classify_subcategories_vision_sampled
