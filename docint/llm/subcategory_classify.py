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
        "You are a document subcategory classifier. Classify the document into ONE of these subcategories based on its content, structure, and purpose:",
        "",
    ]
    
    for key, subcat in SUBCATEGORIES.items():
        features_desc = ", ".join([f.name for f in subcat.detectable_features[:3]])
        prompt_lines.append(f"- {key}: {subcat.description} (look for: {features_desc})")
    
    prompt_lines.extend([
        "",
        "Return ONLY valid JSON with:",
        "1. 'subcategory': the key of the best matching subcategory",
        "2. 'confidence': your confidence 0.0-1.0",
        "3. 'rationale': brief explanation citing specific evidence from the text",
        "4. 'probs': object with probability for EACH subcategory (should sum to 1.0)",
        "",
        "Be honest about uncertainty - if multiple categories seem possible, distribute probability accordingly.",
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


def llm_classify_subcategories_vision_sliding_window(
    pdf_path: str,
    *,
    base_url: str,
    api_key: str,
    model: str,
    window_size: int = 4,
    overlap: int = 2,
    max_total_pages: int = 20,
    temperature: float = 0.2,
) -> SubcategoryLlmResult:
    """
    Classify PDF using vision model with sliding window (map-reduce style).
    
    Args:
        pdf_path: Path to PDF file
        base_url: VLM endpoint
        api_key: API key
        model: VLM model name
        window_size: Pages per batch (default 4, max 8 for InternVL)
        overlap: Overlapping pages between batches (default 2)
        max_total_pages: Maximum total pages to process
        temperature: Sampling temperature
    
    Returns:
        Combined SubcategoryLlmResult from all windows
    """
    from PyPDF2 import PdfReader
    
    # Get total pages
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    pages_to_process = min(total_pages, max_total_pages)
    
    if pages_to_process <= window_size:
        # Small document - process all at once
        page_numbers = list(range(1, pages_to_process + 1))
        return llm_classify_subcategories_vision_batch(
            pdf_path, page_numbers,
            base_url=base_url, api_key=api_key, model=model, temperature=temperature
        )
    
    # Create sliding windows
    windows = []
    start = 1
    while start <= pages_to_process:
        end = min(start + window_size - 1, pages_to_process)
        windows.append(list(range(start, end + 1)))
        
        # Move start with overlap
        start = start + window_size - overlap
        if end == pages_to_process:
            break
    
    print(f"Processing {len(windows)} windows: {windows}")
    
    # Process each window
    results = []
    for window_pages in windows:
        try:
            result = llm_classify_subcategories_vision_batch(
                pdf_path, window_pages,
                base_url=base_url, api_key=api_key, model=model, temperature=temperature
            )
            results.append(result)
        except Exception as e:
            print(f"Error processing window {window_pages}: {e}")
            continue
    
    if not results:
        raise ValueError("All vision windows failed")
    
    # Combine results (average probabilities weighted by confidence)
    combined_probs = {k: 0.0 for k in SUBCAT_TYPES}
    total_weight = 0.0
    
    for r in results:
        weight = r.confidence  # Weight by confidence
        for k, v in r.probs.items():
            combined_probs[k] += v * weight
        total_weight += weight
    
    if total_weight > 0:
        combined_probs = {k: v / total_weight for k, v in combined_probs.items()}
    
    combined_probs = normalize_subcategory_probs(combined_probs)
    
    # Best match
    best_key = max(combined_probs.items(), key=lambda x: x[1])[0]
    subcat_def = SUBCATEGORIES[best_key]
    
    # Combine rationales
    combined_rationale = " | ".join([
        f"[Pages {r.raw_json.get('pages_analyzed', '?')}] {r.rationale[:100]}"
        for r in results
    ])
    
    return SubcategoryLlmResult(
        subcategory_key=best_key,
        subcategory_name=subcat_def.name,
        parent_type=subcat_def.parent_type.value,
        confidence=combined_probs[best_key],
        rationale=combined_rationale,
        probs=combined_probs,
        raw_json={
            "sliding_window": True,
            "windows": [r.raw_json for r in results],
            "window_size": window_size,
            "overlap": overlap,
            "total_pages_analyzed": sum(len(r.raw_json.get('pages_analyzed', [])) for r in results),
        },
    )


# Alias for backward compatibility
llm_classify_subcategories_vision = llm_classify_subcategories_vision_sliding_window
