from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx


@dataclass
class PageSenseResult:
    ok: bool
    text: str
    title: Optional[str]
    raw: Dict[str, Any]
    rationale: str


def _normalized_base_url(url: str) -> str:
    return (url or "").rstrip("/")


def extract_url_text(url: str) -> PageSenseResult:
    base_url = _normalized_base_url(os.getenv("URL_CONTENT_EXTRACTOR_BASE", ""))
    timeout = float(os.getenv("EXTRACTOR_TIMEOUT", "60"))

    if not base_url:
        return PageSenseResult(
            ok=False,
            text="",
            title=None,
            raw={},
            rationale="PageSense base URL is not configured",
        )

    with httpx.Client(timeout=timeout) as client:
        try:
            response = client.post(
                f"{base_url}/api/extract",
                headers={"Content-Type": "application/json"},
                json={"url": url},
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            response = exc.response
            try:
                payload = response.json()
            except Exception:
                payload = {}
            detail = (
                payload.get("detail")
                or payload.get("error")
                or payload.get("message")
                or response.text.strip()
                or f"PageSense returned HTTP {response.status_code}"
            )
            return PageSenseResult(
                ok=False,
                text="",
                title=None,
                raw=payload if isinstance(payload, dict) else {},
                rationale=f"PageSense rejected the URL: {detail}",
            )
        except httpx.HTTPError as exc:
            return PageSenseResult(
                ok=False,
                text="",
                title=None,
                raw={},
                rationale=f"PageSense request failed: {str(exc)}",
            )

    text = (
        payload.get("text")
        or payload.get("content")
        or payload.get("raw_text")
        or payload.get("result", {}).get("text")
        or ""
    )
    text = text.strip()
    title = (
        payload.get("title")
        or payload.get("meta", {}).get("title")
        or payload.get("result", {}).get("title")
    )

    return PageSenseResult(
        ok=bool(text),
        text=text,
        title=title.strip() if isinstance(title, str) and title.strip() else None,
        raw=payload,
        rationale="Extracted raw readable text via PageSense" if text else "PageSense returned no usable text",
    )
