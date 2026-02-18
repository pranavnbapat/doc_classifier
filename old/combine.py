# docint/fusion/combine.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

from docint.llm.classify import DOC_TYPES


@dataclass
class FusionResult:
    label: str
    confidence: float
    probs: Dict[str, float]
    meta: Dict[str, Any]


def _normalise(probs: Dict[str, float]) -> Dict[str, float]:
    out = {k: float(probs.get(k, 0.0)) for k in DOC_TYPES}
    s = sum(out.values())
    if s <= 0:
        return {k: 1.0 / len(DOC_TYPES) for k in DOC_TYPES}
    return {k: v / s for k, v in out.items()}


def fuse_probs(
    heuristic_probs: Dict[str, float],
    llm_probs: Dict[str, float],
    *,
    alpha: float = 0.6,
) -> FusionResult:
    """
    Weighted mixture of heuristic + LLM probabilities.
    alpha: weight for LLM (0..1)

    Returns:
        FusionResult (label/confidence + full probability vector)
    """
    h = _normalise(heuristic_probs)
    l = _normalise(llm_probs)

    fused = {}
    for k in DOC_TYPES:
        fused[k] = alpha * l.get(k, 0.0) + (1.0 - alpha) * h.get(k, 0.0)

    fused = _normalise(fused)
    label, conf = max(fused.items(), key=lambda kv: kv[1])

    return FusionResult(
        label=label,
        confidence=round(conf, 3),
        probs={k: round(v, 4) for k, v in fused.items()},
        meta={"alpha": alpha, "heuristic_probs": h, "llm_probs": l},
    )
