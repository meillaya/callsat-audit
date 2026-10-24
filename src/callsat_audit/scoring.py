"""Deterministic fact-to-score layer.

Kept deliberately transparent: the score is a weighted sum of four auditable facts, so
any change in the final score can be decomposed into a fact change and a weight change.
"""

from __future__ import annotations

import numpy as np

from .schema import Extraction

RESOLUTION_VALUE = {"resolved": 1.0, "partial": 0.5, "unresolved": 0.0}
WEIGHTS = {
    "resolved": 0.55,
    "effort": 0.20,
    "sentiment": 0.15,
    "empathy": 0.10,
}


def fact_components(extraction: Extraction) -> dict[str, float]:
    res = RESOLUTION_VALUE.get(str(extraction.facts["resolved"].value), 0.5)
    effort = max(0, min(3, int(extraction.facts["effort"].value)))
    sentiment = max(-2, min(2, int(extraction.facts["sentiment"].value)))
    empathy = 1.0 if int(extraction.facts["empathy"].value) >= 1 else 0.0
    return {
        "resolved": res,
        "effort": 1.0 - effort / 3.0,
        "sentiment": (sentiment + 2.0) / 4.0,
        "empathy": empathy,
    }


def raw_score(extraction: Extraction) -> float:
    components = fact_components(extraction)
    index = sum(WEIGHTS[name] * components[name] for name in WEIGHTS)
    return float(1.0 + 4.0 * np.clip(index, 0.0, 1.0))


def raw_label(score: float) -> int:
    return max(1, min(5, round(score)))


def fact_confidence(extraction: Extraction) -> float:
    return float(np.mean([fact.confidence for fact in extraction.facts.values()]))


def decomposition(extraction: Extraction) -> dict[str, float]:
    """Contribution of each fact to the raw score (for audit traces)."""
    components = fact_components(extraction)
    parts = {name: 4.0 * WEIGHTS[name] * components[name] for name in WEIGHTS}
    parts["base"] = 1.0
    return parts
