"""Evidence-chain audit: span validity, fact recovery, and paired consistency."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .schema import CallLog, CueKind, Extraction, FactName

FACT_NAMES: tuple[FactName, ...] = ("resolved", "effort", "sentiment", "empathy")
CUE_KIND: dict[FactName, CueKind] = {
    "resolved": "resolution",
    "effort": "effort",
    "sentiment": "sentiment",
    "empathy": "empathy",
}
_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def quote_supported(quote: str, turn_text: str, min_overlap: float = 0.80) -> bool:
    if not quote.strip():
        return False
    if quote.lower() in turn_text.lower():
        return True
    q = _tokens(quote)
    t = _tokens(turn_text)
    if not q:
        return False
    # Longest common subsequence is overkill; token recall is enough for an audit flag.
    overlap = sum(1 for tok in q if tok in t) / len(q)
    return overlap >= min_overlap


@dataclass(slots=True)
class EvidenceAudit:
    call_id: str
    fact_correct: dict[FactName, bool]
    fact_error_count: int
    span_exists: bool
    turn_valid: bool
    gold_turn_hit: float
    quote_supported: bool
    cited_turns: tuple[int, ...]
    mean_fact_confidence: float

    def to_row(self) -> dict[str, float | int | bool | str]:
        return {
            "call_id": self.call_id,
            "fact_error_count": self.fact_error_count,
            **{f"correct_{k}": v for k, v in self.fact_correct.items()},
            "span_exists": self.span_exists,
            "turn_valid": self.turn_valid,
            "gold_turn_hit": self.gold_turn_hit,
            "quote_supported": self.quote_supported,
            "mean_fact_confidence": self.mean_fact_confidence,
        }


def audit_extraction(call: CallLog, extraction: Extraction) -> EvidenceAudit:
    """Check whether the extraction's claims are grounded in the transcript."""
    turn_map = {t.turn_id: t for t in call.turns}
    fact_correct = {
        name: bool(extraction.facts[name].value == call.gold_fact(name)) for name in FACT_NAMES
    }
    cited = tuple(
        turn_id for name in FACT_NAMES for turn_id in extraction.facts[name].evidence_turns
    )
    valid_turns = [turn_id for turn_id in cited if turn_id in turn_map]
    quotes = [
        quote for name in FACT_NAMES for quote in extraction.facts[name].quotes if quote.strip()
    ]
    span_exists = bool(valid_turns) and any(
        quote_supported(quote, turn_map[turn_id].text)
        for quote in quotes
        for turn_id in valid_turns
    )
    quotes_ok = bool(quotes) and all(
        any(quote_supported(quote, turn.text) for turn in call.turns) for quote in quotes
    )
    gold_hits = 0
    gold_total = 0
    for name in FACT_NAMES:
        gold_turns = call.gold_cue_turns(CUE_KIND[name])
        if not gold_turns:
            continue
        gold_total += 1
        if set(extraction.facts[name].evidence_turns) & gold_turns:
            gold_hits += 1
    return EvidenceAudit(
        call_id=call.call_id,
        fact_correct=fact_correct,
        fact_error_count=sum(not v for v in fact_correct.values()),
        span_exists=span_exists,
        turn_valid=bool(cited) and len(valid_turns) == len(cited),
        gold_turn_hit=(gold_hits / gold_total) if gold_total else 0.0,
        quote_supported=quotes_ok,
        cited_turns=cited,
        mean_fact_confidence=float(
            np.mean([extraction.facts[name].confidence for name in FACT_NAMES])
        ),
    )


def audit_flags(call: CallLog, extraction: Extraction) -> dict[str, float]:
    """Cheap, label-free flags a service manager could actually compute at runtime."""
    resolved = str(extraction.facts["resolved"].value)
    sentiment = int(extraction.facts["sentiment"].value)
    missing = {name: not extraction.facts[name].evidence_turns for name in FACT_NAMES}
    conflict = (resolved == "resolved" and sentiment < 0) or (
        resolved == "unresolved" and sentiment > 0
    )
    return {
        "missing_resolution_evidence": float(missing["resolved"]),
        "missing_sentiment_evidence": float(missing["sentiment"]),
        "missing_any_evidence": float(any(missing.values())),
        "sentiment_resolution_conflict": float(conflict),
        "low_mean_confidence": float(
            np.mean([extraction.facts[name].confidence for name in FACT_NAMES]) < 0.55
        ),
        "implicit_resolution": float(bool(call.meta.get("implicit_resolution", False))),
    }


def paired_consistency(
    source_call: CallLog,
    shifted_call: CallLog,
    source_ex: Extraction,
    shifted_ex: Extraction,
    source_score: float,
    shifted_score: float,
    tolerance: float = 0.5,
) -> dict[str, float | int | str]:
    """Compare a call with its label-preserving shifted twin."""
    fact_changes = sum(
        source_ex.facts[name].value != shifted_ex.facts[name].value for name in FACT_NAMES
    )
    source_turns = set(source_ex.facts["resolved"].evidence_turns) | set(
        source_ex.facts["sentiment"].evidence_turns
    )
    shifted_turns = set(shifted_ex.facts["resolved"].evidence_turns) | set(
        shifted_ex.facts["sentiment"].evidence_turns
    )
    union = source_turns | shifted_turns
    jaccard = len(source_turns & shifted_turns) / len(union) if union else 1.0
    return {
        "call_id": source_call.call_id,
        "fact_changes": fact_changes,
        "score_delta": abs(source_score - shifted_score),
        "score_shift_beyond_tolerance": float(abs(source_score - shifted_score) > tolerance),
        "label_changed": int(round(source_score) != round(shifted_score)),
        "evidence_turn_jaccard": jaccard,
        "same_latent": int(source_call.latent_satisfaction == shifted_call.latent_satisfaction),
    }


def summarize_audit(rows: list[dict[str, float | int | bool | str]]) -> dict[str, float]:
    """Aggregate audit rows into the four evidence-chain properties."""
    n = len(rows)
    if n == 0:
        return {}
    return {
        "n": float(n),
        "mean_fact_error_count": float(np.mean([float(r["fact_error_count"]) for r in rows])),
        "resolution_accuracy": float(np.mean([float(r["correct_resolved"]) for r in rows])),
        "sentiment_accuracy": float(np.mean([float(r["correct_sentiment"]) for r in rows])),
        "evidence_turn_hit_rate": float(np.mean([float(r["gold_turn_hit"]) for r in rows])),
        "quote_support_rate": float(np.mean([float(r["quote_supported"]) for r in rows])),
        "missing_evidence_rate": float(np.mean([float(r["missing_any_evidence"]) for r in rows]))
        if "missing_any_evidence" in rows[0]
        else 0.0,
    }
