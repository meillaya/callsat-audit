"""Typed records for synthetic calls, extractions, and scores.

The schema is deliberately small and JSON-serialisable so that every run is auditable:
a score can always be traced back to the facts and transcript turns that produced it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .config import Condition, Resolution, Split

Speaker = Literal["agent", "customer"]
CueKind = Literal["resolution", "effort", "sentiment", "empathy"]
FactName = Literal["resolved", "effort", "sentiment", "empathy"]


@dataclass(slots=True)
class Cue:
    """A gold evidence cue inserted by the generator."""

    cue_id: str
    turn_id: int
    kind: CueKind
    polarity: int
    text: str


@dataclass(slots=True)
class Turn:
    turn_id: int
    speaker: Speaker
    text: str
    cues: list[Cue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "speaker": self.speaker,
            "text": self.text,
            "cues": [asdict(c) for c in self.cues],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Turn:
        return cls(
            turn_id=int(payload["turn_id"]),
            speaker=payload["speaker"],
            text=str(payload["text"]),
            cues=[Cue(**c) for c in payload.get("cues", [])],
        )


@dataclass(slots=True)
class CallLog:
    call_id: str
    pair_id: str
    condition: Condition
    split: Split
    persona: str
    issue_type: str
    turns: list[Turn]
    latent_z: float
    latent_satisfaction: int
    survey_response: int | None
    annotator_a: int
    annotator_b: int
    gold_resolution: Resolution
    gold_effort: int
    gold_sentiment: int
    gold_empathy: int
    meta: dict[str, str | float | int | bool] = field(default_factory=dict)

    def transcript(self, with_ids: bool = True) -> str:
        lines = []
        for turn in self.turns:
            if with_ids:
                lines.append(f"[{turn.turn_id}] {turn.speaker.upper()}: {turn.text}")
            else:
                lines.append(f"{turn.speaker.upper()}: {turn.text}")
        return "\n".join(lines)

    def gold_fact(self, name: FactName) -> str | int:
        return {
            "resolved": self.gold_resolution,
            "effort": self.gold_effort,
            "sentiment": self.gold_sentiment,
            "empathy": self.gold_empathy,
        }[name]

    def gold_cue_turns(self, kind: CueKind) -> set[int]:
        return {c.turn_id for t in self.turns for c in t.cues if c.kind == kind}

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["turns"] = [t.to_dict() for t in self.turns]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CallLog:
        payload = dict(payload)
        payload["turns"] = [Turn.from_dict(t) for t in payload["turns"]]
        return cls(**payload)


@dataclass(slots=True)
class Fact:
    """A typed fact with its supporting evidence.

    ``evidence_turns`` are transcript turn ids; ``quotes`` are the exact substrings
    the extractor claims support the fact.
    """

    name: FactName
    value: str | int
    confidence: float
    evidence_turns: tuple[int, ...] = ()
    quotes: tuple[str, ...] = ()
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "confidence": self.confidence,
            "evidence_turns": list(self.evidence_turns),
            "quotes": list(self.quotes),
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Fact:
        return cls(
            name=payload["name"],
            value=payload["value"],
            confidence=float(payload["confidence"]),
            evidence_turns=tuple(int(i) for i in payload.get("evidence_turns", [])),
            quotes=tuple(str(q) for q in payload.get("quotes", [])),
            note=str(payload.get("note", "")),
        )


@dataclass(slots=True)
class Extraction:
    call_id: str
    model: str
    facts: dict[FactName, Fact]
    raw_response: str = ""
    latency_s: float = 0.0
    parse_ok: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "model": self.model,
            "facts": {k: v.to_dict() for k, v in self.facts.items()},
            "raw_response": self.raw_response,
            "latency_s": self.latency_s,
            "parse_ok": self.parse_ok,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Extraction:
        return cls(
            call_id=str(payload["call_id"]),
            model=str(payload["model"]),
            facts={k: Fact.from_dict(v) for k, v in payload["facts"].items()},
            raw_response=str(payload.get("raw_response", "")),
            latency_s=float(payload.get("latency_s", 0.0)),
            parse_ok=bool(payload.get("parse_ok", True)),
        )
