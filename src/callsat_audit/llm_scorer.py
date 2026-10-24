"""Direct LLM satisfaction scoring with verbalised uncertainty and cited evidence.

This is the "end-to-end" arm of the pilot: one prompt, one score, no typed facts.  It is
kept separate from the evidence-chain arm so calibration and auditability can be compared
without pretending the two are the same measurement instrument.
"""

from __future__ import annotations

import json
import re
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np

from .schema import CallLog

DIRECT_PROMPT = """You rate caller satisfaction from a contact-centre transcript.
Return ONLY JSON, no markdown:
{"score": <1-5>, "probs": [p1,p2,p3,p4,p5], "confidence": <0-1>, "evidence_turns": [<int>], "quote": "<exact sentence>", "rationale": "<one sentence>"}

Anchors:
1 = caller ends angry or worse off; request refused or unresolved.
3 = caller neutral or politely resigned; partial fix or uncertain outcome.
5 = caller clearly relieved and grateful; request fully resolved.
Base the score on the caller's own words AND whether the request was actually resolved.
Ignore the agent's claims if the caller contradicts them.
probs must be five numbers summing to 1. evidence_turns are the transcript turns you
relied on. quote must be copied exactly from one of those turns.

TRANSCRIPT
__TRANSCRIPT__
JSON:"""


@dataclass(slots=True)
class DirectScore:
    call_id: str
    model: str
    score: int
    probs: tuple[float, float, float, float, float]
    confidence: float
    evidence_turns: tuple[int, ...]
    quote: str
    rationale: str
    raw_response: str = ""
    latency_s: float = 0.0
    parse_ok: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "call_id": self.call_id,
            "model": self.model,
            "score": self.score,
            "probs": list(self.probs),
            "confidence": self.confidence,
            "evidence_turns": list(self.evidence_turns),
            "quote": self.quote,
            "rationale": self.rationale,
            "raw_response": self.raw_response,
            "latency_s": self.latency_s,
            "parse_ok": self.parse_ok,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> DirectScore:
        probs_raw = cast(list[float], payload["probs"])
        probs = tuple(float(x) for x in probs_raw)
        turns_raw = cast(list[int], payload["evidence_turns"])
        return cls(
            call_id=str(payload["call_id"]),
            model=str(payload["model"]),
            score=int(cast(int, payload["score"])),
            probs=cast(tuple[float, float, float, float, float], (probs + (0.2,) * 5)[:5]),
            confidence=float(cast(float, payload["confidence"])),
            evidence_turns=tuple(int(x) for x in turns_raw),
            quote=str(payload.get("quote", "")),
            rationale=str(payload.get("rationale", "")),
            raw_response=str(payload.get("raw_response", "")),
            latency_s=float(cast(float, payload.get("latency_s", 0.0))),
            parse_ok=bool(payload.get("parse_ok", True)),
        )


class DirectScorer(Protocol):
    name: str

    def score(self, call: CallLog) -> DirectScore: ...


def _extract_json(text: str) -> dict[str, object]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        payload = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def parse_direct_score(raw: str, call_id: str, model: str, latency_s: float = 0.0) -> DirectScore:
    payload = _extract_json(raw)
    parse_ok = bool(payload)
    score = 3
    with suppress(TypeError, ValueError):
        score = round(float(str(payload.get("score", 3))))
    score = max(1, min(5, score))

    raw_probs = payload.get("probs", [])
    probs = np.ones(5) / 5.0
    if isinstance(raw_probs, list) and len(raw_probs) == 5:
        with suppress(TypeError, ValueError):
            candidate = np.array([float(x) for x in raw_probs], dtype=float)
            if np.all(candidate >= 0) and candidate.sum() > 0:
                probs = candidate / candidate.sum()
    if not np.all(np.isfinite(probs)):
        probs = np.ones(5) / 5.0

    try:
        confidence = float(str(payload.get("confidence", probs.max())))
    except (TypeError, ValueError):
        confidence = float(probs.max())
    confidence = max(0.0, min(1.0, confidence))

    raw_turns = payload.get("evidence_turns", [])
    turns: list[int] = []
    if isinstance(raw_turns, list):
        for item in raw_turns:
            try:
                turns.append(int(item))
            except (TypeError, ValueError):
                continue

    return DirectScore(
        call_id=call_id,
        model=model,
        score=score,
        probs=cast(tuple[float, float, float, float, float], tuple(float(x) for x in probs)),
        confidence=confidence,
        evidence_turns=tuple(turns),
        quote=str(payload.get("quote", "")),
        rationale=str(payload.get("rationale", "")),
        raw_response=raw,
        latency_s=latency_s,
        parse_ok=parse_ok,
    )


class LocalMlxDirectScorer:
    def __init__(self, model_name: str, max_tokens: int = 260, temperature: float = 0.0) -> None:
        from mlx_lm import load  # type: ignore[import-not-found]

        self.name = f"direct-mlx:{model_name}"
        loaded = load(model_name)
        self.model, self.tokenizer = loaded[0], loaded[1]
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _generate(self, prompt: str) -> str:
        from mlx_lm import generate  # type: ignore[import-not-found]
        from mlx_lm.sample_utils import make_sampler  # type: ignore[import-not-found]

        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return generate(
            self.model,
            self.tokenizer,
            prompt=formatted,
            max_tokens=self.max_tokens,
            sampler=make_sampler(temp=self.temperature),
            verbose=False,
        )

    def score(self, call: CallLog) -> DirectScore:
        start = time.perf_counter()
        raw = self._generate(DIRECT_PROMPT.replace("__TRANSCRIPT__", call.transcript()))
        return parse_direct_score(raw, call.call_id, self.name, time.perf_counter() - start)
