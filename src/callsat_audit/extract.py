"""Fact extractors: a strict offline lexicon baseline and an optional LLM backend.

The offline extractor is deliberately brittle: it is the *baseline* that an LLM is meant
to improve on, and it makes lexical-shift failures easy to localise.  The LLM backend is
kept behind a small protocol so the same audit code runs on either.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Protocol

from .schema import CallLog, Extraction, Fact, FactName

ResolutionValue = str


class Extractor(Protocol):
    name: str

    def extract(self, call: CallLog) -> Extraction: ...


@dataclass(frozen=True)
class Pattern:
    regex: re.Pattern[str]
    value: str | int
    weight: float


def _compile(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


STRICT_RESOLUTION = (
    Pattern(_compile(r"processed the refund"), "resolved", 1.0),
    Pattern(_compile(r"reversed the charge"), "resolved", 1.0),
    Pattern(_compile(r"applied the credit"), "resolved", 1.0),
    Pattern(_compile(r"sent you a confirmation"), "resolved", 0.9),
    Pattern(_compile(r"updated the account"), "resolved", 0.9),
    Pattern(_compile(r"escalated this"), "partial", 1.0),
    Pattern(_compile(r"submitted the request"), "partial", 1.0),
    Pattern(_compile(r"opened a ticket"), "partial", 0.9),
    Pattern(_compile(r"needs a supervisor"), "partial", 0.9),
    Pattern(_compile(r"nothing more i can do"), "unresolved", 1.0),
    Pattern(_compile(r"not eligible for the refund"), "unresolved", 1.0),
    Pattern(_compile(r"can't override"), "unresolved", 1.0),
)

BROAD_RESOLUTION = (
    *STRICT_RESOLUTION,
    Pattern(_compile(r"refund has been issued"), "resolved", 1.0),
    Pattern(_compile(r"charge has been backed out"), "resolved", 1.0),
    Pattern(_compile(r"credit is now on your account"), "resolved", 1.0),
    Pattern(_compile(r"with the billing team"), "partial", 0.8),
    Pattern(_compile(r"supervisor still has to sign"), "partial", 0.9),
    Pattern(_compile(r"ticket on your record"), "partial", 0.9),
    Pattern(_compile(r"hands are tied"), "unresolved", 1.0),
    Pattern(_compile(r"policy doesn't cover"), "unresolved", 1.0),
    Pattern(_compile(r"able to change from here"), "unresolved", 0.8),
)

STRICT_EFFORT = (
    Pattern(_compile(r"brief hold"), 1, 1.0),
    Pattern(_compile(r"transfer you"), 1, 1.0),
    Pattern(_compile(r"call back tomorrow"), 1, 1.0),
    Pattern(_compile(r"repeat that"), 1, 0.9),
)

BROAD_EFFORT = (
    *STRICT_EFFORT,
    Pattern(_compile(r"hang on for a moment"), 1, 1.0),
    Pattern(_compile(r"put you through"), 1, 1.0),
    Pattern(_compile(r"ring us again"), 1, 1.0),
)

STRICT_SENTIMENT = (
    Pattern(_compile(r"huge relief"), 2, 1.0),
    Pattern(_compile(r"that helps"), 1, 0.8),
    Pattern(_compile(r"wait and see"), 0, 0.7),
    Pattern(_compile(r"not really happy"), -1, 1.0),
    Pattern(_compile(r"this is ridiculous"), -2, 1.0),
    Pattern(_compile(r"still annoyed"), -1, 1.0),
    Pattern(_compile(r"far too long"), -1, 0.8),
    Pattern(_compile(r"that's fine"), 1, 0.7),
)

BROAD_SENTIMENT = (
    *STRICT_SENTIMENT,
    Pattern(_compile(r"weight off my mind"), 2, 1.0),
    Pattern(_compile(r"that's better"), 1, 0.8),
    Pattern(_compile(r"see what happens"), 0, 0.7),
    Pattern(_compile(r"not a great outcome"), -1, 1.0),
    Pattern(_compile(r"beyond frustrating"), -2, 1.0),
    Pattern(_compile(r"pretty poor"), -1, 0.8),
    Pattern(_compile(r"shouldn't have taken"), -1, 0.8),
)

STRICT_EMPATHY = (
    Pattern(_compile(r"completely understand"), 1, 1.0),
    Pattern(_compile(r"i'm sorry you've"), 1, 0.9),
    Pattern(_compile(r"appreciate your patience"), 1, 0.9),
)

BROAD_EMPATHY = (
    *STRICT_EMPATHY,
    Pattern(_compile(r"can see why this would"), 1, 0.9),
    Pattern(_compile(r"real hassle"), 1, 0.9),
    Pattern(_compile(r"bearing with us"), 1, 0.8),
    Pattern(_compile(r"i understand"), 1, 0.7),
)

RESOLUTION_PRIORITY = {"resolved": 2, "partial": 1, "unresolved": 0}


@dataclass
class LexiconExtractor:
    """Rule-based extractor mirroring call-eval's offline backend.

    ``broad`` toggles a lexical-robustness intervention: paraphrases are added to the
    lexicon.  The notebook evaluates whether the extra recall is worth the noise.
    """

    broad: bool = False
    name: str = "offline-strict"

    def __post_init__(self) -> None:
        if self.broad:
            self.name = "offline-broad"

    def _resolution(self, call: CallLog) -> Fact:
        patterns = BROAD_RESOLUTION if self.broad else STRICT_RESOLUTION
        matches: list[tuple[int, str, str, float]] = []
        for turn in call.turns:
            if turn.speaker != "agent":
                continue
            for pattern in patterns:
                found = pattern.regex.search(turn.text)
                if found is not None:
                    matches.append(
                        (turn.turn_id, str(pattern.value), found.group(0), pattern.weight)
                    )
        if not matches:
            return Fact(
                name="resolved",
                value="partial",
                confidence=0.20,
                note="no resolution cue found; weak prior",
            )
        best = max(matches, key=lambda m: (m[0], RESOLUTION_PRIORITY.get(m[1], 0), m[3]))
        return Fact(
            name="resolved",
            value=best[1],
            confidence=0.90,
            evidence_turns=(best[0],),
            quotes=(best[2],),
        )

    def _effort(self, call: CallLog) -> Fact:
        patterns = BROAD_EFFORT if self.broad else STRICT_EFFORT
        turns: list[int] = []
        quotes: list[str] = []
        for turn in call.turns:
            if turn.speaker != "agent":
                continue
            for pattern in patterns:
                found = pattern.regex.search(turn.text)
                if found is not None:
                    turns.append(turn.turn_id)
                    quotes.append(found.group(0))
                    break
        value = min(3, len(turns))
        return Fact(
            name="effort",
            value=value,
            confidence=0.85 if turns else 0.35,
            evidence_turns=tuple(turns),
            quotes=tuple(quotes),
        )

    def _sentiment(self, call: CallLog) -> Fact:
        patterns = BROAD_SENTIMENT if self.broad else STRICT_SENTIMENT
        matches: list[tuple[int, int, str, float]] = []
        for turn in call.turns:
            if turn.speaker != "customer":
                continue
            for pattern in patterns:
                found = pattern.regex.search(turn.text)
                if found is not None:
                    matches.append(
                        (turn.turn_id, int(pattern.value), found.group(0), pattern.weight)
                    )
        if not matches:
            return Fact(
                name="sentiment",
                value=0,
                confidence=0.30,
                note="no explicit sentiment cue",
            )
        best = max(matches, key=lambda m: (abs(m[1]), m[3], m[0]))
        return Fact(
            name="sentiment",
            value=best[1],
            confidence=0.80,
            evidence_turns=(best[0],),
            quotes=(best[2],),
        )

    def _empathy(self, call: CallLog) -> Fact:
        patterns = BROAD_EMPATHY if self.broad else STRICT_EMPATHY
        for turn in call.turns:
            if turn.speaker != "agent":
                continue
            for pattern in patterns:
                found = pattern.regex.search(turn.text)
                if found is not None:
                    return Fact(
                        name="empathy",
                        value=1,
                        confidence=0.85,
                        evidence_turns=(turn.turn_id,),
                        quotes=(found.group(0),),
                    )
        return Fact(name="empathy", value=0, confidence=0.55)

    def extract(self, call: CallLog) -> Extraction:
        start = time.perf_counter()
        facts: dict[FactName, Fact] = {
            "resolved": self._resolution(call),
            "effort": self._effort(call),
            "sentiment": self._sentiment(call),
            "empathy": self._empathy(call),
        }
        return Extraction(
            call_id=call.call_id,
            model=self.name,
            facts=facts,
            latency_s=time.perf_counter() - start,
        )


LLM_INSTRUCTIONS = """You are auditing a customer-service call transcript.
Return ONLY a JSON object with this schema:
{
  "resolved": {"value": "resolved|partial|unresolved", "evidence_turn": <int>, "quote": "<exact span>", "confidence": <0-1>},
  "effort": {"value": <0-3>, "evidence_turns": [<int>], "confidence": <0-1>},
  "sentiment": {"value": <-2..2>, "evidence_turn": <int>, "quote": "<exact span>", "confidence": <0-1>},
  "empathy": {"value": <0|1>, "evidence_turn": <int>, "quote": "<exact span>", "confidence": <0-1>}
}
Rules:
- Use only the transcript. Cite the turn id that contains the evidence.
- "resolved" is whether the caller's request was fully resolved by the end.
- "effort" counts extra caller work: holds, transfers, repeats, callbacks (0-3).
- "sentiment" is the caller's expressed feeling at the end (-2 very negative, +2 very positive).
- "empathy" is 1 only if the agent explicitly acknowledges the caller's difficulty.
Do not include markdown fences or commentary."""


def build_llm_prompt(call: CallLog) -> str:
    return f"{LLM_INSTRUCTIONS}\n\nTRANSCRIPT\n{call.transcript()}\n\nJSON:"


def _evidence_turns(value: object) -> tuple[int, ...]:
    if not isinstance(value, list):
        return ()
    turns: list[int] = []
    for item in value:
        try:
            turns.append(int(str(item)))
        except (TypeError, ValueError):
            continue
    return tuple(turns)


def parse_llm_json(raw: str, call_id: str, model: str, latency_s: float) -> Extraction:
    """Parse a (possibly noisy) JSON response into typed facts."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    parse_ok = start >= 0 and end > start
    payload: dict[str, object] = {}
    if parse_ok:
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            parse_ok = False
    if not isinstance(payload, dict):
        payload = {}

    def subset(name: str) -> dict[str, object]:
        value = payload.get(name, {})
        return value if isinstance(value, dict) else {}

    res = subset("resolved")
    eff = subset("effort")
    sen = subset("sentiment")
    emp = subset("empathy")

    def as_int(value: object, default: int) -> int:
        try:
            return int(float(str(value)))
        except (TypeError, ValueError):
            return default

    def as_float(value: object, default: float) -> float:
        try:
            return float(str(value))
        except (TypeError, ValueError):
            return default

    resolution = str(res.get("value", "partial")).lower()
    if resolution not in {"resolved", "partial", "unresolved"}:
        resolution = "partial"
    sentiment = max(-2, min(2, as_int(sen.get("value"), 0)))
    effort = max(0, min(3, as_int(eff.get("value"), 0)))
    empathy = 1 if as_int(emp.get("value"), 0) >= 1 else 0

    facts: dict[FactName, Fact] = {
        "resolved": Fact(
            name="resolved",
            value=resolution,
            confidence=as_float(res.get("confidence"), 0.5),
            evidence_turns=(as_int(res.get("evidence_turn"), -1),)
            if res.get("evidence_turn") is not None
            else (),
            quotes=(str(res.get("quote", "")),) if res.get("quote") else (),
        ),
        "effort": Fact(
            name="effort",
            value=effort,
            confidence=as_float(eff.get("confidence"), 0.5),
            evidence_turns=_evidence_turns(eff.get("evidence_turns")),
            quotes=(str(eff.get("quote", "")),) if eff.get("quote") else (),
        ),
        "sentiment": Fact(
            name="sentiment",
            value=sentiment,
            confidence=as_float(sen.get("confidence"), 0.5),
            evidence_turns=(as_int(sen.get("evidence_turn"), -1),)
            if sen.get("evidence_turn") is not None
            else (),
            quotes=(str(sen.get("quote", "")),) if sen.get("quote") else (),
        ),
        "empathy": Fact(
            name="empathy",
            value=empathy,
            confidence=as_float(emp.get("confidence"), 0.5),
            evidence_turns=(as_int(emp.get("evidence_turn"), -1),)
            if emp.get("evidence_turn") is not None
            else (),
            quotes=(str(emp.get("quote", "")),) if emp.get("quote") else (),
        ),
    }
    return Extraction(
        call_id=call_id,
        model=model,
        facts=facts,
        raw_response=raw,
        latency_s=latency_s,
        parse_ok=parse_ok,
    )


class LocalMlxExtractor:
    """Small open-model pilot backend (mlx-lm), with the same audit interface.

    Lazy import keeps the core package usable without the optional extra.
    """

    def __init__(self, model_name: str, max_tokens: int = 320, temperature: float = 0.0) -> None:
        from mlx_lm import load  # type: ignore[import-not-found]

        self.name = f"mlx:{model_name}"
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
        sampler = make_sampler(temp=self.temperature)
        return generate(
            self.model,
            self.tokenizer,
            prompt=formatted,
            max_tokens=self.max_tokens,
            sampler=sampler,
            verbose=False,
        )

    def extract(self, call: CallLog) -> Extraction:
        start = time.perf_counter()
        raw = self._generate(build_llm_prompt(call))
        return parse_llm_json(raw, call.call_id, self.name, time.perf_counter() - start)
