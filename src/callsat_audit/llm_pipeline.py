"""Runner and metrics for the direct-LLM scoring arm."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from .audit import quote_supported
from .config import CACHE_DIR
from .llm_scorer import DirectScore, DirectScorer
from .metrics import (
    binary_ece,
    brier_multiclass,
    classwise_ece,
    mae,
    nll,
    quadratic_weighted_kappa,
    risk_coverage_auc,
    spearman,
)
from .schema import CallLog


def score_all(
    calls: Iterable[CallLog],
    scorer: DirectScorer,
    condition: str,
    cache_dir: Path = CACHE_DIR / "direct_scores",
    resume: bool = True,
) -> dict[str, DirectScore]:
    calls = list(calls)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", scorer.name)
    path = cache_dir / safe / f"{condition}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, DirectScore] = {}
    if resume and path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    score = DirectScore.from_dict(json.loads(line))
                    done[score.call_id] = score
    todo = [c for c in calls if c.call_id not in done]
    if todo:
        with path.open("a", encoding="utf-8") as handle:
            for i, call in enumerate(todo, start=1):
                score = scorer.score(call)
                done[call.call_id] = score
                handle.write(json.dumps(score.to_dict(), ensure_ascii=False) + "\n")
                handle.flush()
                if i % 10 == 0:
                    print(f"  {scorer.name}: {i}/{len(todo)}")
    return {c.call_id: done[c.call_id] for c in calls}


def direct_frame(calls: Iterable[CallLog], scores: dict[str, DirectScore]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for call in calls:
        score = scores[call.call_id]
        gold_turns = {cue.turn_id for turn in call.turns for cue in turn.cues}
        cited = set(score.evidence_turns)
        cited_valid = {t for t in cited if 1 <= t <= len(call.turns)}
        turn_map = {t.turn_id: t.text for t in call.turns}
        quote_ok = (
            any(quote_supported(score.quote, turn_map[t]) for t in cited_valid)
            if score.quote
            else False
        )
        rows.append(
            {
                "call_id": call.call_id,
                "condition": call.condition,
                "split": call.split,
                "persona": call.persona,
                "latent": call.latent_satisfaction,
                "survey": call.survey_response,
                "score": score.score,
                "confidence": score.confidence,
                "model_confidence": score.confidence,
                "parse_ok": float(score.parse_ok),
                "latency_s": score.latency_s,
                **{f"p{k + 1}": float(score.probs[k]) for k in range(5)},
                "correct": int(score.score == call.latent_satisfaction),
                "within_one": int(abs(score.score - call.latent_satisfaction) <= 1),
                "any_citation": float(bool(cited_valid)),
                "cited_in_gold": float(bool(cited_valid & gold_turns)),
                "citation_precision": (
                    len(cited_valid & gold_turns) / len(cited_valid) if cited_valid else 0.0
                ),
                "quote_supported": float(quote_ok),
                "n_citations": len(cited_valid),
                "rationale": score.rationale,
            }
        )
    return pd.DataFrame(rows)


def summarize_direct(frame: pd.DataFrame, target: str = "latent") -> dict[str, float]:
    sub = frame.dropna(subset=[target])
    y = sub[target].to_numpy(dtype=float)
    probs = sub[[f"p{k}" for k in range(1, 6)]].to_numpy(dtype=float)
    score = sub["score"].to_numpy(dtype=float)
    conf = sub["confidence"].to_numpy(dtype=float)
    correct = (score == y).astype(float)
    return {
        "n": float(len(sub)),
        "qwk": quadratic_weighted_kappa(y, score),
        "mae": mae(y, score),
        "spearman": spearman(y, score),
        "exact_accuracy": float(np.mean(correct)),
        "within_one": float(np.mean(np.abs(score - y) <= 1)),
        "ece_probs": classwise_ece(y, probs),
        "brier_probs": brier_multiclass(y, probs),
        "nll_probs": nll(y, probs),
        "ece_confidence": binary_ece(conf, correct),
        "mean_confidence": float(np.mean(conf)),
        "large_error_rate": float(np.mean(np.abs(score - y) > 1)),
        "risk_coverage_auc": risk_coverage_auc(y, score, conf, 1.0),
        "parse_rate": float(sub["parse_ok"].mean()),
        "any_citation_rate": float(sub["any_citation"].mean()),
        "cited_in_gold_rate": float(sub["cited_in_gold"].mean()),
        "citation_precision": float(sub["citation_precision"].mean()),
        "quote_support_rate": float(sub["quote_supported"].mean()),
        "mean_latency_s": float(sub["latency_s"].mean()),
    }


def direct_metrics(frames: dict[str, pd.DataFrame], target: str = "latent") -> pd.DataFrame:
    rows = []
    for condition, frame in frames.items():
        metrics = summarize_direct(frame, target=target)
        rows.append({**metrics, "condition": condition})
    return pd.DataFrame(rows).set_index("condition")
