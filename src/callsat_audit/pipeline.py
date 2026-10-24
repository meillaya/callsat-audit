"""Experiment pipeline: extraction -> scoring -> calibration -> audit -> metrics."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import KFold

from .audit import audit_extraction, audit_flags, paired_consistency
from .calibration import (
    IsotonicCalibrator,
    OrdinalCalibrator,
    fit_temperature_from_probs,
)
from .config import CACHE_DIR, SEED
from .extract import Extractor
from .metrics import (
    aurc,
    brier_multiclass,
    calibration_slope,
    classwise_ece,
    confusion,
    mae,
    nll,
    quadratic_weighted_kappa,
    risk_coverage_auc,
    spearman,
)
from .schema import CallLog, Extraction
from .scoring import fact_confidence, raw_label, raw_score

FloatArray = NDArray[np.float64]


@dataclass
class CalibrationBundle:
    ordinal: OrdinalCalibrator
    isotonic: IsotonicCalibrator
    temperature: float
    qhat: float
    alpha: float = 0.10
    target_prior: FloatArray = field(default_factory=lambda: np.ones(5) / 5)

    def predict_proba(self, raw: FloatArray) -> FloatArray:
        return self.ordinal.predict_proba(raw)

    def predict_expected(self, raw: FloatArray) -> FloatArray:
        return self.ordinal.predict_expected(raw)

    def conformal_sets(self, raw: FloatArray) -> list[list[int]]:
        probs = self.predict_proba(raw)
        sets: list[list[int]] = []
        for row in probs:
            members = [int(k + 1) for k, p in enumerate(row) if 1.0 - p <= self.qhat]
            sets.append(members or [int(row.argmax() + 1)])
        return sets

    def to_dict(self) -> dict[str, float | list[float]]:
        return {
            "beta": self.ordinal.beta,
            "cutpoints": [float(x) for x in self.ordinal.cutpoints],
            "x_mean": self.ordinal.x_mean,
            "x_std": self.ordinal.x_std,
            "temperature": self.temperature,
            "qhat": self.qhat,
            "alpha": self.alpha,
        }


def _calibration_pairs(
    calls: Iterable[CallLog], extractions: dict[str, Extraction]
) -> tuple[FloatArray, NDArray[np.int64]]:
    raw: list[float] = []
    survey: list[int] = []
    for call in calls:
        if call.split not in {"train", "dev"} or call.survey_response is None:
            continue
        raw.append(raw_score(extractions[call.call_id]))
        survey.append(call.survey_response)
    return np.asarray(raw, dtype=float), np.asarray(survey, dtype=int)


def fit_calibration(
    calls: Iterable[CallLog],
    extractions: dict[str, Extraction],
    seed: int = SEED,
    n_folds: int = 5,
    alpha: float = 0.10,
) -> CalibrationBundle:
    """Fit survey calibration on source train+dev; temperature and qhat use OOF probs."""
    calls = list(calls)
    raw, y = _calibration_pairs(calls, extractions)
    if len(raw) < 20:
        raise ValueError("too few survey-labelled source calls to fit calibration")
    ordinal = OrdinalCalibrator().fit(raw, y)

    oof = np.zeros((len(raw), 5), dtype=float)
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train_idx, valid_idx in kfold.split(raw):
        fold = OrdinalCalibrator().fit(raw[train_idx], y[train_idx])
        oof[valid_idx] = fold.predict_proba(raw[valid_idx])
    temperature = fit_temperature_from_probs(oof, y)
    ordinal.temperature = temperature
    isotonic = IsotonicCalibrator().fit(raw, y)

    n = len(y)
    level = min(1.0, float(np.ceil((n + 1) * (1 - alpha)) / n))
    scores = 1.0 - oof[np.arange(n), y - 1]
    qhat = float(np.quantile(scores, level))

    target_prior = np.array([(y == k).mean() for k in range(1, 6)], dtype=float)
    return CalibrationBundle(
        ordinal=ordinal,
        isotonic=isotonic,
        temperature=temperature,
        qhat=qhat,
        alpha=alpha,
        target_prior=target_prior,
    )


def _probs_to_columns(probs: FloatArray) -> dict[str, float]:
    return {f"p{k + 1}": float(probs[k]) for k in range(len(probs))}


def build_frame(
    calls: Iterable[CallLog],
    extractions: dict[str, Extraction],
    bundle: CalibrationBundle,
) -> pd.DataFrame:
    """One row per call, with scores, audit flags, and every intermediate fact."""
    rows: list[dict[str, object]] = []
    for call in calls:
        ex = extractions[call.call_id]
        raw = raw_score(ex)
        probs = bundle.predict_proba(np.array([raw]))[0]
        expected = float(probs @ np.arange(1, 6))
        pred = int(probs.argmax() + 1)
        iso_expected = float(bundle.isotonic.predict_expected(np.array([raw]))[0])
        audit = audit_extraction(call, ex)
        flags = audit_flags(call, ex)
        row: dict[str, object] = {
            "call_id": call.call_id,
            "pair_id": call.pair_id,
            "condition": call.condition,
            "split": call.split,
            "persona": call.persona,
            "issue": call.issue_type,
            "latent": call.latent_satisfaction,
            "survey": call.survey_response,
            "annotator_a": call.annotator_a,
            "annotator_b": call.annotator_b,
            "raw_score": raw,
            "raw_label": raw_label(raw),
            "expected": expected,
            "pred_label": pred,
            "confidence": float(probs.max()),
            "iso_expected": iso_expected,
            "iso_label": int(np.clip(round(iso_expected), 1, 5)),
            "fact_confidence": fact_confidence(ex),
            "fact_error_count": audit.fact_error_count,
            "gold_turn_hit": audit.gold_turn_hit,
            "quote_supported": float(audit.quote_supported),
            "span_exists": float(audit.span_exists),
            "mean_fact_confidence": audit.mean_fact_confidence,
            "gold_resolution": call.gold_resolution,
            "gold_effort": call.gold_effort,
            "gold_sentiment": call.gold_sentiment,
            "gold_empathy": call.gold_empathy,
            **flags,
            **_probs_to_columns(probs),
            **{f"correct_{k}": v for k, v in audit.fact_correct.items()},
            **{f"fact_{k}": v for k, v in _fact_values(ex).items()},
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _fact_values(extraction: Extraction) -> dict[str, object]:
    return {name: extraction.facts[name].value for name in extraction.facts}


def probs_matrix(frame: pd.DataFrame) -> FloatArray:
    cols = [f"p{k}" for k in range(1, 6)]
    return frame[cols].to_numpy(dtype=float)


def summarize_frame(
    frame: pd.DataFrame,
    target: str = "latent",
    large_error_threshold: float = 1.0,
) -> dict[str, float]:
    sub = frame.dropna(subset=[target])
    y = sub[target].to_numpy(dtype=float)
    probs = probs_matrix(sub)
    raw = sub["raw_score"].to_numpy(dtype=float)
    iso = sub["iso_expected"].to_numpy(dtype=float)
    pred = sub["pred_label"].to_numpy(dtype=int)
    slope, intercept = calibration_slope(y, sub["expected"].to_numpy(dtype=float))
    return {
        "n": float(len(sub)),
        "qwk_raw": quadratic_weighted_kappa(y, sub["raw_label"]),
        "qwk_ordinal": quadratic_weighted_kappa(y, pred),
        "qwk_isotonic": quadratic_weighted_kappa(y, sub["iso_label"]),
        "mae_raw": mae(y, raw),
        "mae_ordinal": mae(y, sub["expected"]),
        "mae_isotonic": mae(y, iso),
        "spearman_raw": spearman(y, raw),
        "ece": classwise_ece(y, probs),
        "brier": brier_multiclass(y, probs),
        "nll": nll(y, probs),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "mean_confidence": float(sub["confidence"].mean()),
        "accuracy": float((pred == y).mean()),
        "large_error_rate": float((np.abs(sub["expected"] - y) > large_error_threshold).mean()),
        "risk_coverage_auc": risk_coverage_auc(
            y, sub["expected"], sub["confidence"], large_error_threshold
        ),
        "aurc": aurc(y, sub["expected"], sub["confidence"]),
    }


def condition_metrics(
    frames: dict[str, pd.DataFrame],
    target: str = "latent",
) -> pd.DataFrame:
    rows = []
    for condition, frame in frames.items():
        metrics = summarize_frame(frame, target=target)
        rows.append({**metrics, "condition": condition})
    return pd.DataFrame(rows).set_index("condition")


def paired_consistency_frame(
    source_calls: Iterable[CallLog],
    shifted_calls: Iterable[CallLog],
    source_ex: dict[str, Extraction],
    shifted_ex: dict[str, Extraction],
    bundle: CalibrationBundle,
) -> pd.DataFrame:
    shifted_by_id = {c.call_id: c for c in shifted_calls}
    rows = []
    for call in source_calls:
        other = shifted_by_id.get(call.call_id)
        if other is None:
            continue
        s_raw = raw_score(source_ex[call.call_id])
        t_raw = raw_score(shifted_ex[other.call_id])
        s_prob = bundle.predict_proba(np.array([s_raw]))[0]
        t_prob = bundle.predict_proba(np.array([t_raw]))[0]
        row = paired_consistency(
            call,
            other,
            source_ex[call.call_id],
            shifted_ex[other.call_id],
            s_raw,
            t_raw,
            tolerance=0.5,
        )
        row["source_expected"] = float(s_prob @ np.arange(1, 6))
        row["shifted_expected"] = float(t_prob @ np.arange(1, 6))
        row["expected_delta"] = abs(row["source_expected"] - row["shifted_expected"])
        row["source_label"] = int(s_prob.argmax() + 1)
        row["shifted_label"] = int(t_prob.argmax() + 1)
        row["label_changed"] = int(row["source_label"] != row["shifted_label"])
        row["source_confidence"] = float(s_prob.max())
        row["shifted_confidence"] = float(t_prob.max())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_consistency(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "n": float(len(frame)),
        "mean_fact_changes": float(frame["fact_changes"].mean()),
        "any_fact_change_rate": float((frame["fact_changes"] > 0).mean()),
        "score_shift_rate": float((frame["score_delta"] > 0.5).mean()),
        "expected_shift_rate": float((frame["expected_delta"] > 0.5).mean()),
        "label_change_rate": float(frame["label_changed"].mean()),
        "mean_evidence_jaccard": float(frame["evidence_turn_jaccard"].mean()),
    }


def extract_all(
    calls: Iterable[CallLog],
    extractor: Extractor,
    cache_dir: Path = CACHE_DIR / "extractions",
    condition: str = "source",
    resume: bool = True,
) -> dict[str, Extraction]:
    """Extract every call, with a JSONL cache so LLM runs are resumable."""
    calls = list(calls)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", extractor.name)
    path = cache_dir / safe / f"{condition}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, Extraction] = {}
    if resume and path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    ex = Extraction.from_dict(json.loads(line))
                    done[ex.call_id] = ex
    todo = [c for c in calls if c.call_id not in done]
    if todo:
        with path.open("a", encoding="utf-8") as handle:
            for i, call in enumerate(todo, start=1):
                ex = extractor.extract(call)
                done[call.call_id] = ex
                handle.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
                handle.flush()
                if i % 20 == 0:
                    print(f"  {extractor.name}: {i}/{len(todo)}")
    return {c.call_id: done[c.call_id] for c in calls}


def confusion_frame(frame: pd.DataFrame, target: str = "latent") -> NDArray[np.int64]:
    sub = frame.dropna(subset=[target])
    return confusion(sub[target], sub["pred_label"])
