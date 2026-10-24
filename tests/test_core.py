"""Tests for metrics and audit invariants."""

from __future__ import annotations

import numpy as np

from callsat_audit.audit import quote_supported
from callsat_audit.calibration import IsotonicCalibrator, OrdinalCalibrator
from callsat_audit.generate import apply_shift, generate_source
from callsat_audit.metrics import (
    binary_ece,
    brier_multiclass,
    classwise_ece,
    quadratic_weighted_kappa,
)


def test_qwk_perfect() -> None:
    assert quadratic_weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == 1.0


def test_ece_zero_for_perfect_onehot() -> None:
    y = np.array([1, 2, 3, 4, 5])
    probs = np.eye(5)
    assert classwise_ece(y, probs) < 1e-9
    assert brier_multiclass(y, probs) < 1e-9


def test_binary_ece_zero_when_confidence_matches_rate() -> None:
    conf = np.array([0.0, 0.0, 1.0, 1.0])
    correct = np.array([0, 0, 1, 1])
    assert binary_ece(conf, correct) < 1e-9


def test_ordinal_probabilities_sum_to_one() -> None:
    rng = np.random.default_rng(0)
    raw = rng.normal(size=200)
    y = np.clip(np.round(3 + raw + rng.normal(scale=0.5, size=200)), 1, 5).astype(int)
    model = OrdinalCalibrator().fit(raw, y)
    probs = model.predict_proba(raw)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert probs.shape == (200, 5)
    assert model.temperature > 0


def test_isotonic_is_monotone() -> None:
    raw = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1, 1, 3, 4, 5])
    model = IsotonicCalibrator().fit(raw, y)
    pred = model.predict_expected(raw)
    assert np.all(np.diff(pred) >= -1e-9)


def test_quote_support_exact_and_fuzzy() -> None:
    assert quote_supported("processed the refund", "I've processed the refund today.")
    assert not quote_supported("completely different sentence", "I've processed the refund today.")


def test_shifts_preserve_latent_label_and_cues() -> None:
    calls = generate_source(n_train=10, n_dev=2, n_holdout=10)
    for call in calls:
        for turn in call.turns:
            for cue in turn.cues:
                assert cue.text in turn.text
        for condition in ("disfluency", "paraphrase", "emotional_arc"):
            shifted = apply_shift(call, condition)
            assert shifted.latent_satisfaction == call.latent_satisfaction
            assert shifted.survey_response == call.survey_response
            for turn in shifted.turns:
                for cue in turn.cues:
                    assert cue.text in turn.text
