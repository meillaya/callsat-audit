"""Calibration layers for ordinal satisfaction scores.

Two post-hoc calibrators are compared:

* ``IsotonicCalibrator`` -- call-eval's monotone survey mapping (point score).
* ``OrdinalCalibrator`` -- a proportional-odds model with temperature scaling, which
  yields class probabilities for ECE, Brier, selective prediction, and conformal sets.

Both are fit on source-domain surveys only.  The stress test asks whether they transfer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _as_float(x: object) -> FloatArray:
    return np.asarray(x, dtype=float)


def _as_int(y: object) -> IntArray:
    return np.asarray(y, dtype=int)


def _ordinal_nll(
    params: FloatArray,
    x: FloatArray,
    y: IntArray,
    temperature: float = 1.0,
) -> float:
    beta = params[0]
    c1 = params[1]
    gaps = np.exp(np.clip(params[2:], -10.0, 10.0))
    cuts = np.concatenate([[c1], c1 + np.cumsum(gaps)])
    linear = beta * x
    cdf = expit((cuts[None, :] - linear[:, None]) / temperature)
    cdf = np.concatenate([np.zeros((len(x), 1)), cdf, np.ones((len(x), 1))], axis=1)
    probs = np.diff(cdf, axis=1)
    chosen = probs[np.arange(len(y)), y - 1]
    return float(-np.mean(np.log(np.clip(chosen, 1e-12, 1.0))))


def _fit_ordinal_params(x: FloatArray, y: IntArray) -> FloatArray:
    freq = np.array([(y <= k).mean() for k in range(1, 5)])
    init_cuts = np.quantile(x, np.clip(freq, 0.02, 0.98))
    init_cuts = np.maximum.accumulate(init_cuts + np.arange(4) * 1e-3)
    gaps = np.diff(init_cuts)
    params0 = np.concatenate([[0.8, init_cuts[0]], np.log(np.maximum(gaps, 1e-2))])
    result = minimize(
        _ordinal_nll,
        params0,
        args=(x, y),
        method="L-BFGS-B",
        options={"maxiter": 2000},
    )
    if not result.success:
        # Retry from a flatter start rather than silently returning the initial point.
        result = minimize(
            _ordinal_nll,
            np.concatenate([[0.5, -1.0], np.log(np.ones(3) * 0.7)]),
            args=(x, y),
            method="L-BFGS-B",
            options={"maxiter": 4000},
        )
    return _as_float(result.x)


def fit_temperature_from_probs(probs: FloatArray, y: IntArray) -> float:
    """Single-parameter temperature scaling on held-out or out-of-fold probabilities."""
    p = np.clip(_as_float(probs), 1e-12, 1.0)
    y_int = _as_int(y)

    def objective(log_t: FloatArray) -> float:
        t = float(np.exp(np.clip(log_t[0], -3.0, 3.0)))
        adjusted = p ** (1.0 / t)
        adjusted = adjusted / adjusted.sum(axis=1, keepdims=True)
        chosen = adjusted[np.arange(len(y_int)), y_int - 1]
        return float(-np.mean(np.log(np.clip(chosen, 1e-12, 1.0))))

    result = minimize(objective, np.array([0.0]), method="Nelder-Mead")
    return float(np.exp(np.clip(result.x[0], -3.0, 3.0)))


@dataclass
class OrdinalCalibrator:
    """Proportional-odds model on the raw score, with optional temperature scaling."""

    beta: float = 0.0
    cutpoints: FloatArray = field(default_factory=lambda: np.zeros(4))
    x_mean: float = 0.0
    x_std: float = 1.0
    temperature: float = 1.0
    name: str = "ordinal"

    def fit(self, raw: FloatArray, y: IntArray) -> OrdinalCalibrator:
        x = _as_float(raw)
        self.x_mean = float(x.mean())
        self.x_std = float(x.std() or 1.0)
        xs = (x - self.x_mean) / self.x_std
        params = _fit_ordinal_params(xs, _as_int(y))
        self.beta = float(params[0])
        c1 = float(params[1])
        gaps = np.exp(np.clip(params[2:], -10.0, 10.0))
        self.cutpoints = np.concatenate([[c1], c1 + np.cumsum(gaps)])
        self.temperature = 1.0
        return self

    def _linear(self, raw: FloatArray) -> FloatArray:
        return self.beta * ((_as_float(raw) - self.x_mean) / self.x_std)

    def predict_proba(self, raw: FloatArray) -> FloatArray:
        linear = self._linear(raw)
        cdf = expit((self.cutpoints[None, :] - linear[:, None]) / self.temperature)
        cdf = np.concatenate([np.zeros((len(linear), 1)), cdf, np.ones((len(linear), 1))], axis=1)
        return np.diff(cdf, axis=1)

    def fit_temperature(self, raw: FloatArray, y: IntArray) -> float:
        """Fit a single temperature on held-out or out-of-fold probabilities."""
        self.temperature = fit_temperature_from_probs(self.predict_proba(raw), _as_int(y))
        return self.temperature

    def predict_expected(self, raw: FloatArray) -> FloatArray:
        probs = self.predict_proba(raw)
        return probs @ np.arange(1, probs.shape[1] + 1)

    def predict_label(self, raw: FloatArray) -> IntArray:
        return self.predict_proba(raw).argmax(axis=1) + 1


@dataclass
class IsotonicCalibrator:
    """Monotone survey mapping used by call-eval."""

    model: IsotonicRegression | None = None
    name: str = "isotonic"

    def fit(self, raw: FloatArray, y: IntArray) -> IsotonicCalibrator:
        self.model = IsotonicRegression(y_min=1.0, y_max=5.0, out_of_bounds="clip")
        self.model.fit(_as_float(raw), _as_int(y))
        return self

    def predict_expected(self, raw: FloatArray) -> FloatArray:
        if self.model is None:
            raise RuntimeError("calibrator not fitted")
        return _as_float(self.model.predict(_as_float(raw)))

    def predict_label(self, raw: FloatArray) -> IntArray:
        return np.clip(np.round(self.predict_expected(raw)), 1, 5).astype(int)


@dataclass
class ConformalOrdinal:
    """Split-conformal prediction sets around an ordinal posterior.

    Nonconformity is ``1 - p(y)``.  The finite-sample coverage guarantee assumes
    exchangeability, which the shift conditions deliberately violate; coverage is
    therefore an audit outcome, not an assumption.
    """

    base: OrdinalCalibrator
    alpha: float = 0.10
    qhat: float = 1.0

    def fit(self, raw: FloatArray, y: IntArray) -> ConformalOrdinal:
        probs = self.base.predict_proba(raw)
        y_int = _as_int(y)
        scores = 1.0 - probs[np.arange(len(y_int)), y_int - 1]
        n = len(scores)
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.qhat = float(np.quantile(scores, level))
        return self

    def prediction_sets(self, raw: FloatArray) -> list[list[int]]:
        probs = self.base.predict_proba(raw)
        sets: list[list[int]] = []
        for row in probs:
            members = [int(k + 1) for k, p in enumerate(row) if 1.0 - p <= self.qhat]
            sets.append(members or [int(row.argmax() + 1)])
        return sets

    def coverage(self, raw: FloatArray, y: IntArray) -> float:
        sets = self.prediction_sets(raw)
        return float(np.mean([int(truth) in s for s, truth in zip(sets, _as_int(y), strict=True)]))

    def mean_set_size(self, raw: FloatArray) -> float:
        return float(np.mean([len(s) for s in self.prediction_sets(raw)]))
