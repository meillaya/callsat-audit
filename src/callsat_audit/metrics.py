"""Metrics for ordinal satisfaction scoring and calibration audits."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import cohen_kappa_score

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
FloatLike = Sequence[float] | NDArray[np.floating]
IntLike = Sequence[int] | NDArray[np.integer]


def as_float(values: FloatLike) -> FloatArray:
    return np.asarray(values, dtype=float)


def as_int(values: IntLike) -> IntArray:
    return np.asarray(values, dtype=int)


def quadratic_weighted_kappa(y_true: IntLike, y_pred: IntLike) -> float:
    return float(cohen_kappa_score(as_int(y_true), as_int(y_pred), weights="quadratic"))


def accuracy(y_true: IntLike, y_pred: IntLike) -> float:
    return float(np.mean(as_int(y_true) == as_int(y_pred)))


def mae(y_true: FloatLike, y_pred: FloatLike) -> float:
    return float(np.mean(np.abs(as_float(y_true) - as_float(y_pred))))


def spearman(y_true: FloatLike, y_pred: FloatLike) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(as_float(y_true), as_float(y_pred)).statistic)


def expected_score(probs: NDArray[np.floating]) -> FloatArray:
    """Expected ordinal label under a 5-class distribution."""
    probs = np.asarray(probs, dtype=float)
    labels = np.arange(1, probs.shape[1] + 1)
    return probs @ labels


def classwise_ece(
    y_true: IntLike,
    probs: NDArray[np.floating],
    n_bins: int = 10,
) -> float:
    """Standard top-label ECE: |accuracy - confidence| averaged over confidence bins."""
    y = as_int(y_true)
    p = np.asarray(probs, dtype=float)
    conf = p.max(axis=1)
    pred = p.argmax(axis=1) + 1
    correct = (pred == y).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in pairwise(bins):
        mask = (conf > lo) & (conf <= hi)
        if not np.any(mask):
            continue
        total += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(total)


def binary_ece(
    confidence: FloatLike,
    correct: IntLike | Sequence[bool],
    n_bins: int = 10,
) -> float:
    """ECE for a single confidence score and a binary correctness outcome."""
    conf = as_float(confidence)
    hit = np.asarray(correct, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in pairwise(bins):
        mask = (conf > lo) & (conf <= hi)
        if not np.any(mask):
            continue
        total += mask.mean() * abs(hit[mask].mean() - conf[mask].mean())
    return float(total)


def adaptive_ece(
    y_true: IntLike,
    probs: NDArray[np.floating],
    n_bins: int = 10,
) -> float:
    """ECE with equal-mass confidence bins (less sensitive to empty bins)."""
    y = as_int(y_true)
    p = np.asarray(probs, dtype=float)
    conf = p.max(axis=1)
    pred = p.argmax(axis=1) + 1
    correct = (pred == y).astype(float)
    order = np.argsort(conf)
    chunks = np.array_split(order, n_bins)
    total = 0.0
    for idx in chunks:
        if len(idx) == 0:
            continue
        total += (len(idx) / len(y)) * abs(correct[idx].mean() - conf[idx].mean())
    return float(total)


def brier_multiclass(y_true: IntLike, probs: NDArray[np.floating]) -> float:
    y = as_int(y_true)
    p = np.asarray(probs, dtype=float)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y - 1] = 1.0
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def nll(y_true: IntLike, probs: NDArray[np.floating], eps: float = 1e-9) -> float:
    y = as_int(y_true)
    p = np.asarray(probs, dtype=float)
    chosen = p[np.arange(len(y)), y - 1]
    return float(-np.mean(np.log(np.clip(chosen, eps, 1.0))))


def calibration_slope(y_true: Sequence[float], y_pred: Sequence[float]) -> tuple[float, float]:
    """OLS of observed label on predicted score: slope 1 and intercept 0 is calibrated."""
    x = as_float(y_pred)
    y = as_float(y_true)
    design = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coef[1]), float(coef[0])


def reliability_bins(
    y_true: IntLike, probs: NDArray[np.floating], n_bins: int = 10
) -> tuple[FloatArray, FloatArray, IntArray]:
    y = as_int(y_true)
    p = np.asarray(probs, dtype=float)
    conf = p.max(axis=1)
    pred = p.argmax(axis=1) + 1
    correct = (pred == y).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    xs, ys, ns = [], [], []
    for lo, hi in pairwise(bins):
        mask = (conf > lo) & (conf <= hi)
        if np.any(mask):
            xs.append(conf[mask].mean())
            ys.append(correct[mask].mean())
            ns.append(int(mask.sum()))
    return as_float(xs), as_float(ys), as_int(ns)


def risk_coverage(
    y_true: FloatLike,
    y_pred: FloatLike,
    confidence: FloatLike,
    large_error_threshold: float = 1.0,
) -> dict[str, FloatArray]:
    """Selective prediction: sort by confidence, report risk at each coverage level."""
    y = as_float(y_true)
    pred = as_float(y_pred)
    conf = as_float(confidence)
    order = np.argsort(-conf)
    abs_err = np.abs(pred - y)[order]
    large = (abs_err > large_error_threshold).astype(float)
    n = len(y)
    coverage = np.arange(1, n + 1) / n
    cumulative_mae = np.cumsum(abs_err) / np.arange(1, n + 1)
    cumulative_large = np.cumsum(large) / np.arange(1, n + 1)
    return {
        "coverage": coverage,
        "mae": cumulative_mae,
        "large_error_rate": cumulative_large,
    }


def risk_coverage_auc(
    y_true: FloatLike,
    y_pred: FloatLike,
    confidence: FloatLike,
    large_error_threshold: float = 1.0,
) -> float:
    curve = risk_coverage(y_true, y_pred, confidence, large_error_threshold)
    return float(np.trapezoid(curve["large_error_rate"], curve["coverage"]))


def aurc(
    y_true: FloatLike,
    y_pred: FloatLike,
    confidence: FloatLike,
) -> float:
    curve = risk_coverage(y_true, y_pred, confidence)
    return float(np.trapezoid(curve["mae"], curve["coverage"]))


def bootstrap_ci(
    metric: Callable[[], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap over a closure that resamples internally.

    The closure must draw its own bootstrap sample; this helper only supplies the
    quantiles.  Kept deliberately simple so notebook code stays readable.
    """
    rng = np.random.default_rng(seed)
    values = np.array([metric() for _ in range(n_boot)], dtype=float)
    _ = rng  # closure owns randomness; retained for API symmetry
    return float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))


def paired_bootstrap_delta(
    y_true: Sequence[float],
    pred_a: Sequence[float],
    pred_b: Sequence[float],
    metric: Callable[[FloatArray, FloatArray], float],
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI for metric(a) - metric(b) on paired predictions."""
    y = as_float(y_true)
    a = as_float(pred_a)
    b = as_float(pred_b)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    n = len(y)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[i] = metric(y[idx], a[idx]) - metric(y[idx], b[idx])
    point = metric(y, a) - metric(y, b)
    return point, float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def bootstrap_qwk_ci(
    y_true: IntLike,
    y_pred: IntLike,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    y = as_int(y_true)
    p = as_int(y_pred)
    rng = np.random.default_rng(seed)
    values = np.empty(n_boot)
    n = len(y)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        values[i] = quadratic_weighted_kappa(y[idx], p[idx])
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def bootstrap_ece_ci(
    y_true: IntLike,
    probs: NDArray[np.floating],
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    y = as_int(y_true)
    p = np.asarray(probs, dtype=float)
    rng = np.random.default_rng(seed)
    values = np.empty(n_boot)
    n = len(y)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        values[i] = classwise_ece(y[idx], p[idx])
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def paired_qwk_delta_ci(
    y_true: IntLike,
    pred_source: IntLike,
    pred_target: IntLike,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI for QWK(source) - QWK(target) on paired calls."""
    y = as_int(y_true)
    a = as_int(pred_source)
    b = as_int(pred_target)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    n = len(y)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[i] = quadratic_weighted_kappa(y[idx], a[idx]) - quadratic_weighted_kappa(
            y[idx], b[idx]
        )
    point = quadratic_weighted_kappa(y, a) - quadratic_weighted_kappa(y, b)
    return point, float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def confusion(y_true: IntLike, y_pred: IntLike, n_labels: int = 5) -> IntArray:
    y = as_int(y_true)
    p = as_int(y_pred)
    matrix = np.zeros((n_labels, n_labels), dtype=int)
    for truth, pred in zip(y, p, strict=True):
        matrix[truth - 1, pred - 1] += 1
    return matrix
