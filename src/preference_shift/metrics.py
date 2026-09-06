"""Predictive losses and detection metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def binary_log_loss(outcomes: ArrayLike, probabilities: ArrayLike) -> FloatArray:
    """Pointwise Bernoulli log loss."""

    y = np.asarray(outcomes, dtype=float)
    q = _clip_probabilities(probabilities)
    if y.shape != q.shape:
        raise ValueError("outcomes and probabilities must have the same shape")
    return -(y * np.log(q) + (1.0 - y) * np.log1p(-q))


def bernoulli_entropy(probabilities: ArrayLike) -> FloatArray:
    """Entropy of one or more Bernoulli distributions."""

    p = _clip_probabilities(probabilities)
    return -(p * np.log(p) + (1.0 - p) * np.log1p(-p))


def bernoulli_kl(true_probabilities: ArrayLike, estimated_probabilities: ArrayLike) -> FloatArray:
    """KL(Bernoulli(true) || Bernoulli(estimated))."""

    p = _clip_probabilities(true_probabilities)
    q = _clip_probabilities(estimated_probabilities)
    if p.shape != q.shape:
        raise ValueError("probability arrays must have the same shape")
    return p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))


def classification_correct(outcomes: ArrayLike, probabilities: ArrayLike) -> FloatArray:
    """Pointwise correctness using a probability threshold of one half."""

    y = np.asarray(outcomes, dtype=int)
    q = np.asarray(probabilities, dtype=float)
    if y.shape != q.shape:
        raise ValueError("outcomes and probabilities must have the same shape")
    return ((q >= 0.5).astype(int) == y).astype(float)


def detection_delay(alarm_index: int | None, change_point: int) -> float:
    """Return delay, infinity for no alarm, and NaN for a false alarm."""

    if alarm_index is None:
        return float("inf")
    if alarm_index < change_point:
        return float("nan")
    return float(alarm_index - change_point)


def _clip_probabilities(probabilities: ArrayLike) -> FloatArray:
    values = np.asarray(probabilities, dtype=float)
    epsilon = np.finfo(float).eps
    return np.clip(values, epsilon, 1.0 - epsilon)
