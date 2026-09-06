"""Sequential monitoring rules shared by uncertainty and feedback signals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DetectionResult:
    path: FloatArray
    alarm_index: int | None


def standardized_cusum_path(
    signal: ArrayLike,
    reference_mean: float,
    reference_std: float,
    allowance: float,
) -> FloatArray:
    """Compute a one-sided standardized CUSUM path."""

    values = np.asarray(signal, dtype=float)
    if values.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if reference_std <= 0:
        raise ValueError("reference_std must be positive")
    if allowance < 0:
        raise ValueError("allowance cannot be negative")

    path = np.empty(values.size, dtype=float)
    state = 0.0
    for index, value in enumerate(values):
        increment = (value - reference_mean) / reference_std - allowance
        state = max(0.0, state + increment)
        path[index] = state
    return path


def run_cusum(
    signal: ArrayLike,
    reference_mean: float,
    reference_std: float,
    allowance: float,
    threshold: float,
) -> DetectionResult:
    """Run CUSUM and return its first threshold crossing."""

    if threshold <= 0:
        raise ValueError("threshold must be positive")
    path = standardized_cusum_path(signal, reference_mean, reference_std, allowance)
    crossings = np.flatnonzero(path > threshold)
    alarm = int(crossings[0]) if crossings.size else None
    return DetectionResult(path=path, alarm_index=alarm)


def calibrate_cusum_threshold(
    null_signals: ArrayLike,
    reference_mean: float,
    reference_std: float,
    allowance: float,
    false_alarm_probability: float,
) -> float:
    """Calibrate a CUSUM threshold for a fixed monitoring horizon.

    Each row of ``null_signals`` is an independent stationary stream. The
    returned empirical quantile controls the probability of at least one alarm
    over that horizon.
    """

    streams = np.asarray(null_signals, dtype=float)
    if streams.ndim != 2:
        raise ValueError("null_signals must have shape (runs, horizon)")
    if not 0.0 < false_alarm_probability < 1.0:
        raise ValueError("false_alarm_probability must lie in (0, 1)")

    maxima = np.array(
        [
            standardized_cusum_path(row, reference_mean, reference_std, allowance).max()
            for row in streams
        ]
    )
    threshold = float(
        np.quantile(maxima, 1.0 - false_alarm_probability, method="higher")
    )
    return max(threshold, np.finfo(float).eps)


def rolling_accuracy(correct: ArrayLike, window: int) -> FloatArray:
    """Compute rolling accuracy, using NaN until a complete window is available."""

    values = np.asarray(correct, dtype=float)
    if values.ndim != 1:
        raise ValueError("correct must be one-dimensional")
    if window <= 0:
        raise ValueError("window must be positive")

    result = np.full(values.size, np.nan, dtype=float)
    if values.size < window:
        return result
    sums = np.convolve(values, np.ones(window), mode="valid")
    result[window - 1 :] = sums / window
    return result


def first_lower_crossing(path: ArrayLike, threshold: float) -> int | None:
    """Return the first finite index at or below a lower threshold."""

    values = np.asarray(path, dtype=float)
    crossings = np.flatnonzero(np.isfinite(values) & (values <= threshold))
    return int(crossings[0]) if crossings.size else None


def calibrate_rolling_accuracy_threshold(
    null_correctness: ArrayLike,
    window: int,
    false_alarm_probability: float,
) -> float:
    """Calibrate a lower rolling-accuracy threshold over a fixed horizon."""

    streams = np.asarray(null_correctness, dtype=float)
    if streams.ndim != 2:
        raise ValueError("null_correctness must have shape (runs, horizon)")
    if not 0.0 < false_alarm_probability < 1.0:
        raise ValueError("false_alarm_probability must lie in (0, 1)")
    if streams.shape[1] < window:
        raise ValueError("monitoring horizon must be at least as long as the window")

    minima = np.array([np.nanmin(rolling_accuracy(row, window)) for row in streams])
    candidates = np.unique(minima)
    empirical_alarm_rates = np.array([(minima <= value).mean() for value in candidates])
    valid = candidates[empirical_alarm_rates <= false_alarm_probability]
    if valid.size:
        return float(valid[-1])
    return float(np.nextafter(candidates[0], -np.inf))
