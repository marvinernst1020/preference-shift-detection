import numpy as np

from preference_shift.metrics import bernoulli_entropy, bernoulli_kl
from preference_shift.monitoring import (
    calibrate_rolling_accuracy_threshold,
    rolling_accuracy,
    run_cusum,
)


def test_kl_is_zero_for_matching_probabilities() -> None:
    probabilities = np.array([0.1, 0.5, 0.9])
    np.testing.assert_allclose(bernoulli_kl(probabilities, probabilities), 0.0, atol=1e-14)
    assert np.all(bernoulli_entropy(probabilities) > 0)


def test_cusum_accumulates_persistent_upward_shift() -> None:
    signal = np.concatenate([np.zeros(20), np.ones(20)])
    result = run_cusum(
        signal,
        reference_mean=0.0,
        reference_std=1.0,
        allowance=0.25,
        threshold=3.0,
    )
    assert result.alarm_index is not None
    assert result.alarm_index >= 20


def test_rolling_accuracy_waits_for_full_window() -> None:
    path = rolling_accuracy(np.array([1, 0, 1, 1, 0]), window=3)
    assert np.isnan(path[:2]).all()
    np.testing.assert_allclose(path[2:], np.array([2 / 3, 2 / 3, 2 / 3]))


def test_accuracy_threshold_respects_empirical_false_alarm_target() -> None:
    rng = np.random.default_rng(23)
    streams = rng.binomial(1, 0.7, size=(200, 80))
    threshold = calibrate_rolling_accuracy_threshold(
        streams,
        window=20,
        false_alarm_probability=0.05,
    )
    alarm_rate = np.mean(
        [np.nanmin(rolling_accuracy(row, 20)) <= threshold for row in streams]
    )
    assert alarm_rate <= 0.05
