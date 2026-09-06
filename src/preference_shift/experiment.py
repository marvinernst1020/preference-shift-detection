"""End-to-end orchestration for one controlled primary experiment."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray
from scipy.special import expit

from preference_shift.config import PrimaryConfig
from preference_shift.metrics import (
    bernoulli_kl,
    binary_log_loss,
    classification_correct,
)
from preference_shift.model import LaplacePreferenceModel
from preference_shift.monitoring import (
    calibrate_cusum_threshold,
    calibrate_rolling_accuracy_threshold,
    first_lower_crossing,
    rolling_accuracy,
    run_cusum,
    standardized_cusum_path,
)
from preference_shift.shifts import (
    direction_at_angle,
    most_uncertain_direction,
    rank_one_covariate_shift,
    shifted_weights,
)
from preference_shift.simulation import (
    PairDistribution,
    PreferenceRegime,
    simulate_stationary,
    simulate_transition,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PrimaryRun:
    """In-memory output of a primary experiment."""

    summary: dict[str, Any]
    time_series: dict[str, np.ndarray[Any, Any]]
    arrays: dict[str, np.ndarray[Any, Any]]


def run_primary_experiment(config: PrimaryConfig) -> PrimaryRun:
    """Fit, calibrate, and monitor one reproducible synthetic stream."""

    seed_sequence = np.random.SeedSequence(config.seed)
    fit_seed, calibration_seed, direction_seed, null_seed, stream_seed, risk_seed = (
        seed_sequence.spawn(6)
    )

    base_pairs = PairDistribution(
        mean=np.zeros(config.dimension),
        covariance=np.diag(config.training_variances),
    )
    base_weights = np.asarray(config.true_weights, dtype=float)
    reference_regime = PreferenceRegime(base_pairs, base_weights)

    fit_stream = simulate_stationary(
        reference_regime,
        config.n_fit,
        np.random.default_rng(fit_seed),
    )
    model = LaplacePreferenceModel(
        prior_precision=config.prior_precision,
        quadrature_order=config.quadrature_order,
    ).fit(fit_stream.differences, fit_stream.outcomes)

    calibration_stream = simulate_stationary(
        reference_regime,
        config.n_calibration,
        np.random.default_rng(calibration_seed),
    )
    calibration_prediction = model.predict(calibration_stream.differences)
    calibration_uncertainty = calibration_prediction.epistemic_variance
    calibration_loss = binary_log_loss(
        calibration_stream.outcomes,
        calibration_prediction.probability,
    )
    uncertainty_mean, uncertainty_std = _reference_moments(calibration_uncertainty)
    loss_mean, loss_std = _reference_moments(calibration_loss)

    direction_rng = np.random.default_rng(direction_seed)
    uncertainty_direction = most_uncertain_direction(model.posterior_covariance)
    covariate_direction = direction_at_angle(
        uncertainty_direction,
        config.covariate_alignment_degrees,
        direction_rng,
    )
    preference_direction = direction_at_angle(
        uncertainty_direction,
        config.preference_alignment_degrees,
        direction_rng,
    )
    shifted_regime = _make_shifted_regime(
        reference_regime,
        config,
        covariate_direction,
        preference_direction,
    )

    null_uncertainty, null_loss, null_correctness = _simulate_null_signals(
        reference_regime,
        model,
        config,
        null_seed,
    )
    uncertainty_threshold = calibrate_cusum_threshold(
        null_uncertainty,
        uncertainty_mean,
        uncertainty_std,
        config.cusum_allowance,
        config.false_alarm_probability,
    )
    loss_threshold = calibrate_cusum_threshold(
        null_loss,
        loss_mean,
        loss_std,
        config.cusum_allowance,
        config.false_alarm_probability,
    )
    accuracy_threshold = calibrate_rolling_accuracy_threshold(
        null_correctness,
        config.rolling_accuracy_window,
        config.false_alarm_probability,
    )

    stream = simulate_transition(
        reference_regime,
        shifted_regime,
        horizon=config.horizon,
        change_point=config.change_point,
        transition_width=config.transition_width,
        rng=np.random.default_rng(stream_seed),
    )
    prediction = model.predict(stream.differences)
    uncertainty_signal = prediction.epistemic_variance
    loss_signal = binary_log_loss(stream.outcomes, prediction.probability)
    correctness = classification_correct(stream.outcomes, prediction.probability)

    uncertainty_detection = run_cusum(
        uncertainty_signal,
        uncertainty_mean,
        uncertainty_std,
        config.cusum_allowance,
        uncertainty_threshold,
    )
    loss_detection = run_cusum(
        loss_signal,
        loss_mean,
        loss_std,
        config.cusum_allowance,
        loss_threshold,
    )
    accuracy_path = rolling_accuracy(correctness, config.rolling_accuracy_window)
    accuracy_alarm = first_lower_crossing(accuracy_path, accuracy_threshold)

    risk_curve, reference_risk = _estimate_excess_risk_curve(
        reference_regime,
        shifted_regime,
        stream.transition_fraction,
        model,
        config.risk_evaluation_samples,
        np.random.default_rng(risk_seed),
    )
    failure_index = _first_failure_index(
        risk_curve,
        reference_risk + config.failure_excess_risk_increase,
        config.change_point,
    )
    pointwise_excess_loss = bernoulli_kl(
        stream.true_probability,
        prediction.probability,
    )

    model_error = model.posterior_mean - base_weights
    summary = {
        "scenario": config.scenario,
        "seed": config.seed,
        "indexing": {
            "array_indices": "zero-based",
            "reported_times": "one-based",
            "first_post_change_index": config.change_point,
            "first_post_change_time": config.change_point + 1,
        },
        "model": {
            "posterior_mean": model.posterior_mean.tolist(),
            "posterior_covariance": model.posterior_covariance.tolist(),
            "parameter_error_norm": float(np.linalg.norm(model_error)),
        },
        "geometry": {
            "covariate_direction": covariate_direction.tolist(),
            "preference_direction": preference_direction.tolist(),
            "posterior_variance_along_covariate_shift": float(
                covariate_direction @ model.posterior_covariance @ covariate_direction
            ),
            "realized_error_projection_squared": float(
                (model_error @ covariate_direction) ** 2
            ),
            "direction_inner_product": float(covariate_direction @ preference_direction),
        },
        "calibration": {
            "uncertainty_reference_mean": uncertainty_mean,
            "uncertainty_reference_std": uncertainty_std,
            "uncertainty_cusum_threshold": uncertainty_threshold,
            "loss_reference_mean": loss_mean,
            "loss_reference_std": loss_std,
            "loss_cusum_threshold": loss_threshold,
            "rolling_accuracy_threshold": accuracy_threshold,
            "uncertainty_empirical_false_alarm_rate": _cusum_false_alarm_rate(
                null_uncertainty,
                uncertainty_mean,
                uncertainty_std,
                config.cusum_allowance,
                uncertainty_threshold,
            ),
            "loss_empirical_false_alarm_rate": _cusum_false_alarm_rate(
                null_loss,
                loss_mean,
                loss_std,
                config.cusum_allowance,
                loss_threshold,
            ),
            "accuracy_empirical_false_alarm_rate": _accuracy_false_alarm_rate(
                null_correctness,
                config.rolling_accuracy_window,
                accuracy_threshold,
            ),
        },
        "alarms": {
            "uncertainty": _alarm_summary(
                uncertainty_detection.alarm_index,
                config.change_point,
                failure_index,
            ),
            "log_loss": _alarm_summary(
                loss_detection.alarm_index,
                config.change_point,
                failure_index,
            ),
            "rolling_accuracy": _alarm_summary(
                accuracy_alarm,
                config.change_point,
                failure_index,
            ),
        },
        "failure": {
            "reference_excess_risk": reference_risk,
            "risk_increase_threshold": config.failure_excess_risk_increase,
            "failure_index": failure_index,
            "failure_time": None if failure_index is None else failure_index + 1,
        },
    }

    indices = np.arange(config.horizon)
    time_series: dict[str, np.ndarray[Any, Any]] = {
        "index": indices,
        "time": indices + 1,
        "transition_fraction": stream.transition_fraction,
        "outcome": stream.outcomes,
        "true_probability": stream.true_probability,
        "predicted_probability": prediction.probability,
        "latent_uncertainty": prediction.latent_variance,
        "epistemic_uncertainty": prediction.epistemic_variance,
        "aleatoric_uncertainty": prediction.aleatoric_variance,
        "log_loss": loss_signal,
        "correct": correctness,
        "rolling_accuracy": accuracy_path,
        "uncertainty_cusum": uncertainty_detection.path,
        "loss_cusum": loss_detection.path,
        "pointwise_excess_loss": pointwise_excess_loss,
        "oracle_excess_risk": risk_curve,
    }
    arrays: dict[str, np.ndarray[Any, Any]] = {
        "differences": stream.differences,
        "outcomes": stream.outcomes,
        "true_weights": stream.true_weights,
        "posterior_mean": model.posterior_mean,
        "posterior_covariance": model.posterior_covariance,
        "covariate_direction": covariate_direction,
        "preference_direction": preference_direction,
    }
    return PrimaryRun(summary=summary, time_series=time_series, arrays=arrays)


def save_primary_run(
    run: PrimaryRun,
    config: PrimaryConfig,
    output_directory: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Save a self-contained result directory without deleting other files."""

    destination = Path(output_directory)
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        message = (
            f"result directory is not empty: {destination}; "
            "pass overwrite=True to replace known files"
        )
        raise FileExistsError(message)
    destination.mkdir(parents=True, exist_ok=True)

    with (destination / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.as_dict(), handle, sort_keys=False)
    with (destination / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(run.summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")

    columns = list(run.time_series)
    n_rows = len(run.time_series[columns[0]])
    with (destination / "time_series.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in range(n_rows):
            writer.writerow([run.time_series[column][row] for column in columns])

    np.savez_compressed(destination / "run_arrays.npz", **run.arrays)
    return destination


def _make_shifted_regime(
    reference: PreferenceRegime,
    config: PrimaryConfig,
    covariate_direction: FloatArray,
    preference_direction: FloatArray,
) -> PreferenceRegime:
    pairs = reference.pairs
    weights = reference.weights
    if config.scenario in {"covariate", "combined"}:
        pairs = rank_one_covariate_shift(
            reference.pairs,
            covariate_direction,
            config.mean_shift_magnitude,
            config.covariance_shift_magnitude,
        )
    if config.scenario in {"preference", "combined"}:
        weights = shifted_weights(
            reference.weights,
            preference_direction,
            config.preference_shift_magnitude,
        )
    return PreferenceRegime(pairs, weights)


def _simulate_null_signals(
    regime: PreferenceRegime,
    model: LaplacePreferenceModel,
    config: PrimaryConfig,
    seed_sequence: np.random.SeedSequence,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    shape = (config.null_calibration_runs, config.horizon)
    uncertainty = np.empty(shape)
    loss = np.empty(shape)
    correctness = np.empty(shape)
    for run_index, child_seed in enumerate(seed_sequence.spawn(config.null_calibration_runs)):
        stream = simulate_stationary(
            regime,
            config.horizon,
            np.random.default_rng(child_seed),
        )
        prediction = model.predict(stream.differences)
        uncertainty[run_index] = prediction.epistemic_variance
        loss[run_index] = binary_log_loss(stream.outcomes, prediction.probability)
        correctness[run_index] = classification_correct(
            stream.outcomes,
            prediction.probability,
        )
    return uncertainty, loss, correctness


def _estimate_excess_risk_curve(
    before: PreferenceRegime,
    after: PreferenceRegime,
    transition_fraction: FloatArray,
    model: LaplacePreferenceModel,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[FloatArray, float]:
    z_before = rng.multivariate_normal(
        before.pairs.mean,
        before.pairs.covariance,
        size=n_samples,
    )
    z_after = rng.multivariate_normal(
        after.pairs.mean,
        after.pairs.covariance,
        size=n_samples,
    )
    q_before = model.predict(z_before).probability
    q_after = model.predict(z_after).probability

    observed_fractions, inverse = np.unique(transition_fraction, return_inverse=True)
    evaluated_fractions = np.unique(np.concatenate([np.array([0.0]), observed_fractions]))
    risks = np.empty(evaluated_fractions.size)
    for index, fraction in enumerate(evaluated_fractions):
        weights = (1.0 - fraction) * before.weights + fraction * after.weights
        p_before = expit(z_before @ weights)
        p_after = expit(z_after @ weights)
        risks[index] = (1.0 - fraction) * np.mean(bernoulli_kl(p_before, q_before)) + (
            fraction * np.mean(bernoulli_kl(p_after, q_after))
        )

    fraction_to_index = {value: index for index, value in enumerate(evaluated_fractions)}
    observed_risks = np.array([risks[fraction_to_index[value]] for value in observed_fractions])
    return observed_risks[inverse], float(risks[fraction_to_index[0.0]])


def _reference_moments(signal: FloatArray) -> tuple[float, float]:
    mean = float(np.mean(signal))
    standard_deviation = float(np.std(signal, ddof=1))
    return mean, max(standard_deviation, np.finfo(float).eps)


def _first_failure_index(
    risk_curve: FloatArray,
    threshold: float,
    change_point: int,
) -> int | None:
    candidates = np.flatnonzero(
        (np.arange(risk_curve.size) >= change_point) & (risk_curve >= threshold)
    )
    return int(candidates[0]) if candidates.size else None


def _alarm_summary(
    alarm_index: int | None,
    change_point: int,
    failure_index: int | None,
) -> dict[str, Any]:
    false_alarm = alarm_index is not None and alarm_index < change_point
    valid_alarm = alarm_index is not None and not false_alarm
    delay = int(alarm_index - change_point) if valid_alarm else None
    lead_time = (
        int(failure_index - alarm_index)
        if valid_alarm and failure_index is not None
        else None
    )
    return {
        "alarm_index": alarm_index,
        "alarm_time": None if alarm_index is None else alarm_index + 1,
        "false_alarm": false_alarm,
        "detection_delay": delay,
        "lead_time_to_failure": lead_time,
    }


def _cusum_false_alarm_rate(
    null_signals: FloatArray,
    reference_mean: float,
    reference_std: float,
    allowance: float,
    threshold: float,
) -> float:
    alarms = [
        standardized_cusum_path(row, reference_mean, reference_std, allowance).max()
        > threshold
        for row in null_signals
    ]
    return float(np.mean(alarms))


def _accuracy_false_alarm_rate(
    null_correctness: FloatArray,
    window: int,
    threshold: float,
) -> float:
    alarms = [
        np.nanmin(rolling_accuracy(row, window)) <= threshold for row in null_correctness
    ]
    return float(np.mean(alarms))
