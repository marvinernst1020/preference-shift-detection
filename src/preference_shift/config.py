"""Validated configuration for the primary synthetic experiment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

Scenario = Literal["stationary", "covariate", "preference", "combined"]


@dataclass(frozen=True)
class PrimaryConfig:
    """All choices needed to reproduce one primary experiment."""

    dimension: int
    training_variances: tuple[float, ...]
    true_weights: tuple[float, ...]
    n_fit: int
    n_calibration: int
    horizon: int
    change_point: int
    transition_width: int
    scenario: Scenario
    covariate_alignment_degrees: float
    preference_alignment_degrees: float
    mean_shift_magnitude: float
    covariance_shift_magnitude: float
    preference_shift_magnitude: float
    prior_precision: float
    quadrature_order: int
    cusum_allowance: float
    false_alarm_probability: float
    null_calibration_runs: int
    rolling_accuracy_window: int
    risk_evaluation_samples: int
    failure_excess_risk_increase: float
    seed: int

    def __post_init__(self) -> None:
        if self.dimension < 2:
            raise ValueError("dimension must be at least two")
        if len(self.training_variances) != self.dimension:
            raise ValueError("training_variances must have one entry per dimension")
        if len(self.true_weights) != self.dimension:
            raise ValueError("true_weights must have one entry per dimension")
        if any(value <= 0 for value in self.training_variances):
            raise ValueError("training_variances must be positive")
        if min(self.n_fit, self.n_calibration, self.horizon) <= 0:
            raise ValueError("sample sizes and horizon must be positive")
        if not 0 <= self.change_point < self.horizon:
            raise ValueError("change_point must be a zero-based index within the horizon")
        if self.transition_width < 0:
            raise ValueError("transition_width cannot be negative")
        if self.scenario not in {"stationary", "covariate", "preference", "combined"}:
            raise ValueError(f"unknown scenario: {self.scenario}")
        for angle in (
            self.covariate_alignment_degrees,
            self.preference_alignment_degrees,
        ):
            if not 0.0 <= angle <= 90.0:
                raise ValueError("alignment angles must lie in [0, 90]")
        if self.prior_precision <= 0:
            raise ValueError("prior_precision must be positive")
        if self.quadrature_order < 5:
            raise ValueError("quadrature_order must be at least five")
        if self.cusum_allowance < 0:
            raise ValueError("cusum_allowance cannot be negative")
        if not 0.0 < self.false_alarm_probability < 1.0:
            raise ValueError("false_alarm_probability must lie in (0, 1)")
        if self.null_calibration_runs < 20:
            raise ValueError("null_calibration_runs must be at least 20")
        if not 1 <= self.rolling_accuracy_window <= self.horizon:
            raise ValueError("rolling_accuracy_window must lie within the horizon")
        if self.risk_evaluation_samples <= 0:
            raise ValueError("risk_evaluation_samples must be positive")
        if self.failure_excess_risk_increase <= 0:
            raise ValueError("failure_excess_risk_increase must be positive")
        if self.seed < 0:
            raise ValueError("seed cannot be negative")

    @classmethod
    def from_yaml(cls, path: str | Path) -> PrimaryConfig:
        """Load a configuration and reject missing or unknown fields."""

        with Path(path).open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, Mapping):
            raise ValueError("configuration root must be a mapping")

        values: dict[str, Any] = dict(raw)
        for field in ("training_variances", "true_weights"):
            if field in values:
                values[field] = tuple(float(value) for value in values[field])
        return cls(**values)

    def as_dict(self) -> dict[str, Any]:
        """Return a serialization-safe representation."""

        values = asdict(self)
        values["training_variances"] = list(self.training_variances)
        values["true_weights"] = list(self.true_weights)
        return values
