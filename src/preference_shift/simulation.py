"""Synthetic pairwise comparison streams."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class PairDistribution:
    """Gaussian distribution of ordered feature differences."""

    mean: FloatArray
    covariance: FloatArray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=float)
        covariance = np.asarray(self.covariance, dtype=float)
        if mean.ndim != 1:
            raise ValueError("mean must be a vector")
        if covariance.shape != (mean.size, mean.size):
            raise ValueError("covariance shape must match mean dimension")
        if not np.allclose(covariance, covariance.T, atol=1e-10):
            raise ValueError("covariance must be symmetric")
        if np.linalg.eigvalsh(covariance).min() <= 0:
            raise ValueError("covariance must be positive definite")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)

    @property
    def dimension(self) -> int:
        return int(self.mean.size)


@dataclass(frozen=True)
class PreferenceRegime:
    """Pair distribution and true preference parameter for one regime."""

    pairs: PairDistribution
    weights: FloatArray

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=float)
        if weights.shape != (self.pairs.dimension,):
            raise ValueError("weights must match the pair-distribution dimension")
        object.__setattr__(self, "weights", weights)


@dataclass(frozen=True)
class PairStream:
    """Realized pairwise stream, including simulator-only ground truth."""

    x_a: FloatArray
    x_b: FloatArray
    differences: FloatArray
    outcomes: IntArray
    true_probability: FloatArray
    transition_fraction: FloatArray
    true_weights: FloatArray


def simulate_stationary(
    regime: PreferenceRegime,
    n_observations: int,
    rng: np.random.Generator,
    center_scale: float = 1.0,
) -> PairStream:
    """Draw an i.i.d. stream from one stationary preference regime."""

    if n_observations <= 0:
        raise ValueError("n_observations must be positive")
    return simulate_transition(
        before=regime,
        after=regime,
        horizon=n_observations,
        change_point=n_observations,
        transition_width=0,
        rng=rng,
        center_scale=center_scale,
    )


def simulate_transition(
    before: PreferenceRegime,
    after: PreferenceRegime,
    horizon: int,
    change_point: int,
    transition_width: int,
    rng: np.random.Generator,
    center_scale: float = 1.0,
) -> PairStream:
    """Simulate an abrupt or gradual transition between two regimes.

    ``change_point`` is the number of fully pre-change observations. For an
    abrupt transition, observation ``change_point`` in zero-based indexing is
    the first post-change observation.
    """

    if before.pairs.dimension != after.pairs.dimension:
        raise ValueError("before and after regimes must have the same dimension")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not 0 <= change_point <= horizon:
        raise ValueError("change_point must lie between zero and horizon")
    if transition_width < 0:
        raise ValueError("transition_width cannot be negative")
    if center_scale < 0:
        raise ValueError("center_scale cannot be negative")

    dimension = before.pairs.dimension
    times = np.arange(horizon)
    if transition_width == 0:
        fractions = (times >= change_point).astype(float)
    else:
        fractions = np.clip((times - change_point) / transition_width, 0.0, 1.0)

    use_after_distribution = rng.random(horizon) < fractions
    differences = np.empty((horizon, dimension), dtype=float)
    before_indices = np.flatnonzero(~use_after_distribution)
    after_indices = np.flatnonzero(use_after_distribution)
    if before_indices.size:
        differences[before_indices] = rng.multivariate_normal(
            before.pairs.mean,
            before.pairs.covariance,
            size=before_indices.size,
        )
    if after_indices.size:
        differences[after_indices] = rng.multivariate_normal(
            after.pairs.mean,
            after.pairs.covariance,
            size=after_indices.size,
        )

    true_weights = (
        (1.0 - fractions[:, None]) * before.weights
        + fractions[:, None] * after.weights
    )

    centers = rng.normal(scale=center_scale, size=(horizon, dimension))
    x_a = centers + 0.5 * differences
    x_b = centers - 0.5 * differences
    true_probability = expit(np.einsum("ni,ni->n", true_weights, differences))
    outcomes = rng.binomial(1, true_probability).astype(np.int64)

    return PairStream(
        x_a=x_a,
        x_b=x_b,
        differences=differences,
        outcomes=outcomes,
        true_probability=true_probability,
        transition_fraction=fractions,
        true_weights=true_weights,
    )


def as_pair_distribution(mean: ArrayLike, covariance: ArrayLike) -> PairDistribution:
    """Construct a validated pair distribution from array-like inputs."""

    return PairDistribution(np.asarray(mean, dtype=float), np.asarray(covariance, dtype=float))
