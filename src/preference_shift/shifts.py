"""Geometrically controlled changes to comparison and preference parameters."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from preference_shift.simulation import PairDistribution

FloatArray = NDArray[np.float64]


def direction_at_angle(
    reference: ArrayLike,
    angle_degrees: float,
    rng: np.random.Generator,
) -> FloatArray:
    """Return a unit vector at a requested angle from a reference direction."""

    vector = _unit_vector(reference)
    if not 0.0 <= angle_degrees <= 90.0:
        raise ValueError("angle_degrees must lie in [0, 90]")
    if vector.size < 2 and angle_degrees != 0.0:
        raise ValueError("nonzero angles require at least two dimensions")

    orthogonal = rng.normal(size=vector.size)
    orthogonal -= (orthogonal @ vector) * vector
    norm = np.linalg.norm(orthogonal)
    while norm < 1e-12:
        orthogonal = rng.normal(size=vector.size)
        orthogonal -= (orthogonal @ vector) * vector
        norm = np.linalg.norm(orthogonal)
    orthogonal /= norm

    angle = np.deg2rad(angle_degrees)
    return np.cos(angle) * vector + np.sin(angle) * orthogonal


def rank_one_covariate_shift(
    base: PairDistribution,
    direction: ArrayLike,
    mean_magnitude: float = 0.0,
    covariance_magnitude: float = 0.0,
) -> PairDistribution:
    """Shift the pair mean and covariance along one controlled direction."""

    vector = _unit_vector(direction, expected_dimension=base.dimension)
    shifted_mean = base.mean + mean_magnitude * vector
    shifted_covariance = base.covariance + covariance_magnitude * np.outer(vector, vector)
    return PairDistribution(shifted_mean, shifted_covariance)


def shifted_weights(
    base_weights: ArrayLike,
    direction: ArrayLike,
    magnitude: float,
) -> FloatArray:
    """Apply an additive preference shift in a unit direction."""

    weights = np.asarray(base_weights, dtype=float)
    vector = _unit_vector(direction, expected_dimension=weights.size)
    return weights + magnitude * vector


def most_uncertain_direction(covariance: ArrayLike) -> FloatArray:
    """Return the eigenvector with largest posterior parameter variance."""

    matrix = np.asarray(covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance must be square")
    _, eigenvectors = np.linalg.eigh(matrix)
    direction = eigenvectors[:, -1]
    pivot = int(np.argmax(np.abs(direction)))
    return direction if direction[pivot] >= 0 else -direction


def _unit_vector(vector: ArrayLike, expected_dimension: int | None = None) -> FloatArray:
    result = np.asarray(vector, dtype=float)
    if result.ndim != 1:
        raise ValueError("direction must be a vector")
    if expected_dimension is not None and result.size != expected_dimension:
        raise ValueError("direction has the wrong dimension")
    norm = np.linalg.norm(result)
    if norm <= 0:
        raise ValueError("direction must be nonzero")
    return result / norm
