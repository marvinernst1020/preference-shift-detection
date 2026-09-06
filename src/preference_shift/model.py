"""Bayesian linear Bradley-Terry model with a Laplace posterior."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import expit

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PredictiveMoments:
    """Posterior predictive quantities for one or more pair differences."""

    probability: FloatArray
    latent_mean: FloatArray
    latent_variance: FloatArray
    epistemic_variance: FloatArray
    aleatoric_variance: FloatArray


class LaplacePreferenceModel:
    """Bayesian logistic preference model with an isotropic Gaussian prior.

    The model is fitted once on the stationary reference sample. It remains
    frozen during monitoring so that detection and adaptation are not mixed.
    """

    def __init__(self, prior_precision: float = 1.0, quadrature_order: int = 30) -> None:
        if prior_precision <= 0:
            raise ValueError("prior_precision must be positive")
        if quadrature_order < 5:
            raise ValueError("quadrature_order must be at least 5")

        self.prior_precision = float(prior_precision)
        self.quadrature_order = int(quadrature_order)
        self.posterior_mean_: FloatArray | None = None
        self.posterior_covariance_: FloatArray | None = None

    def fit(self, differences: ArrayLike, outcomes: ArrayLike) -> LaplacePreferenceModel:
        """Fit the posterior mode and Laplace covariance."""

        z = _as_design_matrix(differences)
        y = np.asarray(outcomes, dtype=float)
        if y.ndim != 1 or y.shape[0] != z.shape[0]:
            raise ValueError("outcomes must be a vector with one entry per comparison")
        if not np.all(np.isin(y, (0.0, 1.0))):
            raise ValueError("outcomes must contain only zero and one")
        if not np.all(np.isfinite(z)):
            raise ValueError("differences must be finite")

        dimension = z.shape[1]
        precision = self.prior_precision

        def objective(weights: FloatArray) -> float:
            logits = z @ weights
            likelihood = np.logaddexp(0.0, logits).sum() - y @ logits
            prior = 0.5 * precision * (weights @ weights)
            return float(likelihood + prior)

        def gradient(weights: FloatArray) -> FloatArray:
            probabilities = expit(z @ weights)
            return z.T @ (probabilities - y) + precision * weights

        result = minimize(
            objective,
            x0=np.zeros(dimension, dtype=float),
            jac=gradient,
            method="L-BFGS-B",
            options={"maxiter": 2_000, "ftol": 1e-12, "gtol": 1e-9},
        )
        if not result.success:
            raise RuntimeError(f"posterior optimization failed: {result.message}")

        mean = np.asarray(result.x, dtype=float)
        probabilities = expit(z @ mean)
        curvature = probabilities * (1.0 - probabilities)
        hessian = precision * np.eye(dimension) + z.T @ (curvature[:, None] * z)
        factor = cho_factor(hessian, lower=True, check_finite=True)
        covariance = cho_solve(factor, np.eye(dimension), check_finite=True)

        self.posterior_mean_ = mean
        self.posterior_covariance_ = 0.5 * (covariance + covariance.T)
        return self

    def latent_moments(self, differences: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Return posterior mean and variance of the latent utility difference."""

        mean, covariance = self._fitted_parameters()
        z = _as_design_matrix(differences, expected_dimension=mean.size)
        latent_mean = z @ mean
        latent_variance = np.einsum("ni,ij,nj->n", z, covariance, z)
        return latent_mean, np.maximum(latent_variance, 0.0)

    def predict(self, differences: ArrayLike) -> PredictiveMoments:
        """Compute Bayesian predictive probability and uncertainty decomposition."""

        latent_mean, latent_variance = self.latent_moments(differences)
        nodes, weights = hermgauss(self.quadrature_order)

        latent_draws = latent_mean[:, None] + np.sqrt(2.0 * latent_variance[:, None]) * nodes
        conditional_probabilities = expit(latent_draws)
        normalized_weights = weights / np.sqrt(np.pi)

        probability = conditional_probabilities @ normalized_weights
        second_moment = (conditional_probabilities**2) @ normalized_weights
        epistemic = np.maximum(second_moment - probability**2, 0.0)
        aleatoric = (conditional_probabilities * (1.0 - conditional_probabilities)) @ (
            normalized_weights
        )

        return PredictiveMoments(
            probability=probability,
            latent_mean=latent_mean,
            latent_variance=latent_variance,
            epistemic_variance=epistemic,
            aleatoric_variance=aleatoric,
        )

    @property
    def posterior_mean(self) -> FloatArray:
        """Posterior mode used as the Laplace mean."""

        mean, _ = self._fitted_parameters()
        return mean.copy()

    @property
    def posterior_covariance(self) -> FloatArray:
        """Laplace approximation to the posterior covariance."""

        _, covariance = self._fitted_parameters()
        return covariance.copy()

    def _fitted_parameters(self) -> tuple[FloatArray, FloatArray]:
        if self.posterior_mean_ is None or self.posterior_covariance_ is None:
            raise RuntimeError("fit must be called before prediction")
        return self.posterior_mean_, self.posterior_covariance_


def _as_design_matrix(
    differences: ArrayLike,
    expected_dimension: int | None = None,
) -> FloatArray:
    z = np.asarray(differences, dtype=float)
    if z.ndim == 1:
        z = z[None, :]
    if z.ndim != 2:
        raise ValueError("differences must be a one- or two-dimensional array")
    if expected_dimension is not None and z.shape[1] != expected_dimension:
        raise ValueError(f"expected comparison dimension {expected_dimension}, got {z.shape[1]}")
    return z
