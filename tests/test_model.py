import numpy as np
from scipy.special import expit

from preference_shift.model import LaplacePreferenceModel


def test_predictive_variance_decomposition() -> None:
    rng = np.random.default_rng(7)
    z = rng.normal(size=(1_500, 3))
    true_weights = np.array([0.8, -0.5, 0.3])
    y = rng.binomial(1, expit(z @ true_weights))

    model = LaplacePreferenceModel(prior_precision=1.0, quadrature_order=40).fit(z, y)
    moments = model.predict(z[:20])

    total = moments.epistemic_variance + moments.aleatoric_variance
    expected_total = moments.probability * (1.0 - moments.probability)
    np.testing.assert_allclose(total, expected_total, atol=1e-10)


def test_posterior_covariance_is_positive_definite() -> None:
    rng = np.random.default_rng(11)
    z = rng.normal(size=(500, 4))
    y = rng.binomial(1, 0.5, size=500)
    model = LaplacePreferenceModel().fit(z, y)

    eigenvalues = np.linalg.eigvalsh(model.posterior_covariance_)
    assert np.all(eigenvalues > 0)


def test_latent_uncertainty_is_larger_in_weakly_observed_direction() -> None:
    rng = np.random.default_rng(19)
    z = np.column_stack([rng.normal(size=2_000), 0.02 * rng.normal(size=2_000)])
    y = rng.binomial(1, expit(z @ np.array([1.0, 1.0])))
    model = LaplacePreferenceModel(prior_precision=0.2).fit(z, y)

    _, variance_first = model.latent_moments(np.array([1.0, 0.0]))
    _, variance_second = model.latent_moments(np.array([0.0, 1.0]))
    assert variance_second.item() > variance_first.item()
