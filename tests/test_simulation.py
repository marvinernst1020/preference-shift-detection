import numpy as np

from preference_shift.simulation import (
    PairDistribution,
    PreferenceRegime,
    simulate_transition,
)


def test_pair_construction_preserves_difference() -> None:
    regime = PreferenceRegime(
        pairs=PairDistribution(np.zeros(2), np.eye(2)),
        weights=np.array([1.0, -0.5]),
    )
    stream = simulate_transition(
        regime,
        regime,
        horizon=100,
        change_point=50,
        transition_width=0,
        rng=np.random.default_rng(3),
    )
    np.testing.assert_allclose(stream.x_a - stream.x_b, stream.differences)


def test_abrupt_transition_changes_only_after_change_point() -> None:
    before = PreferenceRegime(
        pairs=PairDistribution(np.zeros(2), np.eye(2)),
        weights=np.array([1.0, 0.0]),
    )
    after = PreferenceRegime(
        pairs=PairDistribution(np.ones(2), np.eye(2)),
        weights=np.array([0.0, 1.0]),
    )
    stream = simulate_transition(
        before,
        after,
        horizon=20,
        change_point=8,
        transition_width=0,
        rng=np.random.default_rng(5),
    )

    np.testing.assert_array_equal(stream.transition_fraction[:8], 0.0)
    np.testing.assert_array_equal(stream.transition_fraction[8:], 1.0)
    np.testing.assert_allclose(stream.true_weights[:8], np.tile(before.weights, (8, 1)))
    np.testing.assert_allclose(stream.true_weights[8:], np.tile(after.weights, (12, 1)))
