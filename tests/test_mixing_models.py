"""Tests for functionalcurves.mixing_models: transition functions and the
Markov/linear mixing simulators.
"""
import numpy as np
import pytest

from functionalcurves.mixing_models import (
    transition_markov,
    transition_diff,
    MixingMarkovModel,
    MixingLinearModel,
)


class TestTransitionFunctions:
    """X, p, e are positional-only (defined with a `/` marker), so they must
    be passed positionally rather than by keyword.
    """

    def test_transition_markov_zero_state_zero_noise(self):
        result = transition_markov(np.array([0.0, 0.0]), 1.0, np.array([0.0, 0.0]))
        np.testing.assert_allclose(result, [0.0, 0.0])

    def test_transition_markov_adds_innovation(self):
        X = np.array([0.0, 0.0])
        e = np.array([1.0, -1.0])
        result = transition_markov(X, 1.0, e)
        np.testing.assert_allclose(result, e)

    def test_transition_diff_default_rho(self):
        X = np.array([1.0, 1.0])
        e = np.array([0.0, 0.0])
        result = transition_diff(X, 1.0, e)
        np.testing.assert_allclose(result, 0.9 * X)


class TestMixingMarkovModel:
    def test_simulate_returns_expected_shape(self):
        model = MixingMarkovModel(mixing_rate=1.0)
        X = model.simulate(n_samples=50)
        assert X.shape == (50, 2)

    def test_simulate_starts_at_origin(self):
        model = MixingMarkovModel(mixing_rate=1.0)
        X = model.simulate(n_samples=10)
        np.testing.assert_allclose(X[0], [0.0, 0.0])

    def test_unsupported_innovation_raises(self):
        model = MixingMarkovModel(mixing_rate=1.0)
        with pytest.raises(NotImplementedError):
            model.simulate(innov="not_a_real_distribution", n_samples=5)


class TestMixingLinearModel:
    def test_simulate_returns_requested_length(self):
        model = MixingLinearModel(mixing_rate=4.0)
        X = model.simulate(n_samples=100, error=1e-3, verbose=0)
        assert X.shape[0] == 101  # includes the initial sample per range(0, n_samples+1)
        assert X.shape[1] == 2

    def test_distribution_matches_config_mean_scaling(self):
        model = MixingLinearModel(mixing_rate=4.0)
        model.simulate(n_samples=20, error=1e-2, verbose=0,
                        innov_config={"mean": [0, 0], "cov": [[1, 0], [0, 1]]})
        mean, cov = model.distribution()
        np.testing.assert_allclose(mean, [0.0, 0.0])
        assert cov.shape == (2, 2)

    def test_weights_are_decreasing(self):
        model = MixingLinearModel(mixing_rate=2.0)
        model.simulate(n_samples=50, error=1e-2, verbose=0)
        weights = model.weights
        assert np.all(np.diff(weights) <= 0)
