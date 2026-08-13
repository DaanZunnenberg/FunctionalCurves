"""Tests for functionalcurves.depth: angle helper, Gaussian/analytic depth,
and the empirical Estimator.
"""
import numpy as np
import pytest

from functionalcurves.depth import rad, GaussianDepth, Estimator, Analytic_Depth


class TestRad:
    def test_zero_angle_between_identical_vectors(self):
        assert rad(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(0.0, abs=1e-9)

    def test_quarter_turn(self):
        # v1=(1,0) -> v2=(0,1) is a +90 degree (pi/2) rotation
        angle = rad(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
        assert angle == pytest.approx(np.pi / 2, abs=1e-9)

    def test_half_turn(self):
        angle = rad(np.array([1.0, 0.0]), np.array([-1.0, 0.0]))
        assert angle == pytest.approx(np.pi, abs=1e-9)

    def test_negative_angles_wrapped_to_0_2pi(self):
        # v1=(0,1) -> v2=(1,0) is a -90 degree rotation, should wrap to 3pi/2
        angle = rad(np.array([0.0, 1.0]), np.array([1.0, 0.0]))
        assert angle == pytest.approx(3 * np.pi / 2, abs=1e-9)
        assert 0 <= angle <= 2 * np.pi

    def test_batched_input(self):
        v1 = np.array([[1.0, 0.0], [1.0, 0.0]])
        v2 = np.array([[0.0, 1.0], [-1.0, 0.0]])
        angles = rad(v1, v2)
        assert angles.shape == (2,)
        np.testing.assert_allclose(angles, [np.pi / 2, np.pi], atol=1e-9)


class TestGaussianDepth:
    def test_depth_close_to_mean_is_close_to_one_half(self):
        # For a symmetric (e.g. standard normal) distribution, the halfspace
        # depth approaches 1/2 as X0 approaches the mean in any direction.
        # X0 == mean exactly is a singular case (zero-norm direction vector),
        # so we probe a point very close to, but not exactly at, the mean.
        depth, direction, vector = GaussianDepth(mean=[0, 0], cov=[[1, 0], [0, 1]], X0=[0, 1e-6])
        assert depth == pytest.approx(0.5, abs=1e-3)

    def test_depth_decreases_further_from_mean(self):
        depth_near, _, _ = GaussianDepth(mean=[0, 0], cov=[[1, 0], [0, 1]], X0=[0, 0.5])
        depth_far, _, _ = GaussianDepth(mean=[0, 0], cov=[[1, 0], [0, 1]], X0=[0, 2.0])
        assert 0 <= depth_far < depth_near <= 0.5

    def test_returns_unit_minimal_vector(self):
        _, _, vector = GaussianDepth(mean=[0, 0], cov=[[1, 0], [0, 1]], X0=[1, 1])
        assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-9)


class TestAnalyticDepth:
    def test_matches_gaussian_depth_for_stationary_var1(self):
        """A VAR(1) with A1=0 degenerates to i.i.d. Gaussian noise, so the
        analytic depth should match the closed-form GaussianDepth at the
        same (mean, covariance, point).
        """
        A0 = np.array([[0.0], [0.0]])
        A1 = np.zeros((2, 2))
        S0 = np.eye(2)
        X0 = np.array([0.5, -0.3])

        X_dummy = np.zeros((2, 2))  # only .shape[1] is used before mean/variance override
        ad = Analytic_Depth(X=X_dummy, A0=A0, A1=A1, S0=S0, X0=X0)

        analytic_depth, analytic_direction = ad.TD_analytic()
        gauss_depth, _, _ = GaussianDepth(mean=[0, 0], cov=[[1, 0], [0, 1]], X0=X0)

        assert analytic_depth == pytest.approx(gauss_depth, abs=1e-6)

    def test_stationary_mean_zero_for_zero_intercept(self):
        A0 = np.array([[0.0], [0.0]])
        A1 = np.array([[0.5, 0.0], [0.0, 0.3]])
        S0 = np.eye(2)
        X0 = np.array([1.0, 1.0])
        X_dummy = np.zeros((2, 2))

        ad = Analytic_Depth(X=X_dummy, A0=A0, A1=A1, S0=S0, X0=X0)
        np.testing.assert_allclose(ad.mean.flatten(), [0.0, 0.0], atol=1e-9)


class TestEstimator:
    @pytest.fixture
    def symmetric_sample(self):
        # A large symmetric sample around the origin so the empirical depth
        # at the origin should be close to the theoretical 0.5.
        rng = np.random.default_rng(0)
        return rng.normal(size=(4000, 2))

    def test_point_wise_depth_near_origin_is_close_to_half(self, symmetric_sample):
        est = Estimator(X=symmetric_sample, X0=(0, 0), method="point_wise")
        min_depth, min_dir, angles, depths, directions = est.main()
        assert 0.3 < min_depth <= 0.5

    def test_deg_method_runs_and_returns_consistent_shapes(self, symmetric_sample):
        est = Estimator(X=symmetric_sample, X0=(0, 0), method="deg", num=200)
        min_depth, min_dir, angles, depths, directions = est.main()
        assert len(angles) == len(depths) == len(directions) == 200
        assert 0 <= min_depth <= 1

    def test_invalid_method_raises(self, symmetric_sample):
        est = Estimator(X=symmetric_sample, X0=(0, 0), method="bogus")
        with pytest.raises(ValueError):
            est.main()
