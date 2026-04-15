import unittest
from copy import copy

import numpy.testing as npt
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, pi
from pyrecest.distributions.circle.piecewise_constant_distribution import (
    PiecewiseConstantDistribution,
)
from pyrecest.distributions.circle.wrapped_normal_distribution import (
    WrappedNormalDistribution,
)
from pyrecest.filters.piecewise_constant_filter import PiecewiseConstantFilter


class TestPiecewiseConstantFilter(unittest.TestCase):
    def setUp(self):
        self.n = 10
        self.f = PiecewiseConstantFilter(self.n)

    def test_init(self):
        """Filter should start with a uniform distribution."""
        self.assertIsInstance(self.f.filter_state, PiecewiseConstantDistribution)
        self.assertEqual(len(self.f.filter_state.w), self.n)
        # All weights equal after normalization
        npt.assert_allclose(
            self.f.filter_state.w,
            self.f.filter_state.w[0] * array([1.0] * self.n),
            rtol=1e-10,
        )

    def test_set_state_pwc(self):
        """Setting state with a PiecewiseConstantDistribution should work directly."""
        w = array([1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        pwc = PiecewiseConstantDistribution(w)
        self.f.filter_state = pwc
        self.assertIsInstance(self.f.filter_state, PiecewiseConstantDistribution)
        npt.assert_allclose(self.f.filter_state.w, pwc.w, rtol=1e-10)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_set_state_other_distribution(self):
        """Setting state with a non-PWC circular distribution should convert it."""
        import warnings

        wn = WrappedNormalDistribution(array(1.0), array(0.5))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.f.filter_state = wn
            self.assertTrue(
                any(issubclass(warning.category, RuntimeWarning) for warning in w)
            )
        self.assertIsInstance(self.f.filter_state, PiecewiseConstantDistribution)
        self.assertEqual(len(self.f.filter_state.w), self.n)

    def test_predict_with_identity_matrix(self):
        """Predict with identity system matrix should leave the state unchanged."""
        import numpy as np

        w = array([1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        self.f.filter_state = PiecewiseConstantDistribution(w)
        w_before = copy(self.f.filter_state.w)

        A = array(np.eye(self.n))
        self.f.predict(A)

        npt.assert_allclose(self.f.filter_state.w, w_before, rtol=1e-10)

    def test_predict_changes_distribution(self):
        """Predict with a non-identity matrix should change the distribution."""
        import numpy as np

        w = array([1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        self.f.filter_state = PiecewiseConstantDistribution(w)
        w_before = copy(self.f.filter_state.w)

        # Shift all weight to next interval (cyclic permutation)
        A = array(np.roll(np.eye(self.n), 1, axis=0))
        self.f.predict(A)

        # The state should have changed
        self.assertFalse(
            bool((self.f.filter_state.w == w_before).all()),
            "predict() with permutation matrix should change the distribution",
        )

    def test_update_with_measurement_matrix(self):
        """Update with a measurement matrix should update the state."""
        import numpy as np

        # Start with a non-uniform distribution
        w = array([1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        self.f.filter_state = PiecewiseConstantDistribution(w)

        lw = 5
        H = array(np.random.default_rng(0).uniform(0.5, 1.5, (lw, self.n)))
        z = 0.5  # falls in measurement interval 0 (first interval [0, 2pi/5))

        w_before = copy(self.f.filter_state.w)
        self.f.update(H, z)

        # The updated weights should be proportional to H[0, :] * w_before
        row = int(z / (2.0 * float(pi)) * lw) % lw
        expected_raw = H[row, :] * w_before
        expected = PiecewiseConstantDistribution(expected_raw).w
        npt.assert_allclose(self.f.filter_state.w, expected, rtol=1e-10)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_update_likelihood(self):
        """update_likelihood should weight intervals by the integral of the likelihood."""
        wn = WrappedNormalDistribution(array(1.0), array(0.5))
        self.f.filter_state = PiecewiseConstantDistribution(
            array([1.0] * self.n, dtype=float)
        )

        # Likelihood peaked near x=1.0
        def likelihood(_, x):
            return float(wn.pdf(array([x])))

        self.f.update_likelihood(likelihood, 0.0)

        self.assertIsInstance(self.f.filter_state, PiecewiseConstantDistribution)
        # Distribution should integrate to 1
        n = len(self.f.filter_state.w)
        integral = float(sum(self.f.filter_state.w)) * (2.0 * float(pi) / n)
        npt.assert_allclose(integral, 1.0, rtol=1e-5)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_get_point_estimate(self):
        """get_point_estimate should return the mean direction."""
        w = array([1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        self.f.filter_state = PiecewiseConstantDistribution(w)
        estimate = self.f.get_point_estimate()
        self.assertIsNotNone(estimate)
        self.assertTrue(0.0 <= float(estimate) < 2.0 * float(pi))


if __name__ == "__main__":
    unittest.main()
