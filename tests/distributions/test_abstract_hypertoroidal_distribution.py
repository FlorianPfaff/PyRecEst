import inspect
import unittest

import matplotlib
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import array, ones, pi
from pyrecest.distributions import AbstractHypertoroidalDistribution
from pyrecest.distributions.circle.wrapped_normal_distribution import (
    WrappedNormalDistribution,
)
from pyrecest.distributions.hypertorus.toroidal_wrapped_normal_distribution import (
    ToroidalWrappedNormalDistribution,
)

matplotlib.pyplot.close("all")
matplotlib.use("Agg")


class ZeroMomentHypertoroidalDistribution(AbstractHypertoroidalDistribution):
    def __init__(self):
        super().__init__(dim=2)

    def pdf(self, xs):
        return ones(xs.shape[0])

    def trigonometric_moment(self, _):
        return array([0.0 + 0.0j, 1.0 + 0.0j])


class TestAbstractHypertoroidalDistribution(unittest.TestCase):
    def test_angular_error(self):
        npt.assert_allclose(
            AbstractHypertoroidalDistribution.angular_error(array(pi), array(0.0)), pi
        )
        npt.assert_allclose(
            AbstractHypertoroidalDistribution.angular_error(array(0), array(2 * pi)), 0
        )
        npt.assert_allclose(
            AbstractHypertoroidalDistribution.angular_error(
                array(pi / 4), array(7 * pi / 4)
            ),
            pi / 2,
            rtol=2e-07,
        )

    def test_angular_error_rejects_nan_inputs(self):
        with self.assertRaisesRegex(ValueError, "NaN"):
            AbstractHypertoroidalDistribution.angular_error(
                array(float("nan")), array(0.0)
            )

    def test_integrate_fun_over_domain_part_rejects_invalid_boundaries(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            AbstractHypertoroidalDistribution.integrate_fun_over_domain_part(
                lambda *_args: array(0.0), array([0.0, 1.0, 2.0])
            )

    def test_integrate_numerically_rejects_non_numpy_backend(self):
        dist = ZeroMomentHypertoroidalDistribution()
        original_backend_name = pyrecest.backend.__backend_name__
        pyrecest.backend.__backend_name__ = "jax"
        try:
            with self.assertRaisesRegex(NotImplementedError, "numpy backend"):
                dist.integrate_numerically()
        finally:
            pyrecest.backend.__backend_name__ = original_backend_name

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ != "numpy",
        reason="integrate_numerically validates boundaries only on the numpy backend",
    )
    def test_integrate_numerically_rejects_wrong_boundary_count(self):
        dist = ZeroMomentHypertoroidalDistribution()

        with self.assertRaisesRegex(ValueError, "one row per dimension"):
            dist.integrate_numerically(array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]))

    def test_numerical_distances_reject_invalid_partner(self):
        dist = ZeroMomentHypertoroidalDistribution()
        one_dimensional = WrappedNormalDistribution(array(0.0), array(1.0))

        for distance_fun in (
            dist.hellinger_distance_numerical,
            dist.total_variation_distance_numerical,
        ):
            with self.subTest(distance_fun=distance_fun.__name__):
                with self.assertRaisesRegex(
                    TypeError, "AbstractHypertoroidalDistribution"
                ):
                    distance_fun(object())

                with self.assertRaisesRegex(ValueError, "different number"):
                    distance_fun(one_dimensional)

    def test_mean_direction_rejects_zero_resultant_moment(self):
        dist = ZeroMomentHypertoroidalDistribution()

        with self.assertRaisesRegex(ValueError, "undefined"):
            dist.mean_direction()

    def test_setup_axis_circular_does_not_capture_axes_at_import(self):
        signature = inspect.signature(
            AbstractHypertoroidalDistribution.setup_axis_circular
        )

        self.assertIsNone(signature.parameters["ax"].default)

    def test_plot_2d(self):
        mu = array([0.0, 1.0])
        sigma1 = 0.5
        sigma2 = 0.5
        rho = 0.5
        dist = ToroidalWrappedNormalDistribution(
            mu,
            array([[sigma1, sigma1 * sigma2 * rho], [sigma1 * sigma2 * rho, sigma2]]),
        )
        dist.plot()
