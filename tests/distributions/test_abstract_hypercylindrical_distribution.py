import unittest

import matplotlib
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    allclose,
    arange,
    array,
    column_stack,
    diff,
    ones,
    pi,
    zeros,
)
from pyrecest.distributions.cart_prod.abstract_hypercylindrical_distribution import (
    AbstractHypercylindricalDistribution,
)
from pyrecest.distributions.cart_prod.partially_wrapped_normal_distribution import (
    PartiallyWrappedNormalDistribution,
)


class DummyHypercylindricalDistribution(AbstractHypercylindricalDistribution):
    def __init__(self, bound_dim=1, lin_dim=1):
        super().__init__(bound_dim, lin_dim)

    def pdf(self, xs):
        xs = array(xs)
        return ones(1 if xs.ndim == 1 else xs.shape[0])

    def marginalize_linear(self):
        raise NotImplementedError

    def marginalize_periodic(self):
        raise NotImplementedError


class AbstractHypercylindricalDistributionTest(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on jax backend",
    )
    def test_mode_numerical_gaussian_2D(self):
        mu = array([5.0, 1.0])
        C = array([[2.0, 1.0], [1.0, 1.0]])
        g = PartiallyWrappedNormalDistribution(mu, C, 1)
        npt.assert_allclose(g.mode_numerical(), mu, atol=5e-5)

    def test_get_reasonable_integration_boundaries(self):
        hwn = PartiallyWrappedNormalDistribution(
            array([1.0, 2.0]), array([[2.0, 0.3], [0.3, 4.0]]), 1
        )

        integration_boundaries = hwn.get_reasonable_integration_boundaries(
            scalingFactor=3
        )

        self.assertEqual(len(integration_boundaries), hwn.input_dim)
        npt.assert_allclose(
            integration_boundaries, array([[0.0, 2.0 * pi], [-4.0, 8.0]])
        )

    def test_constructor_accepts_fully_periodic_dimension(self):
        dist = PartiallyWrappedNormalDistribution(array([1.0]), array([[1.0]]), 1)

        self.assertEqual(dist.bound_dim, 1)
        self.assertEqual(dist.lin_dim, 0)

    def test_linear_mean_numerical(self):
        hwn = PartiallyWrappedNormalDistribution(
            array([1.0, 2.0]), array([[2.0, 0.3], [0.3, 1.0]]), 1
        )
        npt.assert_allclose(hwn.linear_mean_numerical(), hwn.mu[-1])

    def test_linear_covariance_rejects_wrong_mean_dimension(self):
        dist = DummyHypercylindricalDistribution()

        with self.assertRaisesRegex(ValueError, "lin_dim"):
            dist.linear_covariance(array([0.0, 1.0]))

    def test_condition_on_linear_rejects_wrong_input_dimension(self):
        hwn = PartiallyWrappedNormalDistribution(
            array([1.0, 2.0]), array([[2.0, 0.3], [0.3, 1.0]]), 1
        )

        with self.assertRaisesRegex(ValueError, "lin_dim"):
            hwn.condition_on_linear(array([1.0, 2.0]), normalize=False)

    def test_condition_on_periodic_rejects_wrong_input_dimension(self):
        hwn = PartiallyWrappedNormalDistribution(
            array([1.0, 2.0, 3.0]),
            array([[2.0, 0.3, 0.1], [0.3, 1.5, 0.2], [0.1, 0.2, 1.0]]),
            2,
        )

        with self.assertRaisesRegex(ValueError, "bound_dim"):
            hwn.condition_on_periodic(array([1.0]), normalize=False)

    def test_mode_numerical_rejects_unsupported_backend(self):
        hwn = PartiallyWrappedNormalDistribution(
            array([1.0, 2.0]), array([[2.0, 0.3], [0.3, 1.0]]), 1
        )
        original_backend_name = pyrecest.backend.__backend_name__
        pyrecest.backend.__backend_name__ = "jax"
        try:
            with self.assertRaisesRegex(NotImplementedError, "backend"):
                hwn.mode_numerical()
        finally:
            pyrecest.backend.__backend_name__ = original_backend_name

    def test_plot_cylinder_rejects_unsupported_dimensions(self):
        hwn = PartiallyWrappedNormalDistribution(
            array([1.0, 2.0, 3.0]),
            array([[2.0, 0.3, 0.1], [0.3, 1.5, 0.2], [0.1, 0.2, 1.0]]),
            2,
        )

        with self.assertRaisesRegex(NotImplementedError, "bound_dim == 1"):
            hwn.plot_cylinder()

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_condition_on_periodic(self):
        hwn = PartiallyWrappedNormalDistribution(
            array([1.0, 2.0]), array([[2.0, 0.3], [0.3, 1.0]]), 1
        )
        dist_cond1 = hwn.condition_on_periodic(array(1.5))
        # There is some normalization constant involved, therefore, test if ratio stays the same
        npt.assert_allclose(
            diff(
                hwn.pdf(column_stack([1.5 * ones(11), arange(-5, 6)]))
                / dist_cond1.pdf(arange(-5, 6))
            ),
            zeros(10),
            atol=1e-10,
        )
        dist_cond2 = hwn.condition_on_periodic(array(1.5) + 2.0 * pi)
        npt.assert_allclose(
            diff(
                hwn.pdf(column_stack([1.5 * ones(11), arange(-5, 6)]))
                / dist_cond2.pdf(arange(-5, 6))
            ),
            zeros(10),
            atol=1e-10,
        )

    def test_condition_on_periodic_accepts_periodic_vectors(self):
        hwn = PartiallyWrappedNormalDistribution(
            array([1.0, 2.0, 3.0]),
            array([[2.0, 0.3, 0.1], [0.3, 1.5, 0.2], [0.1, 0.2, 1.0]]),
            2,
        )
        input_periodic = array([1.5, 2.5])
        input_linear = arange(-2, 3)

        dist_cond = hwn.condition_on_periodic(input_periodic, normalize=False)

        npt.assert_allclose(
            dist_cond.pdf(input_linear),
            hwn.pdf(
                column_stack(
                    [
                        input_periodic[0] * ones(input_linear.shape[0]),
                        input_periodic[1] * ones(input_linear.shape[0]),
                        input_linear,
                    ]
                )
            ),
        )

    def test_condition_helpers_accept_python_lists(self):
        hwn = PartiallyWrappedNormalDistribution(
            array([1.0, 2.0]), array([[2.0, 0.3], [0.3, 1.0]]), 1
        )
        input_periodic = [1.5]
        input_linear = [2.5]
        query_points = arange(-2, 3)

        periodic_cond = hwn.condition_on_periodic(input_periodic, normalize=False)
        linear_cond = hwn.condition_on_linear(input_linear, normalize=False)

        npt.assert_allclose(
            periodic_cond.pdf(query_points),
            hwn.pdf(column_stack([input_periodic[0] * ones(5), query_points])),
        )
        npt.assert_allclose(
            linear_cond.pdf(query_points),
            hwn.pdf(column_stack([query_points, input_linear[0] * ones(5)])),
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_plot(self):
        matplotlib.pyplot.close("all")
        matplotlib.use("Agg")
        hwn = PartiallyWrappedNormalDistribution(
            array([1.0, 2.0]), array([[2.0, 0.3], [0.3, 1.0]]), 1
        )
        hwn.plot()
        hwn.plot_cylinder()

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_condition_on_linear(self):
        hwn = PartiallyWrappedNormalDistribution(
            array([1.0, 2.0]), array([[2.0, 0.3], [0.3, 1.0]]), 1
        )
        dist_cond1 = hwn.condition_on_linear(array(1.5))
        npt.assert_allclose(
            diff(
                hwn.pdf(column_stack([arange(-5, 6), 1.5 * ones(11)]))
                / dist_cond1.pdf(arange(-5, 6))
            ),
            zeros(10),
            atol=1e-10,
        )
        dist_cond2 = hwn.condition_on_linear(array(1.5 + 2.0 * pi))
        self.assertFalse(
            (
                allclose(
                    diff(
                        hwn.pdf(column_stack([arange(-5, 6), 1.5 * ones(11)]))
                        / dist_cond2.pdf(arange(-5, 6))
                    ),
                    zeros(10),
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
