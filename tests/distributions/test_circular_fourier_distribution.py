import copy
import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from parameterized import parameterized

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import arange, array, ceil, linspace, pi, sqrt
from pyrecest.distributions import (
    CircularFourierDistribution,
    VonMisesDistribution,
    WrappedNormalDistribution,
)
from scipy import integrate


class TestCircularFourierDistribution(unittest.TestCase):
    @parameterized.expand(
        [
            (
                "identity",
                VonMisesDistribution,
                array(0.4),
                arange(0.1, 2.1, 0.5),
                101,
            ),
            ("sqrt", VonMisesDistribution, array(0.5), arange(0.1, 2.1, 0.5), 101),
            (
                "identity",
                WrappedNormalDistribution,
                array(0.8),
                arange(0.2, 2.1, 0.5),
                101,
            ),
            (
                "sqrt",
                WrappedNormalDistribution,
                array(0.8),
                arange(0.2, 2.1, 0.5),
                101,
            ),
        ]
    )
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def test_fourier_conversion(
        self, transformation, dist_class, mu, param_range, coeffs
    ):
        """
        Test fourier conversion of the given distribution with varying parameter.
        """
        for param in param_range:
            dist = dist_class(mu, param)
            xvals = arange(-2.0 * pi, 3.0 * pi, 0.01)
            fd = CircularFourierDistribution.from_distribution(
                dist, coeffs, transformation
            )
            self.assertEqual(
                fd.c.shape[0],
                ceil(coeffs / 2.0),
                "Length of Fourier Coefficients mismatch.",
            )
            npt.assert_allclose(fd.pdf(xvals), dist.pdf(xvals), rtol=2e-3, atol=5e-5)

    @parameterized.expand(
        [
            (True, "identity"),
            (True, "sqrt"),
            (False, "identity"),
            (False, "sqrt"),
        ]
    )
    def test_vm_to_fourier(self, mult_by_n, transformation):
        xs = linspace(0.0, 2.0 * pi, 100)
        dist = VonMisesDistribution(2.5, 1.5)
        fd = CircularFourierDistribution.from_distribution(
            dist,
            n=31,
            transformation=transformation,
            store_values_multiplied_by_n=mult_by_n,
        )
        npt.assert_array_almost_equal(dist.pdf(xs), fd.pdf(xs))
        fd_real = fd.to_real_fd()
        npt.assert_array_almost_equal(dist.pdf(xs), fd_real.pdf(xs))

    def test_pdf_accepts_scalar_and_list_inputs(self):
        fd = CircularFourierDistribution(
            c=array([1.0]),
            n=1,
            transformation="identity",
            multiplied_by_n=False,
        )

        npt.assert_allclose(fd.pdf(0.5), fd.pdf(array([0.5])))
        npt.assert_allclose(fd.pdf([0.5, 1.0]), fd.pdf(array([0.5, 1.0])))

    def test_constructor_rejects_invalid_coefficient_sets(self):
        with self.assertRaisesRegex(ValueError, "a and b"):
            CircularFourierDistribution(
                a=array([1.0]),
                transformation="identity",
            )

        with self.assertRaisesRegex(ValueError, "either c or"):
            CircularFourierDistribution(
                a=array([1.0]),
                b=array([]),
                c=array([1.0]),
                transformation="identity",
            )

        with self.assertRaisesRegex(ValueError, "c must be one-dimensional"):
            CircularFourierDistribution(
                c=array([[1.0]]),
                n=1,
                transformation="identity",
            )

        with self.assertRaisesRegex(ValueError, "a must be one-dimensional"):
            CircularFourierDistribution(
                a=array([[1.0]]),
                b=array([]),
                transformation="identity",
            )

        with self.assertRaisesRegex(ValueError, "b must be one-dimensional"):
            CircularFourierDistribution(
                a=array([1.0]),
                b=array([[0.0]]),
                transformation="identity",
            )

        with self.assertRaisesRegex(ValueError, "one more coefficient"):
            CircularFourierDistribution(
                a=array([1.0, 2.0, 3.0]),
                b=array([]),
                transformation="identity",
            )

        with self.assertRaisesRegex(ValueError, "len\\(a\\) \\+ len\\(b\\)"):
            CircularFourierDistribution(
                a=array([1.0, 2.0]),
                b=array([0.0]),
                n=5,
                transformation="identity",
            )

        with self.assertRaisesRegex(ValueError, "coefficients for n=5"):
            CircularFourierDistribution(
                c=array([1.0, 0.0]),
                n=5,
                transformation="identity",
            )

    def test_pdf_rejects_more_than_two_dimensions(self):
        fd = CircularFourierDistribution(
            c=array([1.0]),
            n=1,
            transformation="identity",
            multiplied_by_n=False,
        )

        with self.assertRaisesRegex(ValueError, "at most 2 dimensions"):
            fd.pdf(array([[[0.0]]]))

    def test_even_number_of_grid_values_is_rejected(self):
        dist = VonMisesDistribution(2.5, 1.5)

        with self.assertRaisesRegex(ValueError, "requires an odd number"):
            CircularFourierDistribution.from_distribution(
                dist, n=64, transformation="identity"
            )

        with self.assertRaisesRegex(ValueError, "requires an odd number"):
            CircularFourierDistribution.from_function_values(
                array([1.0] * 64), transformation="identity"
            )

        with self.assertRaisesRegex(ValueError, "requires an odd number"):
            CircularFourierDistribution(
                c=array([1.0] * 33),
                n=64,
                transformation="identity",
                multiplied_by_n=True,
            )

        with self.assertRaisesRegex(ValueError, "requires an odd number"):
            CircularFourierDistribution(
                a=array([1.0] * 33),
                b=array([0.0] * 31),
                transformation="identity",
                multiplied_by_n=True,
            )

    def test_real_complex_coefficient_round_trip(self):
        a = array([2.0, 3.0, -5.0])
        b = array([7.0, -11.0])
        xs = linspace(0.0, 2.0 * pi, 64)

        fd_real = CircularFourierDistribution(
            a=a,
            b=b,
            n=5,
            transformation="identity",
            multiplied_by_n=False,
        )
        fd_complex = CircularFourierDistribution(
            c=fd_real.get_c(),
            n=5,
            transformation="identity",
            multiplied_by_n=False,
        )

        a_round_trip, b_round_trip = fd_complex.get_a_b()

        npt.assert_allclose(a_round_trip, a)
        npt.assert_allclose(b_round_trip, b)
        npt.assert_allclose(fd_complex.pdf(xs), fd_real.pdf(xs))

    def test_subtraction_rejects_incompatible_distributions(self):
        fd_real = CircularFourierDistribution(
            a=array([1.0, 0.0]),
            b=array([0.0]),
            n=3,
            transformation="identity",
            multiplied_by_n=False,
        )
        fd_complex = CircularFourierDistribution(
            c=fd_real.get_c(),
            n=3,
            transformation="identity",
            multiplied_by_n=False,
        )
        fd_sqrt = CircularFourierDistribution(
            a=array([1.0, 0.0]),
            b=array([0.0]),
            n=3,
            transformation="sqrt",
            multiplied_by_n=False,
        )
        fd_n5 = CircularFourierDistribution(
            a=array([1.0, 0.0, 0.0]),
            b=array([0.0, 0.0]),
            n=5,
            transformation="identity",
            multiplied_by_n=False,
        )
        fd_scaled = CircularFourierDistribution(
            a=array([1.0, 0.0]),
            b=array([0.0]),
            n=3,
            transformation="identity",
            multiplied_by_n=True,
        )

        with self.assertRaisesRegex(ValueError, "Either both"):
            fd_real - fd_complex

        with self.assertRaisesRegex(ValueError, "same transformation"):
            fd_real - fd_sqrt

        with self.assertRaisesRegex(ValueError, "same n"):
            fd_real - fd_n5

        with self.assertRaisesRegex(ValueError, "multiplied_by_n"):
            fd_real - fd_scaled

    def test_integrate_rejects_partial_boundaries(self):
        fd = CircularFourierDistribution(
            c=array([1.0]),
            n=1,
            transformation="identity",
            multiplied_by_n=False,
        )

        with self.assertRaisesRegex(NotImplementedError, "entire domain"):
            fd.integrate((0.0, 1.0))

    def test_getters_reject_inconsistent_storage(self):
        fd_real = CircularFourierDistribution(
            a=array([1.0, 0.0]),
            b=array([0.0]),
            n=3,
            transformation="identity",
            multiplied_by_n=False,
        )

        with self.assertRaisesRegex(ValueError, "only available when c is set"):
            fd_real.get_full_c()

        fd_complex = CircularFourierDistribution(
            c=array([1.0, 0.0, 0.0]),
            n=5,
            transformation="identity",
            multiplied_by_n=False,
        )
        fd_complex.n = 7

        with self.assertRaisesRegex(ValueError, "does not match n"):
            fd_complex.get_a_b()

    @parameterized.expand(
        [
            (True, "identity"),
            (False, "identity"),
            (True, "sqrt"),
            (False, "sqrt"),
        ]
    )
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_integrate_numerically(self, mult_by_n, transformation):
        scale_by = 2.0 / 5.0
        dist = VonMisesDistribution(2.9, 1.3)
        fd = CircularFourierDistribution.from_distribution(
            dist,
            n=31,
            transformation=transformation,
            store_values_multiplied_by_n=mult_by_n,
        )
        npt.assert_array_almost_equal(fd.integrate_numerically(), 1.0)
        fd_real = fd.to_real_fd()
        npt.assert_array_almost_equal(fd_real.integrate_numerically(), 1.0)
        fd_unnorm = copy.copy(fd)
        fd_unnorm.c = fd.c * (scale_by)
        if transformation == "identity":
            expected_val = scale_by
        else:
            expected_val = (scale_by) ** 2
        npt.assert_array_almost_equal(fd_unnorm.integrate_numerically(), expected_val)
        fd_unnorm_real = fd_unnorm.to_real_fd()
        npt.assert_array_almost_equal(
            fd_unnorm_real.integrate_numerically(), expected_val
        )

    @parameterized.expand(
        [
            (True, "identity"),
            (False, "identity"),
            (True, "sqrt"),
            (False, "sqrt"),
        ]
    )
    def test_integrate(self, mult_by_n, transformation):
        scale_by = 1 / 5
        dist = VonMisesDistribution(2.9, 1.3)
        fd = CircularFourierDistribution.from_distribution(
            dist,
            n=31,
            transformation=transformation,
            store_values_multiplied_by_n=mult_by_n,
        )
        npt.assert_array_almost_equal(fd.integrate(), 1.0)
        fd_real = fd.to_real_fd()
        npt.assert_array_almost_equal(fd_real.integrate(), 1.0)
        fd_unnorm = copy.copy(fd)
        fd_unnorm.c = fd.c * scale_by
        if transformation == "identity":
            expected_val = scale_by
        else:
            expected_val = scale_by**2
        npt.assert_array_almost_equal(fd_unnorm.integrate(), expected_val)
        fd_unnorm_real = fd_unnorm.to_real_fd()
        npt.assert_array_almost_equal(fd_unnorm_real.integrate(), expected_val)
        fd_unnorm = CircularFourierDistribution.from_distribution(
            dist,
            n=31,
            transformation=transformation,
            store_values_multiplied_by_n=mult_by_n,
        )
        fd_unnorm.c = fd_unnorm.c * scale_by
        fd_norm = fd_unnorm.normalize()
        fd_unnorm_real = fd_unnorm.to_real_fd()
        fd_norm_real = fd_unnorm_real.normalize()
        npt.assert_array_almost_equal(fd_norm.integrate(), 1.0)
        npt.assert_array_almost_equal(fd_norm_real.integrate(), 1.0)

    @parameterized.expand(
        [
            (False,),
            (True,),
        ]
    )
    def test_distance(self, mult_by_n):
        dist1 = VonMisesDistribution(array(0.0), array(1.0))
        dist2 = VonMisesDistribution(array(2.0), array(1.0))
        fd1 = CircularFourierDistribution.from_distribution(
            dist1,
            n=31,
            transformation="sqrt",
            store_values_multiplied_by_n=mult_by_n,
        )
        fd2 = CircularFourierDistribution.from_distribution(
            dist2,
            n=31,
            transformation="sqrt",
            store_values_multiplied_by_n=mult_by_n,
        )
        hel_like_distance, _ = integrate.quad(
            lambda x: (sqrt(dist1.pdf(array(x))) - sqrt(dist2.pdf(array(x)))) ** 2,
            0.0,
            2.0 * pi,
        )
        fd_diff = fd1 - fd2
        npt.assert_array_almost_equal(fd_diff.integrate(), hel_like_distance)


if __name__ == "__main__":
    unittest.main()
