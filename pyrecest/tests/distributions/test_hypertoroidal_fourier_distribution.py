import unittest
import warnings

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from parameterized import parameterized

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import (
    array,
    column_stack,
    diag,
    fft,
    linspace,
    meshgrid,
    pi,
    random,
    sin,
    sqrt,
    sum,
    zeros,
)
from pyrecest.distributions import (
    HypertoroidalWrappedNormalDistribution,
    ToroidalWrappedNormalDistribution,
)
from pyrecest.distributions import WrappedNormalDistribution as WNDistribution
from pyrecest.distributions.hypertorus.hypertoroidal_fourier_distribution import (
    HypertoroidalFourierDistribution,
)


def integrate2d(hfd, N=100):
    x = linspace(0, 2 * pi, N, endpoint=False) + (pi / N)
    y = linspace(0, 2 * pi, N, endpoint=False) + (pi / N)
    X, Y = meshgrid(x, y, indexing="ij")
    pts = column_stack((X.flatten(), Y.flatten()))
    p = hfd.pdf(pts)
    area = (2 * pi / N) ** 2
    return sum(p) * area


def integrate3d(hfd, N=30):
    x = linspace(0, 2 * pi, N, endpoint=False) + (pi / N)
    y = linspace(0, 2 * pi, N, endpoint=False) + (pi / N)
    z = linspace(0, 2 * pi, N, endpoint=False) + (pi / N)
    X, Y, Z = meshgrid(x, y, z, indexing="ij")
    pts = column_stack((X.flatten(), Y.flatten(), Z.flatten()))
    p = hfd.pdf(pts)
    vol = (2 * pi / N) ** 3
    return sum(p) * vol


class HypertoroidalFourierDistributionTest(unittest.TestCase):
    def test_constructor_and_single_coeff_warning(self):
        # 2D coefficient matrix, already normalized for 'sqrt'
        coeffs = array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0 / (sqrt(2 * pi) ** 2), 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        hfd = HypertoroidalFourierDistribution(coeffs, "sqrt")
        self.assertEqual(hfd.coeff_mat.shape, (3, 3))
        self.assertEqual(hfd.transformation, "sqrt")

        # Single coefficient -> should emit a warning, like MATLAB testConstructor
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            HypertoroidalFourierDistribution(1.0 / sqrt(2 * pi), "sqrt")
        self.assertTrue(
            any("fourierCoefficients:singleCoefficient" in str(wi.message) for wi in w)
        )

    @parameterized.expand(
        [
            ("2d_identity", "identity", (3, 7), (1, 3)),
            ("2d_sqrt", "sqrt", (3, 7), (1, 3)),
            ("3d_identity", "identity", (3, 11, 7), (1, 5, 3)),
            ("3d_sqrt", "sqrt", (3, 11, 7), (1, 5, 3)),
        ]
    )
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_normalization_nd(self, _, transform, shape, index):
        random.seed(0)
        arr = random.uniform(size=shape) + 0.5
        unnormalizedCoeffs = fft.fftshift(fft.fftn(arr))
        unnormalizedCoeffs[index] = 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            hfd = HypertoroidalFourierDistribution(
                unnormalizedCoeffs, transformation=transform
            )
        if hfd.dim == 2:
            val = integrate2d(hfd, N=30)
        elif hfd.dim == 3:
            val = integrate3d(hfd, N=30)
        else:
            raise NotImplementedError("Only 2D and 3D tests are implemented.")

        npt.assert_allclose(val, 1.0, atol=1e-4)

    def test_constructor(self):
        # Create a 3x3 coefficient matrix
        coeffs = array([[0, 0, 0], [0, 1 / (sqrt(2 * pi) ** 2), 0], [0, 0, 0]])
        hfd = HypertoroidalFourierDistribution(coeffs, "sqrt")
        self.assertEqual(hfd.coeff_mat.shape, (3, 3))
        self.assertEqual(hfd.transformation, "sqrt")

    @staticmethod
    def _sine_function(x, y):
        return 1 / (4 * pi**2) + 0.01 * sin(x) * sin(y)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_sine_function_identity(self):
        # Can be represented precisely using coefficients
        coeffs = (5, 5)
        hfd = HypertoroidalFourierDistribution.from_function(
            HypertoroidalFourierDistributionTest._sine_function,
            coeffs,
            desired_transformation="identity",
        )
        self.assertIsInstance(hfd, HypertoroidalFourierDistribution)
        self.assertEqual(hfd.coeff_mat.shape, coeffs)

        expected_coeffs = zeros((5, 5), dtype=complex)
        expected_coeffs[2, 2] = 1 / (4 * pi**2)  # constant term
        expected_coeffs[3, 3] = -0.0025  # (1,1)
        expected_coeffs[3, 1] = +0.0025  # (1,-1)
        expected_coeffs[1, 3] = +0.0025  # (-1,1)
        expected_coeffs[1, 1] = -0.0025  # (-1,-1)

        npt.assert_allclose(hfd.coeff_mat, expected_coeffs, atol=1e-5)

    def test_sine_function_sqrt(self):
        coeffs = (9, 9)
        hfd = HypertoroidalFourierDistribution.from_function(
            HypertoroidalFourierDistributionTest._sine_function,
            coeffs,
            desired_transformation="sqrt",
        )
        self.assertIsInstance(hfd, HypertoroidalFourierDistribution)
        self.assertEqual(hfd.coeff_mat.shape, coeffs)

        x, y = meshgrid(linspace(0, 2 * pi, 5), linspace(0, 2 * pi, 5), indexing="ij")
        pts = column_stack((x.flatten(), y.flatten()))
        pdf_vals = hfd.pdf(pts)

        expected_pdf_vals = HypertoroidalFourierDistributionTest._sine_function(
            x.flatten(), y.flatten()
        )

        npt.assert_allclose(pdf_vals, expected_pdf_vals, atol=1e-5)

    @parameterized.expand(
        [
            ("identity", "identity"),
            ("sqrt", "sqrt"),
        ]
    )
    def test_from_distribution_1d(self, _, transform):
        wn = WNDistribution(array(1.0), array(1.0))
        hfd = HypertoroidalFourierDistribution.from_distribution(wn, (101,), transform)
        xvals = linspace(0, 2 * pi, 100, endpoint=False)
        p1 = wn.pdf(xvals)
        p2 = hfd.pdf(xvals)
        npt.assert_allclose(p1, p2, atol=5e-5)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_truncation_1d(self):
        wn = WNDistribution(array(1.0), array(1.0))
        hfd1 = HypertoroidalFourierDistribution.from_distribution(wn, (101,))
        hfd2 = hfd1.truncate((51,))
        self.assertEqual(hfd2.coeff_mat.shape, (51,))
        xvals = linspace(0, 2 * pi, 100, endpoint=False)
        p1 = hfd1.pdf(xvals)
        p2 = hfd2.pdf(xvals)
        npt.assert_allclose(p1, p2, atol=1e-8)

    @parameterized.expand(
        [
            ("identity", "identity"),
            ("sqrt", "sqrt"),
        ]
    )
    def test_from_function_2d(self, _, transform):
        xTest, yTest = meshgrid(
            linspace(0, 2 * pi, 10), linspace(0, 2 * pi, 10), indexing="ij"
        )
        coeffs = (17, 19)
        # Loop over some parameter combinations
        mu = array([1.0, 0.0])
        sigma1 = 1.0
        sigma2 = 2.0
        rho = 0.5
        dist = ToroidalWrappedNormalDistribution(
            mu,
            array([[sigma1, sigma1 * sigma2 * rho], [sigma1 * sigma2 * rho, sigma2]]),
        )

        # Define a function f(x,y) that returns the pdf of tvm on a grid.
        def f_from_function(x, y):
            # x and y are arrays of the same shape.
            pts = column_stack((x.flatten(), y.flatten()))
            return dist.pdf(pts).flatten()

        hfd = HypertoroidalFourierDistribution.from_function(
            f_from_function, coeffs, desired_transformation=transform
        )
        self.assertIsInstance(hfd, HypertoroidalFourierDistribution)
        self.assertEqual(hfd.coeff_mat.shape, coeffs)

        pts_test = column_stack((xTest.flatten(), yTest.flatten()))
        orig_pdf_vals = dist.pdf(pts_test)
        approx_pdf_vals = hfd.pdf(pts_test)
        npt.assert_allclose(orig_pdf_vals, approx_pdf_vals, atol=5e-5)
        with self.assertRaises(ValueError):
            HypertoroidalFourierDistribution.from_function(
                f_from_function, coeffs, "abc"
            )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_truncation_3d(self):
        coeffs_shape = (15, 15, 15)
        mu = array([0.0, 0.0, 0.0])
        C = array([[0.7, 0.4, 0.2], [0.4, 0.6, 0.1], [0.2, 0.1, 1.0]])
        hwnd = HypertoroidalWrappedNormalDistribution(mu, C)

        # Identity transformation
        hfd_id = HypertoroidalFourierDistribution.from_distribution(
            hwnd, coeffs_shape, "identity"
        )
        hfd_id_trunc = hfd_id.truncate((3, 3, 3))

        # Verify integration of truncated version
        self.assertAlmostEqual(integrate3d(hfd_id_trunc, N=20), 1.0, delta=1e-3)

        # Sqrt transformation
        hfd_sqrt = HypertoroidalFourierDistribution.from_distribution(
            hwnd, coeffs_shape, "sqrt"
        )
        hfd_sqrt_trunc = hfd_sqrt.truncate((3, 3, 3))

        # Verify integration
        self.assertAlmostEqual(integrate3d(hfd_sqrt_trunc, N=20), 1.0, delta=1e-3)

    def test_from_function_3d(self):
        random.seed(42)
        test_points = random.uniform(size=(100, 3)) * 2 * pi

        C = array([[0.7, 0.4, 0.2], [0.4, 0.6, 0.1], [0.2, 0.1, 1.0]])
        mu = random.uniform(size=(3,)) * 2 * pi
        hwnd = HypertoroidalWrappedNormalDistribution(mu, C)

        coeffs = (25, 27, 23)

        def func_3d(x, y, z):
            pts = column_stack((x.flatten(), y.flatten(), z.flatten()))
            return hwnd.pdf(pts).flatten()

        hfd_id = HypertoroidalFourierDistribution.from_function(
            func_3d, coeffs, "identity"
        )
        hfd_sqrt = HypertoroidalFourierDistribution.from_function(
            func_3d, coeffs, "sqrt"
        )

        npt.assert_array_equal(hfd_id.coeff_mat.shape, coeffs)
        npt.assert_allclose(hfd_id.pdf(test_points), hwnd.pdf(test_points), atol=1e-5)
        npt.assert_allclose(hfd_sqrt.pdf(test_points), hwnd.pdf(test_points), atol=1e-5)

    def test_from_function_4d(self):
        random.seed(42)
        # Using a slightly smaller coefficient set for speed in unit test
        C = (
            array(
                [
                    [0.7, 0.4, 0.2, -0.5],
                    [0.4, 0.6, 0.1, 0],
                    [0.2, 0.1, 1, -0.3],
                    [-0.5, 0, -0.3, 0.9],
                ]
            )
            * 2
        )
        mu = random.uniform(size=(4,)) * 2 * pi
        hwnd = HypertoroidalWrappedNormalDistribution(mu, C)

        coeffs = (11, 11, 11, 11)

        def func_4d(x, y, z, w):
            pts = column_stack((x.flatten(), y.flatten(), z.flatten(), w.flatten()))
            return hwnd.pdf(pts).flatten()

        hfd_id = HypertoroidalFourierDistribution.from_function(
            func_4d, coeffs, "identity"
        )
        hfd_sqrt = HypertoroidalFourierDistribution.from_function(
            func_4d, coeffs, "sqrt"
        )

        test_points = random.uniform(size=(50, 4)) * 2 * pi

        npt.assert_array_equal(hfd_id.coeff_mat.shape, coeffs)
        npt.assert_allclose(hfd_id.pdf(test_points), hwnd.pdf(test_points), atol=1e-4)
        npt.assert_allclose(hfd_sqrt.pdf(test_points), hwnd.pdf(test_points), atol=1e-4)

    def test_from_distribution_3d(self):
        C = array([[0.7, 0.4, 0.2], [0.4, 0.6, 0.1], [0.2, 0.1, 1.0]])
        mu = array([3.0, 5.0, 2.0])
        coeffs = (21, 21, 21)
        hwnd = HypertoroidalWrappedNormalDistribution(mu, C)

        # From Function (Ground Truth Approximation)
        def func_3d(x, y, z):
            pts = column_stack((x.flatten(), y.flatten(), z.flatten()))
            return hwnd.pdf(pts).flatten()

        hfd_func = HypertoroidalFourierDistribution.from_function(
            func_3d, coeffs, "identity"
        )

        # From Distribution
        hfd_dist = HypertoroidalFourierDistribution.from_distribution(
            hwnd, coeffs, "identity"
        )

        npt.assert_allclose(hfd_dist.coeff_mat, hfd_func.coeff_mat, atol=1e-8)

    def test_from_distribution_hwn_closed_form(self):
        # 2D Case
        mu = array([1.0, 2.0])
        C = 2.0 * array([[1.0, 0.5], [0.5, 1.3]])
        hwn = HypertoroidalWrappedNormalDistribution(mu, C)
        coeffs = (35, 35)

        hfd_dist = HypertoroidalFourierDistribution.from_distribution(
            hwn, coeffs, "identity"
        )

        def func_2d(x, y):
            pts = column_stack((x.flatten(), y.flatten()))
            return hwn.pdf(pts).flatten()

        hfd_func = HypertoroidalFourierDistribution.from_function(
            func_2d, coeffs, "identity"
        )

        npt.assert_allclose(hfd_dist.coeff_mat, hfd_func.coeff_mat, atol=1e-7)

        # 3D Case
        mu_3d = array([1.0, 2.0, 4.0])
        C_3d = array([[0.7, 0.4, 0.2], [0.4, 0.6, 0.1], [0.2, 0.1, 1.0]])
        hwn_3d = HypertoroidalWrappedNormalDistribution(mu_3d, C_3d)
        coeffs_3d = (19, 19, 19)

        hfd_dist_3d = HypertoroidalFourierDistribution.from_distribution(
            hwn_3d, coeffs_3d, "identity"
        )

        def func_3d(x, y, z):
            pts = column_stack((x.flatten(), y.flatten(), z.flatten()))
            return hwn_3d.pdf(pts).flatten()

        hfd_func_3d = HypertoroidalFourierDistribution.from_function(
            func_3d, coeffs_3d, "identity"
        )

        npt.assert_allclose(hfd_dist_3d.coeff_mat, hfd_func_3d.coeff_mat, atol=1e-7)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "pytorch",
        reason="Not supported on this backend",
    )
    def test_multiply_2d_without_truncation(self):
        tvm1 = ToroidalWrappedNormalDistribution(
            array([1.0, 3.0]), diag(array([0.5, 0.5]))
        )
        tvm2 = ToroidalWrappedNormalDistribution(
            array([1.0, 4.0]), diag(array([0.2, 0.2]))
        )

        hfd1_id = HypertoroidalFourierDistribution.from_distribution(
            tvm1, (17, 15), "identity"
        )
        hfd2_id = HypertoroidalFourierDistribution.from_distribution(
            tvm2, (15, 17), "identity"
        )

        hfd1_sqrt = HypertoroidalFourierDistribution.from_distribution(
            tvm1, (17, 15), "sqrt"
        )
        hfd2_sqrt = HypertoroidalFourierDistribution.from_distribution(
            tvm2, (15, 17), "sqrt"
        )

        # Assuming multiply() handles expansion.
        hfd_mult_id = hfd1_id.multiply(hfd2_id)
        hfd_mult_sqrt = hfd1_sqrt.multiply(hfd2_sqrt)

        # Verify integration to 1 (normalization)
        self.assertAlmostEqual(integrate2d(hfd_mult_id), 1.0, places=5)
        self.assertAlmostEqual(integrate2d(hfd_mult_sqrt), 1.0, places=5)

        # Verify values roughly against numerical product
        grid_pts = random.uniform(size=(100, 2)) * 2 * pi
        val_true_unnorm = tvm1.pdf(grid_pts) * tvm2.pdf(grid_pts)
        val_true_norm = val_true_unnorm / sum(val_true_unnorm)  # Rough check
        val_est = hfd_mult_id.pdf(grid_pts)
        val_est_norm = val_est / sum(val_est)

        # Correlation check or relative error on normalized values
        npt.assert_allclose(val_est_norm, val_true_norm, atol=1e-2)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_convolve_2d(self):
        tvm1 = ToroidalWrappedNormalDistribution(
            array([1.0, 2.0]), diag(array([0.5, 0.5]))
        )
        tvm2 = ToroidalWrappedNormalDistribution(
            array([1.0, 4.0]), diag(array([0.2, 0.2]))
        )

        hfd1_id = HypertoroidalFourierDistribution.from_distribution(
            tvm1, (17, 15), "identity"
        )
        hfd2_id = HypertoroidalFourierDistribution.from_distribution(
            tvm2, (15, 17), "identity"
        )

        hfd1_sqrt = HypertoroidalFourierDistribution.from_distribution(
            tvm1, (17, 15), "sqrt"
        )
        hfd2_sqrt = HypertoroidalFourierDistribution.from_distribution(
            tvm2, (15, 17), "sqrt"
        )

        hfd_conv_id = hfd1_id.convolve(hfd2_id)
        hfd_conv_sqrt = hfd1_sqrt.convolve(hfd2_sqrt)

        self.assertAlmostEqual(integrate2d(hfd_conv_id), 1.0, places=5)
        self.assertAlmostEqual(integrate2d(hfd_conv_sqrt), 1.0, places=5)

    # pylint: disable=too-many-locals
    def test_shift(self):
        C_list = [
            2 * array([[1.0, 0.5], [0.5, 1.0]]),
            array([[0.7, 0.4, 0.2], [0.4, 0.6, 0.1], [0.2, 0.1, 1.0]]),
        ]

        for dim_idx, dim in enumerate([2, 3]):
            C = C_list[dim_idx]
            offsets = random.uniform(size=(dim,)) * 4 * pi - pi
            coeffs = (13,) * dim

            # Create centered WN
            hwn_centered = HypertoroidalWrappedNormalDistribution(zeros(dim), C)
            # Create shifted WN
            hwn_shifted = HypertoroidalWrappedNormalDistribution(offsets, C)

            # Convert centered to Fourier
            hfd_id = HypertoroidalFourierDistribution.from_distribution(
                hwn_centered, coeffs, "identity"
            )
            hfd_sqrt = HypertoroidalFourierDistribution.from_distribution(
                hwn_centered, coeffs, "sqrt"
            )

            # Apply shift method
            hfd_id_shifted_fd = hfd_id.shift(offsets)
            hfd_sqrt_shifted_fd = hfd_sqrt.shift(offsets)

            # Create Fourier directly from shifted WN
            hfd_id_shifted_wn = HypertoroidalFourierDistribution.from_distribution(
                hwn_shifted, coeffs, "identity"
            )
            hfd_sqrt_shifted_wn = HypertoroidalFourierDistribution.from_distribution(
                hwn_shifted, coeffs, "sqrt"
            )

            # Compare coefficients
            scale = abs(hfd_id_shifted_wn.coeff_mat).max()
            npt.assert_allclose(
                hfd_id_shifted_fd.coeff_mat,
                hfd_id_shifted_wn.coeff_mat,
                atol=scale / 1000,
            )

            scale_sqrt = abs(hfd_sqrt_shifted_wn.coeff_mat).max()
            npt.assert_allclose(
                hfd_sqrt_shifted_fd.coeff_mat,
                hfd_sqrt_shifted_wn.coeff_mat,
                atol=scale_sqrt / 1000,
            )


if __name__ == "__main__":
    unittest.main()
