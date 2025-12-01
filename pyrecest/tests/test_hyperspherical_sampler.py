import importlib.util
import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module
import pyrecest.backend
from parameterized import parameterized

# pylint: disable=no-name-in-module,no-member,redefined-builtin
from pyrecest.backend import (
    all,
    any,
    array,
    column_stack,
    isclose,
    linalg,
    linspace,
    pi,
    random,
    zeros,
)
from pyrecest.sampling.hyperspherical_sampler import (
    get_grid_hyperhemisphere,
    get_grid_hypersphere,
)

from ..sampling.hyperspherical_sampler import (
    AbstractHopfBasedS3Sampler,
    DriscollHealySampler,
    FibonacciHopfSampler,
    HealpixHopfSampler,
    HealpixSampler,
    LeopardiSampler,
    SphericalCoordinatesBasedFixedResolutionSampler,
    SphericalFibonacciSampler,
    SymmetricLeopardiSampler,
)

healpy_installed = importlib.util.find_spec("healpy") is not None


class TestHypersphericalGridGenerationFunction(unittest.TestCase):
    @parameterized.expand(
        [
            ("healpix", 2, 48, "n_side"),
            ("driscoll_healy", 2, 91, "l_max"),
            ("fibonacci", 12, 12, "n_samples"),
            ("spherical_fibonacci", 12, 12, "n_samples"),
        ]
    )
    @unittest.skipIf(not healpy_installed, "healpy is not installed")
    def test_get_grid_sphere(
        self, method, grid_density_parameter, grid_points_expected, desc_key
    ):
        samples, grid_specific_description = get_grid_hypersphere(
            method, grid_density_parameter, dim=2
        )

        self.assertEqual(
            samples.shape[0],
            grid_points_expected,
            f"Expected {grid_points_expected} points but got {samples.shape[0]}",
        )
        self.assertEqual(
            samples.shape[1],
            3,
            f"Expected 3-dimensional-output but got {samples.shape[1]}-dimensional output",
        )
        self.assertEqual(grid_specific_description[desc_key], grid_density_parameter)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    @unittest.skipIf(not healpy_installed, "healpy is not installed")
    def test_get_grid_hypersphere(self):
        samples, _ = get_grid_hypersphere("healpix_hopf", 0, dim=3)

        self.assertEqual(
            samples.shape[0], 72, f"Expected {72} points but got {samples.shape[0]}"
        )
        self.assertEqual(
            samples.shape[1],
            4,
            f"Expected 4-dimensional-output but got {samples.shape[1]}-dimensional output",
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_get_grid_hyperhemisphere_leopardi(self):
        grid_density_parameter = 12
        samples, grid_specific_description = get_grid_hyperhemisphere(
            "leopardi", grid_density_parameter, dim=2
        )

        # Shape: N points on S^2 → N x 3
        self.assertEqual(
            samples.shape,
            (grid_density_parameter, 3),
            f"Expected shape {(grid_density_parameter, 3)} but got {samples.shape}",
        )

        # Description fields
        self.assertEqual(grid_specific_description["scheme"], "leopardi_hemisphere")
        self.assertEqual(grid_specific_description["n_side"], grid_density_parameter)

        # Points should lie on the unit sphere
        norms = linalg.norm(samples, axis=1)
        npt.assert_allclose(norms, 1.0, atol=1e-10)

        # Should be a hemisphere: last coordinate non‑negative (allow tiny numerical noise)
        npt.assert_array_less(-1e-12, samples[:, -1])
        self.assertTrue(
            (samples[:, -1] >= -1e-12).all(),
            "Expected all points to lie in the upper hyperhemisphere (last coord >= 0)",
        )

    def test_get_grid_hyperhemisphere_invalid_method(self):
        with self.assertRaises(ValueError):
            get_grid_hyperhemisphere("unknown_method", 10, dim=10)


class TestHypersphericalSampler(unittest.TestCase):
    @parameterized.expand(
        [
            (HealpixSampler(), 2, 48, "n_side"),
            (DriscollHealySampler(), 2, 91, "l_max"),
            (SphericalFibonacciSampler(), 12, 12, "n_samples"),
        ]
    )
    @unittest.skipIf(not healpy_installed, "healpy is not installed")
    def test_samplers(
        self, sampler, grid_density_parameter, grid_points_expected, desc_key
    ):
        grid, grid_description = sampler.get_grid(grid_density_parameter)

        self.assertEqual(
            grid.shape[0],
            grid_points_expected,
            f"Expected {grid_points_expected} points but got {grid.shape[0]}",
        )
        self.assertEqual(
            grid.shape[1],
            3,
            f"Expected 3-dimensional-output but got {grid.shape[1]}-dimensional output",
        )
        self.assertEqual(grid_description[desc_key], grid_density_parameter)

    @parameterized.expand([(0, 72), (1, 648)])
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    @unittest.skipIf(not healpy_installed, "healpy is not installed")
    def test_healpix_hopf_sampler(self, input_value, expected_grid_points):
        sampler = HealpixHopfSampler()
        dim = 3
        grid, _ = sampler.get_grid(input_value)

        self.assertEqual(
            grid.shape[0],
            expected_grid_points,
            f"Expected {expected_grid_points} points but got {grid.shape[1]}",
        )
        self.assertEqual(
            grid.shape[1],
            dim + 1,
            f"Expected {dim+1}-dimensional-output but got {grid.shape[1]}-dimensional output",
        )

    def test_fibonacci_hopf_sampler(self):
        sampler = FibonacciHopfSampler()
        grid_density_parameter = [12, 4]
        grid, _ = sampler.get_grid(grid_density_parameter)

        expected_points = grid_density_parameter[0] * grid_density_parameter[1]
        self.assertEqual(
            grid.shape[0],
            expected_points,
            f"Expected {expected_points} points but got {grid.shape[0]}",
        )
        self.assertEqual(
            grid.shape[1],
            4,
            f"Expected 4-dimensional-output but got {grid.shape[1]}-dimensional output",
        )

    @parameterized.expand(
        [
            ("test_2d_12_n_only", 12, 2, (12, 3)),
            ("test_3d_15_n_only", 15, 3, (15, 4)),
        ]
    )
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_leopardi_sampler(self, _, points, dim, expected_shape):
        sampler = LeopardiSampler(original_code_column_order=False)
        grid, _ = sampler.get_grid(points, dim=dim)
        npt.assert_equal(grid.shape, expected_shape)

    # jscpd:ignore-start
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_leopardi_sampler_s2_12(self):
        sampler = LeopardiSampler(original_code_column_order=True)
        grid, _ = sampler.get_grid(12, dim=2)
        npt.assert_equal(grid.shape, (12, 3))

        # Define the first six point as obtained by the Matlab implementation
        matlab_result_array = array(
            [
                [0.0000, 0.7128, -0.2723, -0.8811, -0.2723, 0.7128],
                [0.0000, 0.5179, 0.8380, 0.0000, -0.8380, -0.5179],
                [1.0000, 0.4729, 0.4729, 0.4729, 0.4729, 0.4729],
            ]
        ).T

        # First six should be approximately equal to the ones by the Matlab implementation by Leopardi
        npt.assert_allclose(grid[:6], matlab_result_array, atol=1e-4)
        # Next 5 are mirrored from above
        npt.assert_allclose(
            grid[6:11],
            column_stack(
                ((matlab_result_array[1:, [0, 1]]), (-matlab_result_array[1:, 2]))
            ),
            atol=1e-4,
        )
        # Last is just [0, 0, -1]
        npt.assert_allclose(grid[-1], array([0.0, 0.0, -1.0]), atol=1e-4)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_leopardi_sampler_s2_20(self):
        sampler = LeopardiSampler(original_code_column_order=True)
        grid, _ = sampler.get_grid(20, dim=2)
        npt.assert_equal(grid.shape, (20, 3))

        # Define the first six point as obtained by the Matlab implementation
        matlab_result_array = array(
            [
                [0, 0.5833, -0.2228, -0.7209, -0.2228, 0.5833],
                [0, 0.4238, 0.6857, 0.0000, -0.6857, -0.4238],
                [1.0000, 0.6930, 0.6930, 0.6930, 0.6930, 0.6930],
            ]
        ).T

        # Check if they are approximately equal
        npt.assert_allclose(grid[:6], matlab_result_array, atol=1e-4)
        # The other ones like in the plane with z = 0
        npt.assert_allclose(grid[7:14, 2], zeros(7), atol=1e-4)
        # Next 5 are mirrored from above
        npt.assert_allclose(
            grid[14:-1],
            column_stack(
                ((matlab_result_array[1:, [0, 1]]), (-matlab_result_array[1:, 2]))
            ),
            atol=1e-4,
        )
        # Last is just [0, 0, -1]
        npt.assert_allclose(grid[-1], array([0.0, 0.0, -1.0]), atol=1e-4)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_leopardi_sampler_s3_10_first5(self):
        sampler = LeopardiSampler(True)
        grid, _ = sampler.get_grid(10, dim=3)
        npt.assert_equal(grid.shape, (10, 4))

        matlab_result_array = array(
            [
                [0, 0, 0.8660, 0.0000, -0.8660],
                [0, 0, 0.5000, 1.0000, 0.5000],
                [0, 1.0000, 0.0000, 0.0000, 0.0000],
                [1.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            ]
        ).T

        # First five should be approximately equal to the ones by the Matlab implementation by Leopardi
        npt.assert_allclose(grid[:5], matlab_result_array, atol=1e-4)

    # jscpd:ignore-end


class TestSphericalCoordinatesBasedFixedResolutionSampler(unittest.TestCase):
    def test_get_grid_spherical_coordinates(self):
        # Create an instance of the sampler
        sampler = SphericalCoordinatesBasedFixedResolutionSampler()

        # Define the resolution parameters for latitude and longitude
        grid_density_parameter = array(
            [10, 20]
        )  # 10 latitude lines, 20 longitude lines

        # Call the method
        phi, theta, _ = sampler.get_grid_spherical_coordinates(grid_density_parameter)

        expected_phi = linspace(
            0.0, 2 * pi, num=grid_density_parameter[0], endpoint=False
        )
        expected_theta = linspace(
            pi / (grid_density_parameter[1] + 1),
            pi,
            num=grid_density_parameter[1],
            endpoint=False,
        )

        # Check if the first and last values of the generated phi and theta are as expected
        # This assumes that you have access to phi and theta which might not be the case here
        npt.assert_allclose(phi, expected_phi)
        npt.assert_allclose(theta, expected_theta)


class TestHopfConversion(unittest.TestCase):
    def test_conversion(self):
        # Generate a sample matrix of size (n, 4) containing unit vectors.
        n = 100  # sample size
        random_vectors = random.normal(size=(n, 4))
        unit_vectors = random_vectors / linalg.norm(random_vectors, axis=1)[:, None]

        # Pass the quaternions through the conversion functions
        θ, ϕ, ψ = AbstractHopfBasedS3Sampler.quaternion_to_hopf_yershova(unit_vectors)
        recovered_quaternions = (
            AbstractHopfBasedS3Sampler.hopf_coordinates_to_quaterion_yershova(θ, ϕ, ψ)
        )

        # Check if the original quaternions are close to the recovered quaternions.
        npt.assert_allclose(unit_vectors, recovered_quaternions, atol=3e-6)


class TestSymmetricLeopardiSampler(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_antipodally_symmetric_sampling(self):
        """
        Check that antipodally-symmetric sampling really produces antipodal pairs
        and that delete_half behaves as expected.
        """
        dim, N = 2, 40  # N must be even
        tol = 1e-10

        # Half set: one representative from each ± pair
        ls_full = SymmetricLeopardiSampler(
            delete_half=True, original_code_column_order=True, symmetry_type="antipodal"
        )
        pts_half, _ = ls_full.get_grid(N, dim=dim)

        # Full set: all ± pairs
        ls_half = SymmetricLeopardiSampler(
            delete_half=False,
            original_code_column_order=True,
            symmetry_type="antipodal",
        )
        pts_full, _ = ls_half.get_grid(N, dim=dim)

        # Shape checks
        self.assertEqual(pts_half.shape, (N // 2, dim + 1))
        self.assertEqual(pts_full.shape, (N, dim + 1))

        # All points lie on the unit sphere
        norms_half = linalg.norm(pts_half, axis=1)
        norms_full = linalg.norm(pts_full, axis=1)
        npt.assert_allclose(norms_half, 1.0, atol=tol)
        npt.assert_allclose(norms_full, 1.0, atol=tol)

        # As sets (up to numerical noise), full set = half ∪ (-half)
        def as_value_tuple(tup, ndigits=4):
            # Convert to tuple of floats with fixed precision
            # because hashing uses id for pytorch.
            # Note: Using builtin round and not the backend's.
            return tuple(round(x.item(), ndigits) for x in tup)

        half_set = {as_value_tuple(row) for row in pts_half}
        neg_half_set = {as_value_tuple(-row) for row in pts_half}
        self.assertEqual(len(half_set), N // 2)
        self.assertEqual(len(neg_half_set), N // 2)

        full_set = {as_value_tuple(row) for row in pts_full}
        self.assertEqual(len(full_set), N)  # no duplicates beyond the ± pairing
        self.assertTrue(half_set.isdisjoint(neg_half_set))
        self.assertEqual(full_set, half_set | neg_half_set)

    @parameterized.expand(
        [
            (2, 40),
            (3, 80),
        ]
    )
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_plane_reflection_sampling(self, dim, N):
        """
        Check that plane-refelection sampling is symmetric w.r.t. the
        equatorial hyperplane (last coordinate flips sign, others stay).
        """
        dim, N = 2, 40  # N must be even
        tol = 1e-10

        ls = SymmetricLeopardiSampler(
            delete_half=False, original_code_column_order=True, symmetry_type="plane"
        )
        pts, _ = ls.get_grid(N, dim=dim)

        self.assertEqual(pts.shape, (N, dim + 1))

        # Still on the unit sphere
        norms = linalg.norm(pts, axis=1)
        npt.assert_allclose(norms, 1.0, atol=tol)

        # There should be a clear north and south pole
        z = pts[:, -1]
        npt.assert_allclose(z.max(), 1.0, atol=tol)
        npt.assert_allclose(z.min(), -1.0, atol=tol)

        # For every non-polar point v, there exists a point w with:
        #   w[:-1] ≈ v[:-1]  and  w[-1] ≈ -v[-1]
        for i in range(N):
            v = pts[i, :]

            # Skip poles (z ≈ ±1)
            if isclose(v[-1], 1.0, atol=tol) or isclose(v[-1], -1.0, atol=tol):
                continue

            # Find candidates whose first dim coordinates match v's (within tol)
            same_xy = all(abs(pts[:, :-1] - v[None, :-1]) < 5 * tol, axis=1)
            candidates = pts[same_xy, :]

            # Among those, at least one must have opposite z
            self.assertTrue(any(isclose(candidates[:, -1], -v[-1], atol=5 * tol)))


if __name__ == "__main__":
    unittest.main()
