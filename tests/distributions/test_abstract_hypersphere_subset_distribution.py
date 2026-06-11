import unittest
from math import gamma, pi

import numpy.testing as npt
import pyrecest.backend
from parameterized import parameterized

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    array,
    column_stack,
    cos,
    isclose,
    linalg,
    random,
    sin,
    sqrt,
    squeeze,
)
from pyrecest.distributions import (
    AbstractHypersphereSubsetDistribution,
    AbstractSphereSubsetDistribution,
    VonMisesFisherDistribution,
)
from scipy.integrate import nquad


class TestAbstractHypersphereSubsetDistribution(unittest.TestCase):
    @staticmethod
    def _hyperspherical_to_cartesian_2_sphere(angles):
        """
        Converts PyRecEst hyperspherical angles to Cartesian coordinates for S2.
        """
        azimuth, colatitude = angles[:, 0], angles[:, 1]

        x = sin(azimuth) * sin(colatitude)
        y = cos(azimuth) * sin(colatitude)
        z = cos(colatitude)
        return column_stack([x, y, z])

    @staticmethod
    def _hyperspherical_to_cartesian_3_sphere(angles):
        """
        Converts PyRecEst hyperspherical angles to Cartesian coordinates for S3.
        """
        azimuth, theta, chi = angles[:, 0], angles[:, 1], angles[:, 2]

        w = sin(azimuth) * sin(theta) * sin(chi)
        x = cos(azimuth) * sin(theta) * sin(chi)
        y = cos(theta) * sin(chi)
        z = cos(chi)
        return column_stack([w, x, y, z])

    @staticmethod
    def _hyperspherical_to_cartesian_4_sphere(angles):
        """
        Converts PyRecEst hyperspherical angles to Cartesian coordinates for S4.
        """
        azimuth, theta1, theta2, theta3 = (
            angles[:, 0],
            angles[:, 1],
            angles[:, 2],
            angles[:, 3],
        )

        v = sin(azimuth) * sin(theta1) * sin(theta2) * sin(theta3)
        w = cos(azimuth) * sin(theta1) * sin(theta2) * sin(theta3)
        x = cos(theta1) * sin(theta2) * sin(theta3)
        y = cos(theta2) * sin(theta3)
        z = cos(theta3)
        return column_stack([v, w, x, y, z])

    @staticmethod
    def hyperspherical_to_cartesian(r, *angles):
        """
        Convert PyRecEst hyperspherical coordinates to Cartesian coordinates.
        """
        cart_coords = [None] * (len(angles) + 1)
        cart_coords[-1] = r * cos(angles[-1])
        sin_product = r * sin(angles[-1])
        for i in range(2, len(angles) + 1):
            cart_coords[-i] = sin_product * cos(angles[-i])
            sin_product *= sin(angles[-i])
        cart_coords[0] = sin_product
        return tuple(cart_coords)

    @staticmethod
    def _hyperspherical_to_cartesian_s2(r, azimuth, colatitude):
        x = r * sin(azimuth) * sin(colatitude)
        y = r * cos(azimuth) * sin(colatitude)
        z = r * cos(colatitude)
        return x, y, z

    @staticmethod
    def _hyperspherical_to_cartesian_s3(r, azimuth, theta, chi):
        x0 = r * sin(azimuth) * sin(theta) * sin(chi)
        x1 = r * cos(azimuth) * sin(theta) * sin(chi)
        x2 = r * cos(theta) * sin(chi)
        x3 = r * cos(chi)
        return x0, x1, x2, x3

    @staticmethod
    def _uniform_on_sn(*coordinates):
        """
        Calculate the probability density for a point to be on the surface of an n-dimensional sphere (S^(n-1)).

        Parameters:
        - coordinates (float): Cartesian coordinates of the point (x1, x2, ..., xn).

        Returns:
        - float: Probability density if the point is on the surface of the unit n-sphere, otherwise 0.
        """
        n = len(coordinates)  # Dimension of the sphere
        norm = sqrt(sum(coord**2 for coord in coordinates))

        # Check if the point is on the surface of the unit n-sphere
        if not isclose(norm, 1.0):
            return 0

        # Calculate the surface area of the unit n-sphere
        surface_area = 2 * pi ** (n / 2) / gamma(n / 2)
        # Return the reciprocal of the surface area
        return 1 / surface_area

    # Adjusted integrand function for spherical coordinates
    @staticmethod
    def _integrand_s2(azimuth, colatitude):
        (
            x,
            y,
            z,
        ) = TestAbstractHypersphereSubsetDistribution._hyperspherical_to_cartesian_s2(
            1, azimuth, colatitude
        )  # radius = 1 for the unit sphere
        return TestAbstractHypersphereSubsetDistribution._uniform_on_sn(x, y, z) * sin(
            colatitude
        )  # Jacobian factor for spherical coordinates

    # Adjusted integrand function for hyperspherical coordinates
    @staticmethod
    def _integrand_s3(azimuth, theta, chi):
        (
            x,
            y,
            z,
            w,
        ) = TestAbstractHypersphereSubsetDistribution._hyperspherical_to_cartesian_s3(
            1, azimuth, theta, chi
        )  # radius = 1 for the unit 3-sphere
        return (
            TestAbstractHypersphereSubsetDistribution._uniform_on_sn(x, y, z, w)
            * sin(chi) ** 2
            * sin(theta)
        )  # Jacobian factor for hyperspherical coordinates

    @parameterized.expand(
        [
            # 2D case
            (_integrand_s2, [[0, 2 * pi], [0, pi]]),
            # 3D case
            (_integrand_s3, [[0, 2 * pi], [0, pi], [0, pi]]),
        ]
    )
    def test_integrate(self, integrand_method, bounds):
        """Test the locally defined integrands (testing the test class, not the actual distribution)"""
        # Perform the integration
        integral_result, _ = nquad(integrand_method, bounds)
        npt.assert_allclose(integral_result, 1, atol=0.001)

    # Now we test these explicit implementations against the provided generic function
    @parameterized.expand(
        [
            (2, _hyperspherical_to_cartesian_s2, 1),
            (3, _hyperspherical_to_cartesian_s3, 1),
            (2, _hyperspherical_to_cartesian_s2, 10),
            (3, _hyperspherical_to_cartesian_s3, 10),
        ]
    )
    def test_hyperspherical_to_cartesian_specific(
        self, dimensions, specific_function, n_samples: int
    ):
        # Test for single elements
        for _ in range(10):
            size_samples = (n_samples, dimensions) if n_samples > 1 else dimensions
            upper_bounds = array([2.0 * pi] + [pi] * (dimensions - 1))
            angles = upper_bounds * random.uniform(size=size_samples)
            cartesian_specific = squeeze(column_stack(specific_function(1, *angles.T)))
            cartesian_given = AbstractHypersphereSubsetDistribution.hypersph_to_cart(
                angles, mode="colatitude"
            )
            npt.assert_allclose(cartesian_specific, cartesian_given, rtol=2e-7)

    def test_hyperspherical_colatitude_known_points_s2(self):
        """Document the PyRecEst S2 colatitude convention."""
        angles = array(
            [
                [0.0, 0.0],
                [0.0, pi / 2.0],
                [pi / 2.0, pi / 2.0],
                [pi, pi / 2.0],
                [3.0 * pi / 2.0, pi / 2.0],
            ]
        )
        expected_cartesian = array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ]
        )

        cartesian_given = AbstractHypersphereSubsetDistribution.hypersph_to_cart(
            angles, mode="colatitude"
        )

        npt.assert_allclose(cartesian_given, expected_cartesian, atol=1e-7)

    @parameterized.expand([(1,), (2,), (3,), (4,)])
    def test_hyperspherical_colatitude_forward_backward(self, dimensions):
        """Check angles -> Cartesian -> angles -> Cartesian consistency."""
        n_samples = 20
        if dimensions == 1:
            angles = 2.0 * pi * random.uniform(size=(n_samples, 1))
        else:
            azimuth = 2.0 * pi * random.uniform(size=(n_samples, 1))
            colatitudes = 0.1 + (pi - 0.2) * random.uniform(
                size=(n_samples, dimensions - 1)
            )
            angles = column_stack((azimuth, colatitudes))

        cartesian = AbstractHypersphereSubsetDistribution.hypersph_to_cart(
            angles, mode="colatitude"
        )
        angles_round_trip = AbstractHypersphereSubsetDistribution.cart_to_hypersph(
            cartesian, mode="colatitude"
        )
        cartesian_round_trip = AbstractHypersphereSubsetDistribution.hypersph_to_cart(
            angles_round_trip, mode="colatitude"
        )

        npt.assert_allclose(cartesian_round_trip, cartesian, atol=1e-7)

    def test_single_list_inputs_keep_single_output_shape(self):
        angles = [pi / 2.0, pi / 2.0]

        cartesian_from_list = AbstractHypersphereSubsetDistribution.hypersph_to_cart(
            angles, mode="colatitude"
        )
        cartesian_from_array = AbstractHypersphereSubsetDistribution.hypersph_to_cart(
            array(angles), mode="colatitude"
        )

        self.assertEqual(cartesian_from_array.shape, cartesian_from_list.shape)
        npt.assert_allclose(cartesian_from_list, cartesian_from_array, atol=1e-7)

        coords = [0.0, 1.0, 0.0]
        angles_from_list = AbstractHypersphereSubsetDistribution.cart_to_hypersph(
            coords, mode="colatitude"
        )
        angles_from_array = AbstractHypersphereSubsetDistribution.cart_to_hypersph(
            array(coords), mode="colatitude"
        )

        self.assertEqual(angles_from_array.shape, angles_from_list.shape)
        npt.assert_allclose(angles_from_list, angles_from_array, atol=1e-7)

    def test_s2_only_modes_reject_wrong_input_dimension(self):
        hyperspherical_s3 = array([[0.0, pi / 2.0, pi / 4.0]])
        cartesian_s3 = array([[0.0, 1.0, 0.0, 0.0]])

        for mode in ("elevation", "inclination"):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, "S2"):
                    AbstractHypersphereSubsetDistribution.hypersph_to_cart(
                        hyperspherical_s3, mode=mode
                    )
                with self.assertRaisesRegex(ValueError, "S2"):
                    AbstractHypersphereSubsetDistribution.cart_to_hypersph(
                        cartesian_s3, mode=mode
                    )

    def test_sphere_colatitude_matches_hyperspherical_convention(self):
        azimuth = array([0.0, pi / 2.0])
        colatitude = array([pi / 2.0, pi / 2.0])

        x, y, z = AbstractSphereSubsetDistribution.sph_to_cart(
            azimuth, colatitude, mode="colatitude"
        )
        cartesian = column_stack((x, y, z))

        npt.assert_allclose(
            cartesian,
            array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
            atol=1e-7,
        )

        azimuth_back, colatitude_back = AbstractSphereSubsetDistribution.cart_to_sph(
            x, y, z, mode="colatitude"
        )
        x_back, y_back, z_back = AbstractSphereSubsetDistribution.sph_to_cart(
            azimuth_back, colatitude_back, mode="colatitude"
        )

        npt.assert_allclose(
            column_stack((x_back, y_back, z_back)), cartesian, atol=1e-7
        )

    @parameterized.expand(
        [
            ("colatitude",),
            ("elevation",),
            ("inclination",),
        ]
    )
    def test_cart_to_sph_to_cart(self, mode):
        # jscpd:ignore-start
        # Create some Cartesian coordinates
        x = array([1.0, 0.0, 0.0])
        y = array([0.0, 1.0, 0.0])
        z = array([0.0, 0.0, 1.0])
        # jscpd:ignore-end

        # Convert to spherical coordinates and back
        angles = AbstractHypersphereSubsetDistribution.cart_to_hypersph(
            column_stack((x, y, z)), mode=mode
        )
        cart_res = AbstractHypersphereSubsetDistribution.hypersph_to_cart(
            angles, mode=mode
        )

        # The new Cartesian coordinates should be close to the original ones
        npt.assert_allclose(cart_res[:, 0], x, atol=1e-7)
        npt.assert_allclose(cart_res[:, 1], y, atol=1e-7)
        npt.assert_allclose(cart_res[:, 2], z, atol=1e-7)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_pdf_hyperspherical_coords_1d(self):
        mu_ = array([0.5, 1.0]) / linalg.norm(array([0.5, 1.0]))
        kappa_ = 2.0
        vmf = VonMisesFisherDistribution(mu_, kappa_)

        pdf_hyperspherical = vmf.gen_pdf_hyperspherical_coords()

        def fangles_1d(phi):
            return vmf.pdf(array([sin(phi), cos(phi)]).T)

        phi_test = array([1.0, 2.0, 0.0, 0.3, 1.1])

        npt.assert_array_almost_equal(
            pdf_hyperspherical(phi_test), fangles_1d(phi_test)
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_pdf_hyperspherical_coords_2d(self):
        mu_ = array([0.5, 1.0, 1.0]) / linalg.norm(array([0.5, 1.0, 1.0]))
        kappa_ = 2.0
        vmf = VonMisesFisherDistribution(mu_, kappa_)

        pdf_hyperspherical = vmf.gen_pdf_hyperspherical_coords()

        def fangles_2d(phi1, phi2):
            r = 1
            return vmf.pdf(
                array(
                    [
                        r * sin(phi1) * sin(phi2),
                        r * cos(phi1) * sin(phi2),
                        r * cos(phi2),
                    ]
                ).T
            )

        phi1_test = array([1.0, 2.0, 0.0, 0.3, 1.1])
        phi2_test = array([2.0, 3.0, 0.1, 3.0, 1.1])

        npt.assert_array_almost_equal(
            pdf_hyperspherical(phi1_test, phi2_test), fangles_2d(phi1_test, phi2_test)
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_pdf_hyperspherical_coords_3d(self):
        mu_ = array([0.5, 1.0, 1.0, -0.5]) / linalg.norm(array([0.5, 1.0, 1.0, -0.5]))
        kappa_ = 2.0
        vmf = VonMisesFisherDistribution(mu_, kappa_)

        pdf_hyperspherical = vmf.gen_pdf_hyperspherical_coords()

        def fangles_3d(phi1, phi2, phi3):
            r = 1
            return vmf.pdf(
                array(
                    [
                        r * sin(phi1) * sin(phi2) * sin(phi3),
                        r * cos(phi1) * sin(phi2) * sin(phi3),
                        r * cos(phi2) * sin(phi3),
                        r * cos(phi3),
                    ]
                ).T
            )

        phi1_test = array([1.0, 2.0, 0.0, 0.3, 1.1])
        phi2_test = array([2.0, 3.0, 0.1, 3.0, 1.1])
        phi3_test = phi2_test + 0.2

        npt.assert_array_almost_equal(
            pdf_hyperspherical(phi1_test, phi2_test, phi3_test),
            fangles_3d(phi1_test, phi2_test, phi3_test),
        )


if __name__ == "__main__":
    unittest.main()
