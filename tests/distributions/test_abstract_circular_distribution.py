import unittest

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import allclose, arange, array, log, mod, pi
from pyrecest.distributions import (
    CircularUniformDistribution,
    VonMisesDistribution,
    WrappedNormalDistribution,
)


class AbstractCircularDistributionTest(unittest.TestCase):
    def setUp(self):
        self.distributions = [
            WrappedNormalDistribution(array(2.0), array(0.7)),
            VonMisesDistribution(array(6.0), array(1.2)),
        ]

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_cdf_numerical(self):
        """Tests if the numerical computation of cdf matches the actual cdf."""
        x = arange(0, 7)
        starting_point = 2.1

        for dist in self.distributions:
            with self.subTest(distribution=dist):
                self.assertTrue(
                    allclose(dist.cdf_numerical(x), dist.cdf(x), rtol=1e-10)
                )
                self.assertTrue(
                    allclose(
                        dist.cdf_numerical(x, starting_point),
                        dist.cdf(x, starting_point),
                        rtol=1e-10,
                    )
                )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on jax backend",
    )
    def test_trigonometric_moment_numerical(self):
        """Tests if the numerical computation of angular moment matches the actual moment."""
        moments = arange(2)

        for dist in self.distributions:
            for moment in moments:
                with self.subTest(distribution=dist, moment=moment):
                    self.assertTrue(
                        allclose(
                            dist.trigonometric_moment(moment),
                            dist.trigonometric_moment_numerical(moment),
                            rtol=1e-10,
                        )
                    )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_integrate_numerically(self):
        """Tests if the numerical computation of integral matches the actual integral."""
        intervals = [
            (2.0, 2.0),
            (2.0, 3.0),
            (5.0, 4.0),
            (0.0, 4.0 * pi),
            (-pi, pi),
            (0.0, 4.0 * pi),
            (-3.0 * pi, 3.0 * pi),
            (-1.0, 20.0),
            (12.0, -3.0),
        ]

        for dist in self.distributions:
            for interval in intervals:
                with self.subTest(distribution=dist, interval=interval):
                    self.assertTrue(
                        allclose(
                            dist.integrate_numerically(array(interval)),
                            dist.integrate(array(interval)),
                            rtol=1e-10,
                        )
                    )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_kld_numerical(self):
        """Tests numerical computation of the Kullback-Leibler divergence."""
        uniform = CircularUniformDistribution()
        vm = VonMisesDistribution(array(0.3), array(1.8))

        for dist in self.distributions:
            with self.subTest(distribution=dist):
                self.assertTrue(allclose(dist.kld_numerical(dist), 0.0, atol=1e-10))

        self.assertTrue(
            allclose(
                vm.kld_numerical(uniform),
                log(2.0 * pi) - vm.entropy(),
                rtol=1e-8,
            )
        )

    def test_shift_moves_density_forward(self):
        """A positive shift should move mass from x to x + shift_by."""
        dist = VonMisesDistribution(array(1.1), array(4.0))
        shift_by = array([0.7])
        shifted_dist = dist.shift(shift_by)

        xs = array([0.2, 1.1, 1.8, 5.9])
        self.assertTrue(
            allclose(
                shifted_dist.pdf(xs),
                dist.pdf(mod(xs - shift_by[0], 2.0 * pi)),
                rtol=1e-12,
            )
        )

        original_mode = array([1.1])
        shifted_mode = mod(original_mode + shift_by, 2.0 * pi)
        self.assertTrue(
            allclose(
                shifted_dist.pdf(shifted_mode),
                dist.pdf(original_mode),
                rtol=1e-12,
            )
        )


if __name__ == "__main__":
    unittest.main()
