import unittest

import matplotlib
import matplotlib.pyplot as plt
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, linspace
from pyrecest.distributions import VonMisesDistribution


class TestVonMisesDistribution(unittest.TestCase):
    def test_vm_init(self):
        dist1 = VonMisesDistribution(0, 1)
        dist2 = VonMisesDistribution(2, 1)
        self.assertEqual(dist1.kappa, dist2.kappa)
        self.assertNotEqual(dist1.mu, dist2.mu)

    def test_pdf(self):
        dist = VonMisesDistribution(2, 1)
        xs = linspace(1, 7, 7)
        npt.assert_array_almost_equal(
            dist.pdf(xs),
            array(
                [
                    0.215781465110296,
                    0.341710488623463,
                    0.215781465110296,
                    0.0829150854731715,
                    0.0467106111086458,
                    0.0653867888824553,
                    0.166938593220285,
                ],
            ),
        )

    def test_uniform_trigonometric_moments(self):
        dist = VonMisesDistribution(2, 0)

        npt.assert_allclose(dist.trigonometric_moment(0), array(1.0 + 0.0j))
        npt.assert_allclose(dist.trigonometric_moment(1), array(0.0 + 0.0j))
        npt.assert_allclose(dist.trigonometric_moment(2), array(0.0 + 0.0j))

    def test_plot(self):
        matplotlib.pyplot.close("all")
        matplotlib.use("Agg")
        vm = VonMisesDistribution(0, 1)
        vm.plot()
        plt.close()


if __name__ == "__main__":
    unittest.main()
