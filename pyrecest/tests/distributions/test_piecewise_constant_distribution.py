import unittest
from pyrecest.backend import pi, mean, arange, concatenate, abs

from pyrecest.distributions.circle.piecewise_constant_distribution import PieceWiseConstantDistribution
from pyrecest.distributions import WrappedNormalDistribution
import numpy.testing as npt

class test_PWC_distribution(unittest.TestCase):
    w = concatenate((arange(1, 6), arange(10, 0, -1)))
    normal = 1 / (2 * pi * mean(w))
    p = PieceWiseConstantDistribution(w)

    # test pdf
    npt.assert_allclose(p.pdf(0), 1 * normal, rtol=1e-10)
    npt.assert_allclose(p.pdf(4.2), 5 * normal, rtol=1e-10)
    npt.assert_allclose(p.pdf(10.9), 4 * normal, rtol=1e-10)

    # test integral
    npt.assert_allclose(p.integrate(), 1, rtol=1e-6)
    npt.assert_allclose(p.integrate([0, pi]) + p.integrate([pi, 2 * pi]), 1, rtol=1e-6)

    # test trigonometric moments
    npt.assert_allclose(p.trigonometric_moment(1), p.trigonometric_moment_numerical(1), rtol=1e-5)
    npt.assert_allclose(p.trigonometric_moment(2), p.trigonometric_moment_numerical(2), rtol=1e-5)
    npt.assert_allclose(p.trigonometric_moment(3), p.trigonometric_moment_numerical(3), rtol=1e-5)

    # test interval borders
    assert np.isclose(PieceWiseConstantDistribution.left_border(1, 2), 0 * 2 * pi)
    assert np.isclose(PieceWiseConstantDistribution.interval_center(1, 2), 1 / 4 * 2 * pi)
    assert np.isclose(PieceWiseConstantDistribution.right_border(1, 2), 1 / 2 * 2 * pi)
    assert np.isclose(PieceWiseConstantDistribution.left_border(2, 2), 1 / 2 * 2 * pi)
    assert np.isclose(PieceWiseConstantDistribution.interval_center(2, 2), 3 / 4 * 2 * pi)
    assert np.isclose(PieceWiseConstantDistribution.right_border(2, 2), 1 * 2 * pi)
    # more samples should lead to better approximation
    wn = WrappedNormalDistribution(2, 1.3)
    w1 = PieceWiseConstantDistribution.calculate_parameters_numerically(wn.pdf, 40)
    w2 = PieceWiseConstantDistribution.calculate_parameters_numerically(wn.pdf, 45)
    w3 = PieceWiseConstantDistribution.calculate_parameters_numerically(wn.pdf, 50)
    p1 = PieceWiseConstantDistribution(w1)
    p2 = PieceWiseConstantDistribution(w2)
    p3 = PieceWiseConstantDistribution(w3)
    delta1 = abs(wn.trigonometric_moment(1) - p1.trigonometric_moment(1))
    delta2 = abs(wn.trigonometric_moment(1) - p2.trigonometric_moment(1))
    delta3 = abs(wn.trigonometric_moment(1) - p3.trigonometric_moment(1))
    assert delta2 <= delta1
    assert delta3 <= delta2

if __name__ == "__main__":
    unittest.main()
