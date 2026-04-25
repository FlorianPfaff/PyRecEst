import unittest
import warnings
from math import pi

import numpy.testing as npt
import scipy.integrate

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import abs, linspace, sign, sin
from pyrecest.distributions import CircularFourierDistribution, VonMisesDistribution
from pyrecest.filters.circular_fourier_filter import CircularFourierFilter


class CircularFourierFilterTest(unittest.TestCase):

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("jax", "pytorch"),
        reason="HypertoroidalFourierFilter is not supported on this backend",
    )
    def testPredictNonlinear1D(self):
        densityInit = VonMisesDistribution(3, 5)
        fNoiseDist = VonMisesDistribution(0.5, 10)
        noCoeffs = 31

        def aGen(a):
            return lambda te: pi * (sin(sign(te - pi) / 2.0 * abs(te - pi) ** a / pi ** (a - 1)) + 1)

        def f_trans(xkk, xk):
            return fNoiseDist.pdf(xkk - a(xk))

        def fPredUsingInt(xkk):
            res = []
            for xkkCurr in xkk:
                def integrand(xkCurr, xkkValue=xkkCurr):
                    return f_trans(xkkValue, xkCurr) * densityInit.pdf(xkCurr)

                res.append(scipy.integrate.quad(integrand, 0, 2 * pi)[0])
            return res

        a = aGen(4)
        xvals = linspace(0, 2 * pi, 100)
        for transformation in ["identity", "sqrt"]:
            fourierFilterNl = CircularFourierFilter(noCoeffs, transformation)
            fourierFilterNl.filter_state = CircularFourierDistribution.from_distribution(
                densityInit, noCoeffs, transformation
            )
            fourierFilterNl.predict_nonlinear(aGen(4), fNoiseDist)
            npt.assert_allclose(
                fourierFilterNl.filter_state.pdf(xvals),
                fPredUsingInt(xvals),
                atol=2e-5,
            )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("jax", "pytorch"),
        reason="HypertoroidalFourierFilter is not supported on this backend",
    )
    def testPredictNonlinearForLinear1D(self):
        densityInit = VonMisesDistribution(3, 5)
        fNoiseDist = VonMisesDistribution(0.5, 10)
        noCoeffs = 31
        for transformation in ["identity", "sqrt"]:
            fourierFilterLin = CircularFourierFilter(noCoeffs, transformation)
            fourierFilterLin.filter_state = CircularFourierDistribution.from_distribution(
                densityInit, noCoeffs, transformation
            )
            fourierFilterNl = CircularFourierFilter(noCoeffs, transformation)
            fourierFilterNl.filter_state = CircularFourierDistribution.from_distribution(
                densityInit, noCoeffs, transformation
            )

            # Suppress warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                fourierFilterLin.predict_identity(fNoiseDist)
                fourierFilterNl.predict_nonlinear(lambda x: x, fNoiseDist, True)
                npt.assert_allclose(
                    fourierFilterLin.get_estimate().kld_numerical(
                        fourierFilterNl.get_estimate()
                    ),
                    0,
                    atol=1e-8,
                )

                fNoiseDistShifted = VonMisesDistribution(
                    fNoiseDist.mu + 1.0, fNoiseDist.kappa
                )
                fourierFilterLin.predict_identity(fNoiseDistShifted)
                fourierFilterNl.predict_nonlinear(lambda x: x + 1, fNoiseDist, False)
                npt.assert_allclose(
                    fourierFilterLin.get_estimate().kld_numerical(
                        fourierFilterNl.get_estimate()
                    ),
                    0,
                    atol=1e-8,
                )


if __name__ == "__main__":
    unittest.main()
