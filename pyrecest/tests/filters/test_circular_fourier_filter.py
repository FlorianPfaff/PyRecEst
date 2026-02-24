import unittest
from math import pi
from pyrecest.filters.circular_fourier_filter import CircularFourierFilter
from pyrecest.distributions import VonMisesDistribution, CircularFourierDistribution
from pyrecest.backend import linspace, sin, sign, abs, np
import scipy.integrate
import numpy.testing as npt


class CircularFourierFilterTest(unittest.TestCase):

    def testPredictNonlinear1D(self):
        densityInit = VonMisesDistribution(3, 5)
        fNoiseDist = VonMisesDistribution(0.5, 10)
        noCoeffs = 31
        
        def aGen(a):
            return lambda te: pi * (sin(sign(te-pi)/2.*abs(te-pi)**a/pi**(a - 1)) + 1)
        
        def f_trans(xkk, xk):
            return fNoiseDist.pdf(xkk - a(xk))
        
        def fPredUsingInt(xkk):
            res = []
            for xkkCurr in xkk:
                res.append(scipy.integrate.quad(lambda xkCurr: f_trans(xkkCurr, xkCurr) * densityInit.pdf(xkCurr), 0, 2*pi)[0])
            return res
        
        a = aGen(4)
        xvals = linspace(0, 2*pi, 100)
        for transformation in ['identity', 'sqrt']:
            fourierFilterNl = CircularFourierFilter(noCoeffs, transformation)
            fourierFilterNl.filter_state = CircularFourierDistribution.from_distribution(densityInit, noCoeffs, transformation)
            fourierFilterNl.predict_nonlinear(aGen(4), fNoiseDist)
            npt.assert_allclose(fourierFilterNl.filter_state.pdf(xvals), fPredUsingInt(xvals), atol=2E-5)

    def testPredictNonlinearForLinear1D(self):
        densityInit = VonMisesDistribution(3, 5)
        fNoiseDist = VonMisesDistribution(0.5, 10)
        noCoeffs = 31
        for transformation in ['identity', 'sqrt']:
            fourierFilterLin = CircularFourierFilter(noCoeffs, transformation)
            fourierFilterLin.filter_state = CircularFourierDistribution.from_distribution(densityInit, noCoeffs, transformation)
            fourierFilterNl = CircularFourierFilter(noCoeffs, transformation)
            fourierFilterNl.filter_state = CircularFourierDistribution.from_distribution(densityInit, noCoeffs, transformation)

            # Suppress warnings
            with np.errstate(all='ignore'):
                fourierFilterLin.predict_identity(fNoiseDist)
                fourierFilterNl.predict_nonlinear(lambda x: x, fNoiseDist, True)
                npt.assert_allclose(fourierFilterLin.get_estimate().kldNumerical(fourierFilterNl.get_estimate), 0, atol=1E-8)

                fNoiseDistShifted = fNoiseDist.shift(1)
                fourierFilterLin.predict_identity(fNoiseDistShifted)
                fourierFilterNl.predict_nonlinear(lambda x: x+1, fNoiseDist, False)
                npt.assert_allclose(fourierFilterLin.get_estimate().kldNumerical(fourierFilterNl.get_estimate), 0, atol=1E-8)


if __name__ == '__main__':
    unittest.main()
