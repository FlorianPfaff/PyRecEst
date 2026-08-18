import os
import subprocess
import sys
import unittest
import warnings
from math import pi

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
import scipy.integrate
from pyrecest.backend import abs, array, linspace, sign, sin
from pyrecest.distributions import CircularFourierDistribution, VonMisesDistribution
from pyrecest.filters.circular_fourier_filter import CircularFourierFilter


class CircularFourierFilterTest(unittest.TestCase):

    def test_filter_state_rejects_non_circular_distribution_without_asserts(self):
        fourier_filter = CircularFourierFilter(5)

        with self.assertRaisesRegex(ValueError, "AbstractCircularDistribution"):
            fourier_filter.filter_state = object()

    def test_array_predict_identity_validates_transformation_without_asserts(self):
        fourier_filter = CircularFourierFilter(5, "identity")

        with self.assertRaisesRegex(NotImplementedError, "sqrt"):
            fourier_filter.predict_identity(array([1.0, 1.0, 1.0, 1.0, 1.0]))

    def test_array_predict_identity_validates_grid_size_without_asserts(self):
        fourier_filter = CircularFourierFilter(5, "sqrt")

        with self.assertRaisesRegex(ValueError, "expected 5, got 4"):
            fourier_filter.predict_identity(array([1.0, 1.0, 1.0, 1.0]))

    def test_array_predict_identity_uses_circular_convolution(self):
        no_coefficients = 5
        fourier_filter = CircularFourierFilter(no_coefficients, "sqrt")

        # A normalized, deliberately asymmetric circular noise density.  A
        # uniform prior convolved with any normalized circular density must
        # remain uniform.  Linear mode="same" convolution violates this at the
        # wrap boundary.
        noise_values = array([4.0, 1.0, 0.5, 0.25, 0.125])
        noise_values = noise_values / (5.875 * (2.0 * pi / no_coefficients))

        fourier_filter.predict_identity(noise_values)

        x_vals = linspace(0.0, 2.0 * pi, no_coefficients, endpoint=False)
        expected = [1.0 / (2.0 * pi)] * no_coefficients
        npt.assert_allclose(
            fourier_filter.filter_state.pdf(x_vals),
            expected,
            rtol=1e-12,
            atol=1e-12,
        )
        npt.assert_allclose(fourier_filter.filter_state.integrate(), 1.0, atol=1e-12)

    def test_predict_nonlinear_rejects_noncallable_without_asserts(self):
        fourier_filter = CircularFourierFilter(5)

        with self.assertRaisesRegex(TypeError, "f must be callable"):
            fourier_filter.predict_nonlinear(None, VonMisesDistribution(0.0, 1.0))

    def test_validation_survives_optimized_python(self):
        env = os.environ.copy()
        src_path = os.path.abspath("src")
        env["PYTHONPATH"] = (
            src_path
            if not env.get("PYTHONPATH")
            else os.pathsep.join([src_path, env["PYTHONPATH"]])
        )

        code = """
from pyrecest.backend import array
from pyrecest.distributions import VonMisesDistribution
from pyrecest.filters.circular_fourier_filter import CircularFourierFilter

operations = (
    (lambda: setattr(CircularFourierFilter(5), "filter_state", object()), ValueError),
    (lambda: CircularFourierFilter(5, "identity").predict_identity(array([1.0] * 5)), NotImplementedError),
    (lambda: CircularFourierFilter(5).predict_identity(array([1.0] * 4)), ValueError),
    (lambda: CircularFourierFilter(5).predict_nonlinear(None, VonMisesDistribution(0.0, 1.0)), TypeError),
)
for operation, expected in operations:
    try:
        operation()
    except expected:
        pass
    else:
        raise AssertionError(f"{expected.__name__} was not raised under optimized Python")
"""
        subprocess.run([sys.executable, "-O", "-c", code], check=True, env=env)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("jax", "pytorch"),
        reason="HypertoroidalFourierFilter is not supported on this backend",
    )
    def testPredictNonlinear1D(self):
        densityInit = VonMisesDistribution(3, 5)
        fNoiseDist = VonMisesDistribution(0.5, 10)
        noCoeffs = 31

        def aGen(a):
            return lambda te: pi * (
                sin(sign(te - pi) / 2.0 * abs(te - pi) ** a / pi ** (a - 1)) + 1
            )

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
            fourierFilterNl.filter_state = (
                CircularFourierDistribution.from_distribution(
                    densityInit, noCoeffs, transformation
                )
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
            fourierFilterLin.filter_state = (
                CircularFourierDistribution.from_distribution(
                    densityInit, noCoeffs, transformation
                )
            )
            fourierFilterNl = CircularFourierFilter(noCoeffs, transformation)
            fourierFilterNl.filter_state = (
                CircularFourierDistribution.from_distribution(
                    densityInit, noCoeffs, transformation
                )
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
