import unittest

import numpy as np
from pyrecest.backend import allclose, array, diag
from pyrecest.distributions import GaussianDistribution
from pyrecest.models import (
    AdditiveNoiseMeasurementModel,
    AdditiveNoiseTransitionModel,
    SampleableTransitionModel,
)


class ModelObjectTest(unittest.TestCase):
    def test_transition_model_uses_distribution_covariance_and_default_args(self):
        noise_cov = diag(array([0.1, 0.2]))

        def fx(x, dt, scale=1.0):
            return scale * (x + dt)

        model = AdditiveNoiseTransitionModel(
            transition_function=fx,
            noise_distribution=GaussianDistribution(array([0.0, 0.0]), noise_cov),
            dt=0.5,
            function_args={"scale": 2.0},
        )

        self.assertTrue(allclose(model.noise_covariance, noise_cov))
        self.assertTrue(allclose(model.evaluate(array([1.0, 2.0])), array([3.0, 5.0])))
        self.assertTrue(
            allclose(model.evaluate(array([1.0, 2.0]), scale=1.0), array([1.5, 2.5]))
        )

    def test_explicit_transition_dt_overrides_function_args_dt(self):
        def fx(x, dt, scale=1.0):
            return scale * (x + dt)

        model = AdditiveNoiseTransitionModel(
            transition_function=fx,
            dt=0.5,
            function_args={"dt": 10.0, "scale": 2.0},
        )

        self.assertTrue(allclose(model.evaluate(array([1.0, 2.0])), array([3.0, 5.0])))
        self.assertTrue(
            allclose(model.evaluate(array([1.0, 2.0]), dt=0.25), array([2.5, 4.5]))
        )

    def test_measurement_model_accepts_covariance_like_noise(self):
        noise_cov = diag(array([0.3, 0.4]))

        def hx(x, offset=0.0):
            return array([x[0] + offset, x[1] - offset])

        model = AdditiveNoiseMeasurementModel(
            measurement_function=hx,
            noise_distribution=noise_cov,
            function_args={"offset": 0.25},
        )

        self.assertTrue(allclose(model.noise_covariance, noise_cov))
        self.assertTrue(
            allclose(model.evaluate(array([1.0, 2.0])), array([1.25, 1.75]))
        )

    def test_sampleable_transition_model_validates_vectorized_flag(self):
        self.assertTrue(
            SampleableTransitionModel(
                lambda value: value,
                function_is_vectorized=np.bool_(True),
            ).function_is_vectorized
        )
        self.assertFalse(
            SampleableTransitionModel(
                lambda value: value,
                function_is_vectorized=False,
            ).function_is_vectorized
        )

        for invalid_flag in ("False", 1):
            with self.subTest(invalid_flag=invalid_flag):
                with self.assertRaises(TypeError):
                    SampleableTransitionModel(
                        lambda value: value,
                        function_is_vectorized=invalid_flag,
                    )


if __name__ == "__main__":
    unittest.main()
