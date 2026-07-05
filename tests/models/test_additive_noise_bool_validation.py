import unittest

from pyrecest.models import AdditiveNoiseMeasurementModel, AdditiveNoiseTransitionModel


class UncoercibleBoolFlag:
    def __array__(self, dtype=None):
        del dtype
        raise TypeError("cannot convert")


class AdditiveNoiseBoolFlagValidationTest(unittest.TestCase):
    def test_vectorized_flag_wraps_uncoercible_array_errors(self):
        flag = UncoercibleBoolFlag()

        with self.assertRaisesRegex(TypeError, "vectorized"):
            AdditiveNoiseTransitionModel(lambda x: x, vectorized=flag)

        transition_model = AdditiveNoiseTransitionModel(lambda x: x)
        with self.assertRaisesRegex(TypeError, "vectorized"):
            transition_model.vectorized = flag
        with self.assertRaisesRegex(TypeError, "function_is_vectorized"):
            transition_model.function_is_vectorized = flag

        with self.assertRaisesRegex(TypeError, "vectorized"):
            AdditiveNoiseMeasurementModel(lambda x: x, vectorized=flag)


if __name__ == "__main__":
    unittest.main()
