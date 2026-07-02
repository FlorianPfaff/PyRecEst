import unittest

from pyrecest.backend import allclose, array
from pyrecest.models import DensityTransitionModel, SampleableTransitionModel


class UnaryTransitionSamplerCountContractTest(unittest.TestCase):
    def test_sampleable_unary_sampler_allows_default_count(self):
        model = SampleableTransitionModel(lambda state: state + array([1.0]))

        self.assertTrue(allclose(model.sample_next(array([10.0])), array([11.0])))

    def test_sampleable_unary_sampler_rejects_multiple_samples(self):
        model = SampleableTransitionModel(lambda state: state + array([1.0]))

        with self.assertRaisesRegex(ValueError, "does not accept an n argument"):
            model.sample_next(array([10.0]), n=2)

    def test_density_unary_sampler_allows_default_count(self):
        model = DensityTransitionModel(
            lambda state_next, state_previous: state_next + state_previous,
            sample_next=lambda state: state + array([1.0]),
        )

        self.assertTrue(allclose(model.sample_next(array([4.0])), array([5.0])))

    def test_density_unary_sampler_rejects_multiple_samples(self):
        model = DensityTransitionModel(
            lambda state_next, state_previous: state_next + state_previous,
            sample_next=lambda state: state + array([1.0]),
        )

        with self.assertRaisesRegex(ValueError, "does not accept an n argument"):
            model.sample_next(array([4.0]), n=2)


if __name__ == "__main__":
    unittest.main()
