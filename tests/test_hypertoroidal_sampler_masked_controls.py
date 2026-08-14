import unittest

import numpy as np
from pyrecest.sampling.hypertoroidal_sampler import (
    CircularUniformSampler,
    _validate_integral_scalar,
)


class TestHypertoroidalSamplerMaskedControls(unittest.TestCase):
    def setUp(self):
        self.sampler = CircularUniformSampler()

    def test_masked_integer_controls_are_rejected_before_item_unwrap(self):
        invalid_values = (
            np.ma.masked,
            np.ma.array(3, mask=True),
        )

        for value in invalid_values:
            with self.subTest(value=repr(value)):
                with self.assertRaisesRegex(ValueError, "must be an integer"):
                    _validate_integral_scalar(value, "value", minimum=0)
                with self.assertRaisesRegex(ValueError, "must be an integer"):
                    self.sampler.sample_stochastic(value)
                with self.assertRaisesRegex(ValueError, "must be an integer"):
                    self.sampler.get_grid(value)

        with self.assertRaisesRegex(ValueError, "must be an integer"):
            self.sampler.sample_stochastic(2, dim=np.ma.array(1, mask=True))

    def test_object_wrapped_temporal_controls_are_rejected(self):
        temporal_value = np.array(np.timedelta64(3, "ns"), dtype=object)

        with self.assertRaisesRegex(ValueError, "must be an integer"):
            _validate_integral_scalar(temporal_value, "value", minimum=0)
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            self.sampler.sample_stochastic(temporal_value)
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            self.sampler.get_grid(temporal_value)

        temporal_dim = np.array(np.timedelta64(1, "ns"), dtype=object)
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            self.sampler.sample_stochastic(2, dim=temporal_dim)

    def test_fully_unmasked_masked_scalar_remains_supported(self):
        self.assertEqual(
            _validate_integral_scalar(
                np.ma.array(3, mask=False),
                "value",
                minimum=0,
            ),
            3,
        )


if __name__ == "__main__":
    unittest.main()
