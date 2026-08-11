import unittest

import numpy as np
from pyrecest.distributions.abstract_mixture import _validate_mixture_weight_values


class AbstractMixtureTemporalWeightTest(unittest.TestCase):
    def test_native_temporal_weight_arrays_are_rejected(self):
        temporal_weights = [
            np.array([1, 2], dtype="timedelta64[ns]"),
            np.array([1, 2], dtype="timedelta64[us]"),
            np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[D]"),
        ]

        for weights in temporal_weights:
            with self.subTest(dtype=str(weights.dtype)):
                with self.assertRaisesRegex(ValueError, "real-valued numeric"):
                    _validate_mixture_weight_values(weights)

    def test_temporal_scalars_in_object_weight_arrays_are_rejected(self):
        temporal_weights = [
            np.array([np.timedelta64(1, "ns"), 1.0], dtype=object),
            np.array([np.datetime64("2026-01-01"), 1.0], dtype=object),
        ]

        for weights in temporal_weights:
            with self.subTest(weight=repr(weights[0])):
                with self.assertRaisesRegex(ValueError, "real-valued numeric"):
                    _validate_mixture_weight_values(weights)


if __name__ == "__main__":
    unittest.main()
