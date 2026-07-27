import unittest

import numpy as np
import pyrecest.backend
from pyrecest.models.linear_gaussian import (
    IdentityGaussianMeasurementModel,
    IdentityGaussianTransitionModel,
)


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="NumPy temporal scalar regression coverage is backend-specific",
)
class LinearGaussianTemporalDimensionTest(unittest.TestCase):
    def test_identity_models_reject_temporal_dimensions(self):
        temporal_dimensions = (
            np.timedelta64(2, "ns"),
            np.timedelta64(2, "us"),
            np.datetime64("1970-01-01T00:00:00.000000002", "ns"),
            np.asarray(np.timedelta64(2, "ns")),
        )

        for model_class in (
            IdentityGaussianTransitionModel,
            IdentityGaussianMeasurementModel,
        ):
            for dimension in temporal_dimensions:
                with self.subTest(model=model_class.__name__, dimension=dimension):
                    with self.assertRaisesRegex(
                        ValueError,
                        "dim must be a positive integer",
                    ):
                        model_class(dimension, noise_cov=1.0)


if __name__ == "__main__":
    unittest.main()
