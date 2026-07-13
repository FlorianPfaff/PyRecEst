import unittest

import numpy as np
import numpy.testing as npt

import pyrecest.backend
from pyrecest.backend import array
from pyrecest.utils.point_set_registration import estimate_transform


class TestPointSetRegistrationWeightOverflow(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_estimate_transform_normalizes_maximum_finite_weights(self):
        source = array([[0.0, 0.0], [2.0, 1.0]])
        true_offset = array([3.0, -4.0])
        target = source + true_offset

        backend_dtype = pyrecest.backend.to_numpy(array([1.0])).dtype
        maximum_finite_weight = np.finfo(backend_dtype).max

        estimated = estimate_transform(
            source,
            target,
            model="translation",
            weights=array([maximum_finite_weight, maximum_finite_weight]),
        )

        npt.assert_allclose(
            pyrecest.backend.to_numpy(estimated.offset),
            pyrecest.backend.to_numpy(true_offset),
            rtol=1e-6,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
