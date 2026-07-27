import unittest

import numpy as np
import pyrecest.backend
from pyrecest.backend import array
from pyrecest.utils.point_set_registration import joint_registration_assignment


class TestPointRegistrationTemporalCountValidation(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Point-set registration is not supported on this backend",
    )
    def test_rejects_numpy_temporal_count_controls(self):
        points = array([[0.0, 0.0], [1.0, 0.0]])
        invalid_values = (
            np.timedelta64(3, "ns"),
            np.datetime64("1970-01-01T00:00:00.000000003", "ns"),
            np.array(np.timedelta64(3, "ns"), dtype=object),
            np.array(
                np.datetime64("1970-01-01T00:00:00.000000003", "ns"),
                dtype=object,
            ),
        )

        for parameter_name in ("max_iterations", "min_matches"):
            for invalid_value in invalid_values:
                with self.subTest(
                    parameter_name=parameter_name,
                    invalid_value=invalid_value,
                ):
                    with self.assertRaisesRegex(
                        ValueError,
                        f"{parameter_name} must be a scalar integer",
                    ):
                        joint_registration_assignment(
                            points,
                            points,
                            model="translation",
                            **{parameter_name: invalid_value},
                        )


if __name__ == "__main__":
    unittest.main()
