import unittest

import pyrecest.backend
from pyrecest.backend import array
from pyrecest.filters.spherical_harmonics_filter import SphericalHarmonicsFilter

_skip_non_numpy = unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",  # pylint: disable=no-member
    reason="SphericalHarmonicsFilter is only exercised on the numpy backend",
)


class SphericalHarmonicsFilterMeasurementValidationTest(unittest.TestCase):
    @_skip_non_numpy
    def test_update_identity_rejects_invalid_measurements(self):
        invalid_measurements = (
            (array([0.0, 0.0, 0.0]), "unit vector"),
            (array([2.0, 0.0, 0.0]), "unit vector"),
            (array([float("nan"), 0.0, 1.0]), "finite"),
            (array([1.0, 0.0]), "shape"),
            (array([[1.0, 0.0, 0.0]]), "shape"),
            (array([1.0 + 1.0j, 0.0, 0.0]), "real-valued"),
        )

        for measurement, error_fragment in invalid_measurements:
            with self.subTest(measurement=measurement, error=error_fragment):
                sh_filter = SphericalHarmonicsFilter(1)
                with self.assertRaisesRegex(ValueError, error_fragment):
                    sh_filter.update_identity(sh_filter.filter_state, measurement)


if __name__ == "__main__":
    unittest.main()
