import unittest

from pyrecest.backend import zeros
from pyrecest.models.validation import infer_state_dim_from_distribution


class TestCallableDiracDimensionInference(unittest.TestCase):
    def test_infer_state_dim_from_callable_dirac_locations(self):
        class DistributionWithCallableDiracs:
            def d(self):
                return zeros((4, 2))

        self.assertEqual(
            infer_state_dim_from_distribution(DistributionWithCallableDiracs()), 2
        )

    def test_disabled_methods_do_not_call_callable_dirac_locations(self):
        class DistributionWithCallableDiracs:
            def d(self):
                raise RuntimeError("unexpected d call")

        with self.assertRaisesRegex(ValueError, "Could not infer"):
            infer_state_dim_from_distribution(
                DistributionWithCallableDiracs(), allow_methods=False
            )


if __name__ == "__main__":
    unittest.main()
