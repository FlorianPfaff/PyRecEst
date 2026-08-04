import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module
import pyrecest.backend
from pyrecest.backend import array
from pyrecest.smoothers import UnscentedRauchTungStriebelSmoother


class URTSTransitionCallContractTest(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_keyword_only_time_step_is_forwarded(self):
        calls = []

        def transition(state, *, dt):
            calls.append(dt)
            return state + dt

        result = UnscentedRauchTungStriebelSmoother._call_transition(
            transition,
            array([1.0]),
            0.5,
        )

        npt.assert_allclose(result, array([1.5]))
        self.assertEqual(calls, [0.5])

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_transition_type_error_is_not_replaced(self):
        def transition(_state, _dt):
            raise TypeError("transition failed internally")

        with self.assertRaisesRegex(TypeError, "transition failed internally"):
            UnscentedRauchTungStriebelSmoother._call_transition(
                transition,
                array([1.0]),
                0.5,
            )


if __name__ == "__main__":
    unittest.main()
