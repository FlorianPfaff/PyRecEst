import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import array
from pyrecest.smoothers import UnscentedRauchTungStriebelSmoother


class URTSKeywordOnlyTimeStepTest(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_keyword_time_parameters_receive_configured_step(self):
        smoother = UnscentedRauchTungStriebelSmoother()
        sigma_point = array([1.0])

        def transition_with_dt(x, *, dt):
            return x + dt

        def transition_with_time_step(x, *, time_step):
            return x + time_step

        def transition_with_kwargs(x, **kwargs):
            return x + kwargs["dt"]

        for transition in (
            transition_with_dt,
            transition_with_time_step,
            transition_with_kwargs,
        ):
            with self.subTest(transition=transition.__name__):
                npt.assert_allclose(
                    smoother._call_transition(transition, sigma_point, 0.5),
                    array([1.5]),
                )


if __name__ == "__main__":
    unittest.main()
