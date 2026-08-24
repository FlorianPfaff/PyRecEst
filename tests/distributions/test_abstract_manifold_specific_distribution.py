import unittest
from math import log as math_log
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, to_numpy
from pyrecest.distributions.abstract_manifold_specific_distribution import (
    AbstractManifoldSpecificDistribution,
)


class DeterministicOneDimensionalDistribution(AbstractManifoldSpecificDistribution):
    def __init__(self):
        super().__init__(dim=1)

    @property
    def input_dim(self):
        return 1

    def get_manifold_size(self):
        return 1.0

    def pdf(self, xs):
        value = float(to_numpy(xs).squeeze())
        if value == 0.0:
            return array(1.0)
        if value == 1.0:
            return array(0.1)
        return array(value)

    def mean(self):
        return array([0.0])


class AbstractManifoldSpecificDistributionTest(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="This regression test targets the non-JAX MH implementation.",
    )
    def test_sample_metropolis_hastings_records_rejections(self):
        distribution = DeterministicOneDimensionalDistribution()
        proposals = iter([array([1.0]), array([2.0]), array([3.0])])

        def proposal(_):
            return next(proposals)

        with patch(
            "pyrecest.distributions.abstract_manifold_specific_distribution.random.rand",
            return_value=0.5,
        ):
            samples = distribution.sample_metropolis_hastings(
                n=2, burn_in=0, skipping=1, proposal=proposal, start_point=array([0.0])
            )

        npt.assert_allclose(to_numpy(samples), [0.0, 2.0])

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="This regression test targets the non-JAX MH implementation.",
    )
    def test_sample_metropolis_hastings_uses_asymmetric_proposal_correction(self):
        distribution = DeterministicOneDimensionalDistribution()
        proposals = iter([array([1.0])])

        def proposal(_):
            return next(proposals)

        def proposal_log_pdf(candidate, current):
            candidate_value = float(to_numpy(candidate).squeeze())
            current_value = float(to_numpy(current).squeeze())

            if current_value == 0.0 and candidate_value == 1.0:
                return math_log(0.1)

            if current_value == 1.0 and candidate_value == 0.0:
                return math_log(0.6)

            return 0.0

        with patch(
            "pyrecest.distributions.abstract_manifold_specific_distribution.random.rand",
            return_value=0.5,
        ):
            samples = distribution.sample_metropolis_hastings(
                n=1,
                burn_in=0,
                skipping=1,
                proposal=proposal,
                start_point=array([0.0]),
                proposal_log_pdf=proposal_log_pdf,
            )

        npt.assert_allclose(to_numpy(samples), 1.0)

    def test_sample_metropolis_hastings_rejects_wrong_proposal_shape(self):
        distribution = DeterministicOneDimensionalDistribution()

        if pyrecest.backend.__backend_name__ == "jax":

            def proposal(_key, _):
                return array([1.0, 2.0])

        else:

            def proposal(_):
                return array([1.0, 2.0])

        with self.assertRaisesRegex(ValueError, "same shape"):
            distribution.sample_metropolis_hastings(
                n=1, burn_in=0, skipping=1, proposal=proposal, start_point=array([0.0])
            )

    def test_sample_metropolis_hastings_validates_count_parameters(self):
        distribution = DeterministicOneDimensionalDistribution()

        def proposal(x):
            return x

        invalid_parameters = (
            {"n": 0},
            {"n": 1.5},
            {"n": True},
            {"n": [1]},
            {"n": np.array([1])},
            {"n": array([1])},
            {"n": "1"},
            {"n": np.timedelta64(1, "ns")},
            {"n": np.datetime64("1970-01-01T00:00:00.000000001")},
            {"n": np.array(np.timedelta64(1, "ns"), dtype=object)},
            {"burn_in": -1},
            {"burn_in": False},
            {"burn_in": np.array([0])},
            {"burn_in": np.timedelta64(0, "ns")},
            {"skipping": 0},
            {"skipping": 1.5},
            {"skipping": True},
            {"skipping": "1"},
            {"skipping": np.timedelta64(1, "ns")},
        )

        for overrides in invalid_parameters:
            kwargs = {
                "n": np.array(1.0),
                "burn_in": np.array(0.0),
                "skipping": np.array(1.0),
                "proposal": proposal,
                "start_point": array([0.0]),
            }
            kwargs.update(overrides)
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                distribution.sample_metropolis_hastings(**kwargs)


if __name__ == "__main__":
    unittest.main()
