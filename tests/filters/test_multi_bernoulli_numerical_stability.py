from unittest.mock import patch

import numpy as np
import pyrecest.backend
import pytest
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.multi_bernoulli_tracker import (
    BernoulliComponent,
    MultiBernoulliTracker,
)

pytestmark = pytest.mark.skipif(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="MultiBernoulliTracker is only supported by the NumPy backend.",
)


def test_detection_existence_update_avoids_overflow():
    tracker = MultiBernoulliTracker(
        initial_prior=[
            BernoulliComponent(
                0.8,
                GaussianDistribution(np.zeros(1), np.eye(1)),
            )
        ],
        tracker_param={
            "detection_probability": 1.0,
            "clutter_intensity": 1e308,
            "gating_distance_threshold": np.inf,
            "pruning_threshold": 0.0,
        },
    )

    with patch.object(
        tracker,
        "_measurement_likelihood_and_distance",
        return_value=(1e308, 0.0),
    ):
        tracker.update_linear(
            np.array([[0.0]]),
            np.array([[1.0]]),
            np.array([[1.0]]),
        )

    association_to_clutter_ratio = 0.8 * (1.0 - 1e-12)
    expected_existence = association_to_clutter_ratio / (
        1.0 + association_to_clutter_ratio
    )

    assert tracker.get_number_of_components() == 1
    assert tracker.bernoulli_components[0].existence_probability == pytest.approx(
        expected_existence
    )
