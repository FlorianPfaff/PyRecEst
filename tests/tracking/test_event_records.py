from __future__ import annotations

import json

import numpy as np
import pytest
from pyrecest.tracking import (
    TrackingEvent,
    action_counts,
    event_from_measurement,
    record_from_update,
    records_to_dicts,
    records_to_matrix,
)


def test_tracking_event_validates_measurement_covariance_shape() -> None:
    event = TrackingEvent(
        time=1.5,
        source="radar",
        action="update",
        measurement=np.array([1.0, 2.0]),
        covariance=np.eye(2),
        accepted=True,
    )

    assert event.measurement_dim == 2
    assert event.to_dict()["source"] == "radar"

    with pytest.raises(ValueError, match="covariance must match"):
        TrackingEvent(
            time=0.0, source="rf", measurement=np.ones(2), covariance=np.eye(3)
        )


def test_record_from_update_preserves_prior_posterior_and_legacy_aliases() -> None:
    event = event_from_measurement(
        time=2.0,
        source="rf",
        measurement=[10.0, -1.0],
        covariance=np.diag([4.0, 4.0]),
        accepted=False,
        action="reject",
        metadata={"sensor": "keysight"},
    )
    prior_mean = np.array([0.0, 0.0, 0.0, 1.0])
    posterior_mean = np.array([1.0, 0.0, 0.0, 1.0])
    record = record_from_update(
        event=event,
        prior_mean=prior_mean,
        prior_cov=np.eye(4),
        posterior_mean=posterior_mean,
        posterior_cov=2.0 * np.eye(4),
        innovation=[10.0, -1.0],
        innovation_cov=np.eye(2),
        nis=3.0,
    )

    as_dict = record.to_dict(include_legacy_aliases=True)
    assert record.state_dim == 4
    assert as_dict["time_s"] == 2.0
    assert np.allclose(as_dict["state"], posterior_mean)
    assert np.allclose(as_dict["prior_mean"], prior_mean)
    assert as_dict["event_metadata"] == {"sensor": "keysight"}
    json.dumps(as_dict)


def test_records_to_dicts_matrix_and_action_counts() -> None:
    event = event_from_measurement(time=0.0, source="radar", action="update")
    record_a = record_from_update(
        event=event,
        prior_mean=[0.0, 0.0],
        prior_cov=np.eye(2),
        posterior_mean=[1.0, 0.0],
        posterior_cov=np.eye(2),
        action="updated",
    )
    record_b = record_from_update(
        event=event_from_measurement(time=1.0, source="rf", action="coast"),
        prior_mean=[1.0, 0.0],
        prior_cov=np.eye(2),
        posterior_mean=[2.0, 0.0],
        posterior_cov=np.eye(2),
    )

    assert records_to_matrix([record_a, record_b]).shape == (2, 2)
    as_dicts = records_to_dicts([record_a, record_b])
    assert len(as_dicts) == 2
    json.dumps(as_dicts)
    assert action_counts([record_a, record_b]) == {"updated": 1, "coast": 1}
