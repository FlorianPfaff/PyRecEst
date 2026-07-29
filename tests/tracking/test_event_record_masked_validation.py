from __future__ import annotations

import numpy as np
import pytest
from pyrecest.tracking import TrackingEvent, event_from_measurement, record_from_update


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("measurement", np.ma.array([1.0, 2.0], mask=[False, True])),
        (
            "covariance",
            np.ma.array(np.eye(2), mask=[[False, False], [False, True]]),
        ),
    ),
)
def test_tracking_event_rejects_masked_numeric_arrays(field, value) -> None:
    kwargs = {
        "time": 0.0,
        "source": "rf",
        "measurement": [1.0, 2.0],
        "covariance": np.eye(2),
    }
    kwargs[field] = value

    with pytest.raises(
        ValueError, match=f"{field} must contain real-valued numeric entries"
    ):
        TrackingEvent(**kwargs)


def test_tracking_record_rejects_masked_values() -> None:
    event = event_from_measurement(time=0.0, source="rf")
    base = {
        "event": event,
        "prior_mean": [0.0, 0.0],
        "prior_cov": np.eye(2),
        "posterior_mean": [0.0, 0.0],
        "posterior_cov": np.eye(2),
    }
    cases = (
        (
            "prior_mean",
            np.ma.array([0.0, 1.0], mask=[False, True]),
            "prior_mean must contain real-valued numeric entries",
        ),
        (
            "innovation_cov",
            np.ma.array([[1.0]], mask=[[True]]),
            "innovation_cov must contain real-valued numeric entries",
        ),
        (
            "nis",
            np.ma.array(1.0, mask=True),
            "nis must be finite and nonnegative",
        ),
    )

    for field, value, message in cases:
        kwargs = dict(base)
        kwargs[field] = value
        with pytest.raises(ValueError, match=message):
            record_from_update(**kwargs)


def test_tracking_event_rejects_masked_scalar_fields() -> None:
    with pytest.raises(ValueError, match="time must be finite"):
        TrackingEvent(time=np.ma.array(1.0, mask=True), source="rf")

    with pytest.raises(ValueError, match="accepted must be a boolean or None"):
        TrackingEvent(
            time=0.0,
            source="rf",
            accepted=np.ma.array(True, mask=True),
        )


def test_tracking_event_accepts_fully_unmasked_masked_arrays() -> None:
    event = TrackingEvent(
        time=np.ma.array(1.0, mask=False),
        source="rf",
        measurement=np.ma.array([1.0, 2.0], mask=False),
        covariance=np.ma.array(np.eye(2), mask=False),
        accepted=np.ma.array(True, mask=False),
    )

    assert event.time == 1.0
    assert event.accepted is True
    assert np.array_equal(event.measurement, np.array([1.0, 2.0]))
    assert np.array_equal(event.covariance, np.eye(2))
