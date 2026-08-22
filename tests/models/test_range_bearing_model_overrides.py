"""Regression tests for dynamic range-bearing model geometry overrides."""

import numpy.testing as npt
from pyrecest.backend import array, diag
from pyrecest.models import (
    range_bearing_jacobian,
    range_bearing_measurement,
    range_bearing_model,
)


def test_range_bearing_model_forwards_per_call_geometry_to_jacobian():
    """Measurement and Jacobian must use the same per-call sensor geometry."""
    model = range_bearing_model(
        diag(array([0.1, 0.2])),
        sensor_position=array([10.0, 10.0]),
        position_indices=(0, 1),
    )
    state = array([0.0, 0.0, 4.0, 6.0])
    sensor_position = array([1.0, 2.0])
    position_indices = (2, 3)

    expected_measurement = range_bearing_measurement(
        state,
        sensor_position=sensor_position,
        position_indices=position_indices,
    )
    expected_jacobian = range_bearing_jacobian(
        state,
        sensor_position=sensor_position,
        position_indices=position_indices,
    )

    npt.assert_allclose(
        model.evaluate(
            state,
            sensor_position=sensor_position,
            position_indices=position_indices,
        ),
        expected_measurement,
    )
    npt.assert_allclose(
        model.jacobian(
            state,
            sensor_position=sensor_position,
            position_indices=position_indices,
        ),
        expected_jacobian,
    )
