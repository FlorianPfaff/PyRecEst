import numpy.testing as npt
from pyrecest.backend import array
from pyrecest.examples.scgp_measurement_reliability import (
    run_scgp_measurement_reliability_example,
)


def test_scgp_measurement_reliability_example_runs():
    result = run_scgp_measurement_reliability_example()

    assert result.unweighted_active_measurement_indices == (0, 1, 2)
    assert result.weighted_active_measurement_indices == (0, 1)
    assert result.masked_active_measurement_indices == (0, 1)
    npt.assert_allclose(result.measurement_weights, array([1.0, 0.25, 0.0]))


def test_scgp_measurement_reliability_example_reports_position_shifts():
    result = run_scgp_measurement_reliability_example()

    assert result.unweighted_position_shift >= 0.0
    assert result.weighted_position_shift >= 0.0
    assert result.masked_position_shift >= 0.0
    assert result.weighted_quadratic_form is not None
    assert result.masked_quadratic_form is not None
    assert result.unweighted_quadratic_form is not None
