import pytest
from pyrecest.evaluation.generate_measurements import generate_n_measurements_PPP


@pytest.mark.parametrize(
    ("area", "intensity_lambda", "message"),
    [
        (-1.0, 1.0, "area"),
        (1.0, -0.5, "intensity_lambda"),
        (float("inf"), 1.0, "area"),
        (1.0, float("nan"), "intensity_lambda"),
    ],
)
def test_generate_n_measurements_ppp_rejects_invalid_rate_inputs(
    area, intensity_lambda, message
):
    with pytest.raises(ValueError, match=message):
        generate_n_measurements_PPP(area, intensity_lambda)


def test_generate_n_measurements_ppp_returns_python_int_for_zero_rate():
    count = generate_n_measurements_PPP(0.0, 123.0)

    assert count == 0
    assert isinstance(count, int)
