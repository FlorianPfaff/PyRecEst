import numpy as np
import pytest

from pyrecest.calibration.bias import make_bias_training_examples


@pytest.mark.parametrize(
    ("field_name", "matrix_times"),
    [
        ("measurement_times_s", np.array([[0.0, 1.0]])),
        ("measurement_times_s", np.array([[0.0], [1.0]])),
        ("reference_times_s", np.array([[0.0, 1.0]])),
        ("reference_times_s", np.array([[0.0], [1.0]])),
    ],
)
def test_make_bias_training_examples_rejects_matrix_timestamps(
    field_name, matrix_times
):
    kwargs = {
        "measurement_times_s": np.array([0.0, 1.0]),
        "measurement_values": np.array([[1.0], [2.0]]),
        "reference_times_s": np.array([0.0, 1.0]),
        "reference_values": np.array([[0.5], [1.5]]),
        "max_time_delta_s": 0.0,
    }
    kwargs[field_name] = matrix_times

    with pytest.raises(ValueError, match=rf"{field_name} must be one-dimensional"):
        make_bias_training_examples(**kwargs)


def test_make_bias_training_examples_keeps_scalar_single_timestamp_support():
    examples = make_bias_training_examples(
        measurement_times_s=0.0,
        measurement_values=np.array([[2.0]]),
        reference_times_s=0.0,
        reference_values=np.array([[1.0]]),
        max_time_delta_s=0.0,
    )

    np.testing.assert_allclose(examples.residual, np.array([[1.0]]))
