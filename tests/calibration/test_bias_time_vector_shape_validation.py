import numpy as np
import pytest
from pyrecest.calibration.bias import make_bias_training_examples


@pytest.mark.parametrize(
    ("field", "matrix"),
    [
        ("measurement_times_s", np.array([[0.0, 1.0]])),
        ("measurement_times_s", np.array([[0.0], [1.0]])),
        ("reference_times_s", np.array([[0.0, 1.0]])),
        ("reference_times_s", np.array([[0.0], [1.0]])),
    ],
)
def test_make_bias_training_examples_rejects_matrix_time_vectors(field, matrix):
    kwargs = {
        "measurement_times_s": np.array([0.0, 1.0]),
        "measurement_values": np.array([[1.0], [2.0]]),
        "reference_times_s": np.array([0.0, 1.0]),
        "reference_values": np.array([[0.0], [1.0]]),
    }
    kwargs[field] = matrix

    with pytest.raises(ValueError, match=rf"{field} must be one-dimensional"):
        make_bias_training_examples(**kwargs)


def test_make_bias_training_examples_preserves_scalar_time_support():
    examples = make_bias_training_examples(
        measurement_times_s=0.0,
        measurement_values=np.array([[2.0]]),
        reference_times_s=0.0,
        reference_values=np.array([[1.0]]),
        max_time_delta_s=0.0,
    )

    np.testing.assert_allclose(examples.residual, np.array([[1.0]]))
