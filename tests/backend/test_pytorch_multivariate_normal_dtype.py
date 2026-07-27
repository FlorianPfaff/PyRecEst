import pytest

torch = pytest.importorskip("torch")

from pyrecest._backend.pytorch import random  # noqa: E402


@pytest.mark.parametrize(
    ("mean_dtype", "cov_dtype"),
    [
        (torch.float32, torch.float64),
        (torch.float64, torch.float32),
    ],
)
def test_multivariate_normal_promotes_mixed_tensor_precision(mean_dtype, cov_dtype):
    random.seed(0)
    mean = torch.tensor([0.0, 0.0], dtype=mean_dtype)
    covariance = torch.eye(2, dtype=cov_dtype)

    samples = random.multivariate_normal(mean, covariance, size=(4,))

    assert samples.dtype == torch.float64
    assert samples.shape == (4, 2)
