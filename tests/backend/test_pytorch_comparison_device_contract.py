import pytest

torch = pytest.importorskip("torch")

import pyrecest._backend.pytorch as pytorch_backend  # noqa: E402
import pyrecest.backend_tools  # noqa: E402,F401


def _non_cpu_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        torch.empty((), device="meta")
    except Exception as exc:  # pragma: no cover - depends on PyTorch build
        pytest.skip(f"no non-CPU PyTorch test device available: {exc}")
    return torch.device("meta")


@pytest.mark.parametrize(
    ("helper_name", "left", "right_factory"),
    [
        (
            "greater",
            torch.tensor([1.0, 2.0]),
            lambda device: torch.ones(2, device=device),
        ),
        (
            "less",
            torch.tensor([1.0, 2.0]),
            lambda device: torch.full((2,), 3.0, device=device),
        ),
        (
            "logical_or",
            torch.tensor([True, False]),
            lambda device: torch.tensor([False, True], device=device),
        ),
    ],
)
def test_raw_pytorch_comparison_helpers_prefer_existing_non_cpu_device(
    helper_name,
    left,
    right_factory,
):
    device = _non_cpu_device()

    result = getattr(pytorch_backend, helper_name)(left, right_factory(device))

    assert result.device.type == device.type
    assert result.dtype == torch.bool
    assert tuple(result.shape) == (2,)


def test_raw_pytorch_isclose_prefers_existing_non_cpu_device_for_right_operand():
    device = _non_cpu_device()

    result = pytorch_backend.isclose(
        torch.tensor([1.0, 2.0]),
        torch.ones(2, device=device),
        equal_nan=True,
    )

    assert result.device.type == device.type
    assert result.dtype == torch.bool
    assert tuple(result.shape) == (2,)
