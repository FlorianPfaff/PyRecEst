import pytest

from tests.support.backend_runner import run_backend_code


def test_pytorch_random_device_helpers_prefer_existing_non_cpu_tensor():
    torch = pytest.importorskip("torch")

    import pyrecest  # noqa: F401  # Triggers runtime backend compatibility hooks.
    from pyrecest._backend.pytorch import random as torch_random

    cpu_tensor = torch.empty(2)
    meta_tensor = torch.empty(2, device="meta")

    assert torch_random._preferred_tensor_device(cpu_tensor, meta_tensor).type == "meta"
    assert torch_random._randint_device(cpu_tensor, meta_tensor).type == "meta"
    assert torch_random._normal_device(cpu_tensor, meta_tensor).type == "meta"
    assert torch_random._tensor_device(cpu_tensor, meta_tensor).type == "meta"


def test_pytorch_random_device_helpers_honor_explicit_device_override():
    torch = pytest.importorskip("torch")

    import pyrecest  # noqa: F401  # Triggers runtime backend compatibility hooks.
    from pyrecest._backend.pytorch import random as torch_random

    cpu_tensor = torch.empty(2)
    meta_tensor = torch.empty(2, device="meta")
    explicit_device = torch.device("cpu")

    assert (
        torch_random._preferred_tensor_device(
            cpu_tensor,
            meta_tensor,
            device=explicit_device,
        )
        == explicit_device
    )
    assert (
        torch_random._randint_device(
            cpu_tensor,
            meta_tensor,
            device=explicit_device,
        )
        == explicit_device
    )


def test_public_pytorch_random_facade_uses_patched_uniform():
    pytest.importorskip("torch")

    code = """
import pyrecest  # noqa: F401
from pyrecest._backend.pytorch import random as raw_random
from pyrecest.backend import random as public_random

assert getattr(raw_random.uniform, "_pyrecest_device_contract", False)
assert public_random.uniform is raw_random.uniform
"""
    result = run_backend_code("pytorch", code)

    assert result.returncode == 0, result.stderr
