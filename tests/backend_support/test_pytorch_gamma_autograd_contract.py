import pytest

torch = pytest.importorskip("torch")


def test_pytorch_gamma_has_finite_gradients_at_positive_integers():
    import pyrecest._backend.pytorch as pytorch_backend
    import pyrecest.backend as backend
    from pyrecest.backend_support._pytorch_one_hot_scalar_contract import (
        _patch_pytorch_gamma_autograd_contract,
    )

    _patch_pytorch_gamma_autograd_contract(
        pytorch_backend,
        backend,
        torch,
    )

    values = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    result = pytorch_backend.gamma(values)
    result.sum().backward()

    assert torch.allclose(result, torch.tensor([1.0, 1.0, 2.0]))
    assert torch.all(torch.isfinite(values.grad))


def test_pytorch_gamma_keeps_negative_noninteger_gradients_finite():
    import pyrecest._backend.pytorch as pytorch_backend
    import pyrecest.backend as backend
    from pyrecest.backend_support._pytorch_one_hot_scalar_contract import (
        _patch_pytorch_gamma_autograd_contract,
    )

    _patch_pytorch_gamma_autograd_contract(
        pytorch_backend,
        backend,
        torch,
    )

    value = torch.tensor(-0.5, requires_grad=True)
    result = pytorch_backend.gamma(value)
    result.backward()

    assert torch.isfinite(result)
    assert torch.isfinite(value.grad)
