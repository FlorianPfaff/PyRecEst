import pytest


def test_pytorch_fftconvolve_uses_existing_accelerator_tensor_device():
    torch = pytest.importorskip("torch")
    from pyrecest._backend.pytorch import signal as pytorch_signal

    cpu_input = torch.ones(3)
    accelerator_input = torch.ones(3, device="meta")

    first, second = pytorch_signal._as_tensor_pair(cpu_input, accelerator_input)

    assert first.device.type == "meta"
    assert second.device.type == "meta"
