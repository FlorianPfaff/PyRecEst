class Rotation:
    """Dummy Rotation for PyTorch backend."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "Rotation is not supported on the PyTorch backend. "
            "Use a NumPy/JAX backend for this functionality."
        )

    @classmethod
    def from_quat(cls, *args, **kwargs):
        raise RuntimeError(
            "Rotation.from_quat is not supported on the PyTorch backend. "
            "Use a NumPy/JAX backend for this functionality."
        )

    def as_quat(self, *args, **kwargs):
        raise RuntimeError(
            "Rotation.as_quat is not supported on the PyTorch backend. "
            "Use a NumPy/JAX backend for this functionality."
        )

    def __mul__(self, other):
        raise RuntimeError(
            "Rotation composition is not supported on the PyTorch backend. "
            "Use a NumPy/JAX backend for this functionality."
        )
