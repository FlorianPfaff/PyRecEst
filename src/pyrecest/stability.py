"""Runtime metadata helpers for public API stability."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from typing import Final, Literal, ParamSpec, TypeVar

from pyrecest.backend_support._pytorch_allclose_device_contract import (
    patch_pytorch_allclose_device_contract as _patch_pytorch_allclose_device_contract,
)
from pyrecest.backend_support._pytorch_dot_outer_device_contract import (
    patch_pytorch_dot_outer_device_contract as _patch_pytorch_dot_outer_device_contract,
)
from pyrecest.backend_support._pytorch_matmul_device_contract import (
    patch_pytorch_matmul_device_contract as _patch_pytorch_matmul_device_contract,
)
from pyrecest.backend_support._pytorch_minmax_device_contract import (
    patch_pytorch_minmax_device_contract as _patch_pytorch_minmax_device_contract,
)


def _patch_pytorch_raw_comparison_arraylike_contract() -> None:
    """Patch raw/public PyTorch comparisons to accept array-like operands."""

    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"
    helper_names = ("greater", "less", "logical_or")
    if all(
        getattr(getattr(raw_pytorch, helper_name, None), "_pyrecest_arraylike_contract", False)
        for helper_name in helper_names
    ):
        if active_pytorch_backend:
            for helper_name in helper_names:
                setattr(backend, helper_name, getattr(raw_pytorch, helper_name))
        return

    def _preferred_pytorch_device(*values):
        for value in values:
            if torch.is_tensor(value) and value.device.type != "cpu":
                return value.device
        for value in values:
            if torch.is_tensor(value):
                return value.device
        return None

    def _coerce_binary_args(x, y):
        device = _preferred_pytorch_device(x, y)
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, device=device)
        elif device is not None and x.device != device:
            x = x.to(device=device)
        if not torch.is_tensor(y):
            y = torch.as_tensor(y, device=device)
        elif device is not None and y.device != device:
            y = y.to(device=device)
        return x, y

    def _wrap_comparison(helper_name, torch_func):
        def comparison(x, y, **kwargs):
            x, y = _coerce_binary_args(x, y)
            return torch_func(x, y, **kwargs)

        comparison.__name__ = helper_name
        comparison.__doc__ = getattr(torch_func, "__doc__", None)
        comparison._pyrecest_arraylike_contract = True
        return comparison

    helpers = {
        "greater": _wrap_comparison("greater", torch.greater),
        "less": _wrap_comparison("less", torch.less),
        "logical_or": _wrap_comparison("logical_or", torch.logical_or),
    }
    for helper_name, helper in helpers.items():
        setattr(raw_pytorch, helper_name, helper)
        if active_pytorch_backend:
            setattr(backend, helper_name, helper)


def _patch_pytorch_diag_numpy_contract() -> None:
    """Patch raw/public PyTorch ``diag`` to accept NumPy-style inputs."""
    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    original_diag = getattr(raw_pytorch, "diag", None)
    if original_diag is None:
        return
    if getattr(original_diag, "_pyrecest_numpy_contract", False):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            backend.diag = original_diag
        return

    def diag(v, k=0):
        return torch.diag(raw_pytorch.array(v), diagonal=k)

    diag.__name__ = getattr(original_diag, "__name__", "diag")
    diag.__doc__ = getattr(original_diag, "__doc__", None)
    diag._pyrecest_numpy_contract = True
    raw_pytorch.diag = diag
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.diag = diag


_patch_pytorch_allclose_device_contract()
_patch_pytorch_diag_numpy_contract()
_patch_pytorch_raw_comparison_arraylike_contract()
_patch_pytorch_dot_outer_device_contract()
_patch_pytorch_matmul_device_contract()
_patch_pytorch_minmax_device_contract()

P = ParamSpec("P")
R = TypeVar("R")

StabilityLevel = Literal[
    "stable", "experimental", "deprecated", "backend-specific", "internal"
]
STABILITY_LEVELS: Final = (
    "stable",
    "experimental",
    "deprecated",
    "backend-specific",
    "internal",
)


@dataclass(frozen=True)
class PublicAPIStatus:
    """Stability metadata for a public API entry."""

    name: str
    level: StabilityLevel
    since: str | None = None
    remove_in: str | None = None
    replacement: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.level not in STABILITY_LEVELS:
            raise ValueError(f"Unknown stability level: {self.level!r}")

    def to_dict(self) -> dict[str, str | None]:
        """Return a JSON-serializable representation."""
        return asdict(self)


_PUBLIC_API_STATUS: Final[dict[str, PublicAPIStatus]] = {
    "KalmanFilter": PublicAPIStatus(
        "KalmanFilter", "stable", since="2.2.0", notes="Core linear Gaussian filter."
    ),
    "GaussianDistribution": PublicAPIStatus(
        "GaussianDistribution",
        "stable",
        since="2.2.0",
        notes="Core Euclidean distribution.",
    ),
    "pyrecest.backend": PublicAPIStatus(
        "pyrecest.backend",
        "backend-specific",
        since="2.2.0",
        notes="Support depends on the backend capability matrix.",
    ),
    "UKFOnManifolds": PublicAPIStatus(
        "UKFOnManifolds",
        "backend-specific",
        since="2.2.0",
        notes="Backend exclusions are documented in the backend API matrix.",
    ),
}


def stability(
    level: StabilityLevel,
    *,
    since: str | None = None,
    remove_in: str | None = None,
    replacement: str | None = None,
    notes: str = "",
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Attach stability metadata to a function, method, or class."""
    if level not in STABILITY_LEVELS:
        raise ValueError(f"Unknown stability level: {level!r}")

    def decorator(obj: Callable[P, R]) -> Callable[P, R]:
        status = PublicAPIStatus(
            name=f"{obj.__module__}.{obj.__qualname__}",
            level=level,
            since=since,
            remove_in=remove_in,
            replacement=replacement,
            notes=notes,
        )
        setattr(obj, "__pyrecest_stability__", status)
        return obj

    return decorator


def get_public_api_status(name: object) -> PublicAPIStatus | None:
    """Return registered stability metadata for a public API name."""
    if not isinstance(name, str):
        return None
    return _PUBLIC_API_STATUS.get(name)


def iter_public_api_status() -> Iterable[PublicAPIStatus]:
    """Iterate registered public API stability rows in stable name order."""
    return tuple(_PUBLIC_API_STATUS[name] for name in sorted(_PUBLIC_API_STATUS))


__all__ = [
    "PublicAPIStatus",
    "STABILITY_LEVELS",
    "StabilityLevel",
    "get_public_api_status",
    "iter_public_api_status",
    "stability",
]
