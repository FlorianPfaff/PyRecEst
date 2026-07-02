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

_patch_pytorch_allclose_device_contract()
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
