from importlib.metadata import PackageNotFoundError, version

import pyrecest._backend  # noqa
from pyrecest.backend import copy  # noqa: F401
from pyrecest.backend_support import (  # noqa: F401
    backend_support,
    format_backend_support_markdown,
    get_backend_support,
)
from pyrecest.backend_tools import (  # noqa: F401
    assert_backend,
    get_backend_name,
    is_backend,
    warn_if_backend_env_changed,
)
from pyrecest.evidence import (  # noqa: F401
    EvidenceComputationMode,
    resolve_evidence_computation_mode,
)
from pyrecest.exceptions import (  # noqa: F401
    BackendNotSupportedError,
    BackendSupportError,
    DimensionMismatchError,
    NumericalStabilityError,
    OptionalDependencyError,
    PyRecEstError,
    ShapeError,
    ValidationError,
)
from pyrecest.stability import (  # noqa: F401
    get_public_api_status,
    iter_public_api_status,
    stability,
)

try:
    __version__ = version("pyrecest")
except PackageNotFoundError:  # pragma: no cover - source tree without install metadata
    __version__ = "0+unknown"

__all__ = [
    "BackendNotSupportedError",
    "BackendSupportError",
    "DimensionMismatchError",
    "EvidenceComputationMode",
    "NumericalStabilityError",
    "OptionalDependencyError",
    "PyRecEstError",
    "ShapeError",
    "ValidationError",
    "__version__",
    "assert_backend",
    "backend_support",
    "copy",
    "format_backend_support_markdown",
    "get_backend_name",
    "get_backend_support",
    "get_public_api_status",
    "is_backend",
    "iter_public_api_status",
    "stability",
    "warn_if_backend_env_changed",
    "resolve_evidence_computation_mode",
]
