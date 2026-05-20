from importlib.metadata import PackageNotFoundError, version

import pyrecest._backend  # noqa
from pyrecest.backend import copy  # noqa: F401
from pyrecest.backend_tools import (  # noqa: F401
    assert_backend,
    get_backend_name,
    is_backend,
    warn_if_backend_env_changed,
)

try:
    __version__ = version("pyrecest")
except PackageNotFoundError:  # pragma: no cover - source tree without install metadata
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "assert_backend",
    "copy",
    "get_backend_name",
    "is_backend",
    "warn_if_backend_env_changed",
]
