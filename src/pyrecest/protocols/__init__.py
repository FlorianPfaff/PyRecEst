"""Public protocol contracts for PyRecEst extension points.

The :mod:`pyrecest.protocols` package contains small, runtime-checkable
capability protocols for user-defined PyRecEst components. Protocols are public
contracts: they describe the methods, attributes, and conventions that later
modules can rely on without forcing users to inherit from a specific abstract
base class.

This seed package intentionally exposes only common aliases and dimension
protocols. Distribution-, filter-, model-, conversion-, and manifold-specific
protocol modules can be added independently in follow-up pull requests.
"""

from .common import ArrayLike, BackendArray, SupportsDim, SupportsInputDim

__all__ = [
    "ArrayLike",
    "BackendArray",
    "SupportsDim",
    "SupportsInputDim",
]
