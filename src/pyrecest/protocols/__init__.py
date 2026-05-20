"""Public protocol contracts for PyRecEst extension points.

The :mod:`pyrecest.protocols` package contains small, runtime-checkable
capability protocols for user-defined PyRecEst components. Protocols are public
contracts: they describe the methods, attributes, and conventions that later
modules can rely on without forcing users to inherit from a specific abstract
base class.
"""

from .common import ArrayLike, BackendArray, SupportsDim, SupportsInputDim
from .distributions import (
    SupportsCovariance,
    SupportsDistributionApproximation,
    SupportsDistributionConversion,
    SupportsLogPdf,
    SupportsMean,
    SupportsMeanAndCovariance,
    SupportsPdf,
    SupportsSampling,
)
from .filters import (
    SupportsFilterState,
    SupportsHistoryRecording,
    SupportsLinearPrediction,
    SupportsLinearUpdate,
    SupportsPointEstimate,
)

__all__ = [
    "ArrayLike",
    "BackendArray",
    "SupportsCovariance",
    "SupportsDim",
    "SupportsDistributionApproximation",
    "SupportsDistributionConversion",
    "SupportsFilterState",
    "SupportsHistoryRecording",
    "SupportsInputDim",
    "SupportsLinearPrediction",
    "SupportsLinearUpdate",
    "SupportsLogPdf",
    "SupportsMean",
    "SupportsMeanAndCovariance",
    "SupportsPdf",
    "SupportsPointEstimate",
    "SupportsSampling",
]
