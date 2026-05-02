"""Import compatibility checks for the public protocol rollout."""

from __future__ import annotations

from pyrecest.models import (
    SupportsLikelihood as ModelSupportsLikelihood,
    SupportsLogLikelihood as ModelSupportsLogLikelihood,
    SupportsTransitionDensity as ModelSupportsTransitionDensity,
    SupportsTransitionSampling as ModelSupportsTransitionSampling,
)
from pyrecest.models.likelihood import (
    SupportsLikelihood as LikelihoodSupportsLikelihood,
    SupportsLogLikelihood as LikelihoodSupportsLogLikelihood,
    SupportsTransitionDensity as LikelihoodSupportsTransitionDensity,
    SupportsTransitionSampling as LikelihoodSupportsTransitionSampling,
)
from pyrecest.protocols import SupportsDim as PackageSupportsDim
from pyrecest.protocols.common import SupportsDim as CommonSupportsDim


def test_common_protocol_package_import_remains_available():
    assert PackageSupportsDim is CommonSupportsDim


def test_existing_model_protocol_imports_remain_available():
    assert ModelSupportsLikelihood is LikelihoodSupportsLikelihood
    assert ModelSupportsLogLikelihood is LikelihoodSupportsLogLikelihood
    assert ModelSupportsTransitionDensity is LikelihoodSupportsTransitionDensity
    assert ModelSupportsTransitionSampling is LikelihoodSupportsTransitionSampling
