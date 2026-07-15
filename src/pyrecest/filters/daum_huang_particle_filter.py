"""Daum-Huang Gaussian particle-flow filters with weight-preserving updates."""

from . import _daum_huang_particle_filter_impl as _impl
from pyrecest.distributions.nonperiodic.linear_dirac_distribution import (
    LinearDiracDistribution,
)

for _name in dir(_impl):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_impl, _name)


class DaumHuangParticleFlowFilter(_impl.DaumHuangParticleFlowFilter):
    """Exact Daum-Huang filter that retains the transported particle weights."""

    def update_model(self, *args, **kwargs):
        prior_weights = self.filter_state.w
        info = super().update_model(*args, **kwargs)
        self._filter_state = LinearDiracDistribution(self.filter_state.d, prior_weights)
        return info


class LocalizedDaumHuangParticleFlowFilter(DaumHuangParticleFlowFilter):
    """Localized Daum-Huang filter with weight-preserving updates."""

    flow_type: FlowType = "ledh"


EDHParticleFlowFilter = DaumHuangParticleFlowFilter
LEDHParticleFlowFilter = LocalizedDaumHuangParticleFlowFilter

__all__ = _impl.__all__
