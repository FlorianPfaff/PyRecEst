# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, eye
from pyrecest.distributions.hypertorus.toroidal_wrapped_normal_distribution import (
    ToroidalWrappedNormalDistribution,
)

from .abstract_filter import AbstractFilter
from .manifold_mixins import ToroidalFilterMixin


class ToroidalWrappedNormalFilter(AbstractFilter, ToroidalFilterMixin):
    """Filter based on the bivariate wrapped normal distribution.

    References
    ----------
    Kurz, G., Gilitschenski, I., Dolgov, M., & Hanebeck, U. D. (2014).
    Bivariate Angular Estimation Under Consideration of Dependencies Using
    Directional Statistics. Proceedings of the 53rd IEEE Conference on
    Decision and Control.

    Kurz, G., Pfaff, F., & Hanebeck, U. D. (2017). Nonlinear Toroidal
    Filtering Based on Bivariate Wrapped Normal Distributions. Proceedings of
    the 20th International Conference on Information Fusion.
    """

    def __init__(self):
        ToroidalFilterMixin.__init__(self)
        AbstractFilter.__init__(
            self, ToroidalWrappedNormalDistribution(array([0, 0]), eye(2))
        )

    def predict_identity(self, twn_sys):
        if not isinstance(twn_sys, ToroidalWrappedNormalDistribution):
            raise ValueError("twn_sys must be a ToroidalWrappedNormalDistribution.")
        self.filter_state = self.filter_state.convolve(twn_sys)
