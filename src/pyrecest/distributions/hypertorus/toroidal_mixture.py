from .abstract_toroidal_distribution import AbstractToroidalDistribution
from .hypertoroidal_mixture import HypertoroidalMixture


class ToroidalMixture(HypertoroidalMixture, AbstractToroidalDistribution):
    def __init__(self, hds: list[AbstractToroidalDistribution], w):
        """
        Constructor

        :param hds: list of toroidal distributions
        :param w: list of weights
        """
        if len(hds) == 0:
            raise ValueError("Mixture must contain at least one distribution")
        if not all(isinstance(hd, AbstractToroidalDistribution) for hd in hds):
            raise ValueError("hds must be a list of toroidal distributions")

        HypertoroidalMixture.__init__(self, hds, w)
        AbstractToroidalDistribution.__init__(self)
