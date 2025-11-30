from .abstract_lin_periodic_distribution import AbstractLinPeriodicDistribution


class AbstractLinHemisphericalDistribution(AbstractLinPeriodicDistribution):
    def __init__(self):
        self.periodic_manifold_type = 'hyperhemisphere'
