# pylint: disable=redefined-builtin,no-name-in-module,no-member
# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, pi, sin, sum
from scipy.special import comb, iv

from .abstract_toroidal_bivar_vm_distribution import AbstractToroidalBivarVMDistribution


class ToroidalVonMisesSineDistribution(AbstractToroidalBivarVMDistribution):
    def __init__(self, mu, kappa, lambda_):
        AbstractToroidalBivarVMDistribution.__init__(self, mu, kappa)
        assert lambda_.shape == ()
        self.lambda_ = lambda_
        self.C = 1.0 / self.norm_const

    @property
    def norm_const(self):
        def s(m):
            return (
                comb(2 * m, m)
                * (self.lambda_**2 / 4 / self.kappa[0] / self.kappa[1]) ** m
                * iv(m, self.kappa[0])
                * iv(m, self.kappa[1])
            )

        Cinv = 4.0 * pi**2 * sum(array([s(m) for m in range(11)]))
        return Cinv

    def _coupling_term(self, xs):
        return (
            self.lambda_ * sin(xs[..., 0] - self.mu[0]) * sin(xs[..., 1] - self.mu[1])
        )
