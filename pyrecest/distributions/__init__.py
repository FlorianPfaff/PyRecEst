from pyrecest.distributions.abstract_bounded_domain_distribution import (
    AbstractBoundedDomainDistribution,
)
from pyrecest.distributions.abstract_bounded_nonperiodic_distribution import (
    AbstractBoundedNonPeriodicDistribution,
)
from pyrecest.distributions.abstract_custom_distribution import (
    AbstractCustomDistribution,
)
from pyrecest.distributions.abstract_custom_nonperiodic_distribution import (
    AbstractCustomNonPeriodicDistribution,
)
from pyrecest.distributions.abstract_dirac_distribution import AbstractDiracDistribution
from pyrecest.distributions.abstract_disk_distribution import AbstractDiskDistribution
from pyrecest.distributions.abstract_distribution_type import AbstractDistributionType
from pyrecest.distributions.abstract_ellipsoidal_ball_distribution import (
    AbstractEllipsoidalBallDistribution,
)
from pyrecest.distributions.abstract_manifold_specific_distribution import (
    AbstractManifoldSpecificDistribution,
)
from pyrecest.distributions.abstract_mixture import AbstractMixture
from pyrecest.distributions.abstract_nonperiodic_distribution import (
    AbstractNonperiodicDistribution,
)
from pyrecest.distributions.abstract_orthogonal_basis_distribution import (
    AbstractOrthogonalBasisDistribution,
)
from pyrecest.distributions.abstract_periodic_distribution import (
    AbstractPeriodicDistribution,
)
from pyrecest.distributions.abstract_se3_distribution import AbstractSE3Distribution
from pyrecest.distributions.abstract_uniform_distribution import (
    AbstractUniformDistribution,
)
from pyrecest.distributions.cart_prod.abstract_cart_prod_distribution import (
    AbstractCartProdDistribution,
)
from pyrecest.distributions.cart_prod.abstract_hypercylindrical_distribution import (
    AbstractHypercylindricalDistribution,
)
from pyrecest.distributions.cart_prod.abstract_lin_bounded_cart_prod_distribution import (
    AbstractLinBoundedCartProdDistribution,
)
from pyrecest.distributions.cart_prod.abstract_lin_hyperhemisphere_cart_prod_distribution import (
    AbstractLinHyperhemisphereCartProdDistribution,
)
from pyrecest.distributions.cart_prod.abstract_lin_hypersphere_cart_prod_distribution import (
    AbstractLinHypersphereCartProdDistribution,
)
from pyrecest.distributions.cart_prod.abstract_lin_hypersphere_subset_cart_prod_distribution import (
    AbstractLinHypersphereSubsetCartProdDistribution,
)
from pyrecest.distributions.cart_prod.abstract_lin_periodic_cart_prod_distribution import (
    AbstractLinPeriodicCartProdDistribution,
)
from pyrecest.distributions.cart_prod.cart_prod_stacked_distribution import (
    CartProdStackedDistribution,
)
from pyrecest.distributions.cart_prod.hypercylindrical_dirac_distribution import (
    HypercylindricalDiracDistribution,
)
from pyrecest.distributions.cart_prod.lin_bounded_cart_prod_dirac_distribution import (
    LinBoundedCartProdDiracDistribution,
)
from pyrecest.distributions.cart_prod.lin_hypersphere_cart_prod_dirac_distribution import (
    LinHypersphereCartProdDiracDistribution,
)
from pyrecest.distributions.cart_prod.lin_hypersphere_subset_dirac_distribution import (
    LinHypersphereSubsetCartProdDiracDistribution,
)
from pyrecest.distributions.cart_prod.lin_periodic_cart_prod_dirac_distribution import (
    LinPeriodicCartProdDiracDistribution,
)
from pyrecest.distributions.cart_prod.partially_wrapped_normal_distribution import (
    PartiallyWrappedNormalDistribution,
)
from pyrecest.distributions.circle.abstract_circular_distribution import (
    AbstractCircularDistribution,
)
from pyrecest.distributions.circle.circular_dirac_distribution import (
    CircularDiracDistribution,
)
from pyrecest.distributions.circle.circular_fourier_distribution import (
    CircularFourierDistribution,
)
from pyrecest.distributions.circle.circular_mixture import CircularMixture
from pyrecest.distributions.circle.circular_uniform_distribution import (
    CircularUniformDistribution,
)
from pyrecest.distributions.circle.custom_circular_distribution import (
    CustomCircularDistribution,
)
from pyrecest.distributions.circle.sine_skewed_distributions import (
    GeneralizedKSineSkewedVonMisesDistribution,
)
from pyrecest.distributions.circle.von_mises_distribution import VonMisesDistribution
from pyrecest.distributions.circle.wrapped_cauchy_distribution import (
    WrappedCauchyDistribution,
)
from pyrecest.distributions.circle.wrapped_laplace_distribution import (
    WrappedLaplaceDistribution,
)
from pyrecest.distributions.circle.wrapped_normal_distribution import (
    WrappedNormalDistribution,
)
from pyrecest.distributions.custom_hyperrectangular_distribution import (
    CustomHyperrectangularDistribution,
)
from pyrecest.distributions.disk_uniform_distribution import DiskUniformDistribution
from pyrecest.distributions.ellipsoidal_ball_uniform_distribution import (
    EllipsoidalBallUniformDistribution,
)
from pyrecest.distributions.hypersphere_subset.abstract_hemispherical_distribution import (
    AbstractHemisphericalDistribution,
)
from pyrecest.distributions.hypersphere_subset.abstract_hyperhemispherical_distribution import (
    AbstractHyperhemisphericalDistribution,
)
from pyrecest.distributions.hypersphere_subset.abstract_hypersphere_subset_dirac_distribution import (
    AbstractHypersphereSubsetDiracDistribution,
)
from pyrecest.distributions.hypersphere_subset.abstract_hypersphere_subset_distribution import (
    AbstractHypersphereSubsetDistribution,
)
from pyrecest.distributions.hypersphere_subset.abstract_hypersphere_subset_uniform_distribution import (
    AbstractHypersphereSubsetUniformDistribution,
)
from pyrecest.distributions.hypersphere_subset.abstract_hyperspherical_distribution import (
    AbstractHypersphericalDistribution,
)
from pyrecest.distributions.hypersphere_subset.abstract_sphere_subset_distribution import (
    AbstractSphereSubsetDistribution,
)
from pyrecest.distributions.hypersphere_subset.abstract_spherical_distribution import (
    AbstractSphericalDistribution,
)
from pyrecest.distributions.hypersphere_subset.abstract_spherical_harmonics_distribution import (
    AbstractSphericalHarmonicsDistribution,
)
from pyrecest.distributions.hypersphere_subset.bingham_distribution import (
    BinghamDistribution,
)
from pyrecest.distributions.hypersphere_subset.custom_hemispherical_distribution import (
    CustomHemisphericalDistribution,
)
from pyrecest.distributions.hypersphere_subset.custom_hyperhemispherical_distribution import (
    CustomHyperhemisphericalDistribution,
)
from pyrecest.distributions.hypersphere_subset.hyperspherical_dirac_distribution import (
    HypersphericalDiracDistribution,
)
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_dirac_distribution import (
    HyperhemisphericalDiracDistribution,
)
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_uniform_distribution import (
    HyperhemisphericalUniformDistribution,
)
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_watson_distribution import (
    HyperhemisphericalWatsonDistribution,
)
from pyrecest.distributions.hypersphere_subset.hyperspherical_mixture import (
    HypersphericalMixture,
)
from pyrecest.distributions.hypersphere_subset.hyperspherical_uniform_distribution import (
    HypersphericalUniformDistribution,
)
from pyrecest.distributions.hypersphere_subset.spherical_harmonics_distribution_complex import (
    SphericalHarmonicsDistributionComplex,
)
from pyrecest.distributions.hypersphere_subset.spherical_harmonics_distribution_real import (
    SphericalHarmonicsDistributionReal,
)
from pyrecest.distributions.hypersphere_subset.von_mises_fisher_distribution import (
    VonMisesFisherDistribution,
)
from pyrecest.distributions.hypersphere_subset.watson_distribution import (
    WatsonDistribution,
)
from pyrecest.distributions.hypertorus.abstract_hypertoroidal_distribution import (
    AbstractHypertoroidalDistribution,
)
from pyrecest.distributions.hypertorus.abstract_toroidal_distribution import (
    AbstractToroidalDistribution,
)
from pyrecest.distributions.hypertorus.custom_hypertoroidal_distribution import (
    CustomHypertoroidalDistribution,
)
from pyrecest.distributions.hypertorus.custom_toroidal_distribution import (
    CustomToroidalDistribution,
)
from pyrecest.distributions.hypertorus.hypertoroidal_dirac_distribution import (
    HypertoroidalDiracDistribution,
)
from pyrecest.distributions.hypertorus.hypertoroidal_mixture import HypertoroidalMixture
from pyrecest.distributions.hypertorus.hypertoroidal_uniform_distribution import (
    HypertoroidalUniformDistribution,
)
from pyrecest.distributions.hypertorus.hypertoroidal_wrapped_normal_distribution import (
    HypertoroidalWrappedNormalDistribution,
)
from pyrecest.distributions.hypertorus.toroidal_dirac_distribution import (
    ToroidalDiracDistribution,
)
from pyrecest.distributions.hypertorus.toroidal_mixture import ToroidalMixture
from pyrecest.distributions.hypertorus.toroidal_uniform_distribution import (
    ToroidalUniformDistribution,
)
from pyrecest.distributions.hypertorus.toroidal_von_mises_sine_distribution import (
    ToroidalVonMisesSineDistribution,
)
from pyrecest.distributions.hypertorus.toroidal_wrapped_normal_distribution import (
    ToroidalWrappedNormalDistribution,
)
from pyrecest.distributions.nonperiodic.abstract_hyperrectangular_distribution import (
    AbstractHyperrectangularDistribution,
)
from pyrecest.distributions.nonperiodic.abstract_linear_distribution import (
    AbstractLinearDistribution,
)
from pyrecest.distributions.nonperiodic.custom_linear_distribution import (
    CustomLinearDistribution,
)
from pyrecest.distributions.nonperiodic.gaussian_distribution import (
    GaussianDistribution,
)
from pyrecest.distributions.nonperiodic.gaussian_mixture import GaussianMixture
from pyrecest.distributions.nonperiodic.hyperrectangular_uniform_distribution import (
    HyperrectangularUniformDistribution,
)
from pyrecest.distributions.nonperiodic.linear_dirac_distribution import (
    LinearDiracDistribution,
)
from pyrecest.distributions.nonperiodic.linear_mixture import LinearMixture
from pyrecest.distributions.se3_cart_prod_stacked_distribution import (
    SE3CartProdStackedDistribution,
)
from pyrecest.distributions.se3_dirac_distribution import SE3DiracDistribution

# Aliases for brevity and compatibility with libDirectional
HypertoroidalWNDistribution = HypertoroidalWrappedNormalDistribution
WNDistribution = WrappedNormalDistribution
HypertoroidalWDDistribution = HypertoroidalDiracDistribution
ToroidalWDDistribution = ToroidalDiracDistribution
VMDistribution = VonMisesDistribution
WDDistribution = CircularDiracDistribution
VMFDistribution = VonMisesFisherDistribution

aliases = [
    "HypertoroidalWNDistribution",
    "WNDistribution",
    "HypertoroidalWDDistribution",
    "ToroidalWDDistribution",
    "VMDistribution",
    "WDDistribution",
    "VMFDistribution",
]

__all__ = aliases + [
    "HypersphericalDiracDistribution",
    "GeneralizedKSineSkewedVonMisesDistribution",
    "AbstractBoundedDomainDistribution",
    "AbstractBoundedNonPeriodicDistribution",
    "AbstractCustomDistribution",
    "AbstractCustomNonPeriodicDistribution",
    "AbstractDiracDistribution",
    "AbstractDiskDistribution",
    "AbstractDistributionType",
    "AbstractEllipsoidalBallDistribution",
    "AbstractManifoldSpecificDistribution",
    "AbstractMixture",
    "AbstractNonperiodicDistribution",
    "AbstractOrthogonalBasisDistribution",
    "AbstractPeriodicDistribution",
    "AbstractSE3Distribution",
    "AbstractUniformDistribution",
    "AbstractCartProdDistribution",
    "AbstractHypercylindricalDistribution",
    "AbstractLinBoundedCartProdDistribution",
    "AbstractLinHyperhemisphereCartProdDistribution",
    "AbstractLinHypersphereCartProdDistribution",
    "AbstractLinHypersphereSubsetCartProdDistribution",
    "AbstractLinPeriodicCartProdDistribution",
    "CartProdStackedDistribution",
    "HypercylindricalDiracDistribution",
    "LinBoundedCartProdDiracDistribution",
    "LinHypersphereCartProdDiracDistribution",
    "LinHypersphereSubsetCartProdDiracDistribution",
    "LinPeriodicCartProdDiracDistribution",
    "PartiallyWrappedNormalDistribution",
    "AbstractCircularDistribution",
    "CircularDiracDistribution",
    "CircularFourierDistribution",
    "CircularMixture",
    "CircularUniformDistribution",
    "CustomCircularDistribution",
    "VonMisesDistribution",
    "WrappedCauchyDistribution",
    "WrappedLaplaceDistribution",
    "WrappedNormalDistribution",
    "CustomHyperrectangularDistribution",
    "DiskUniformDistribution",
    "EllipsoidalBallUniformDistribution",
    "AbstractHemisphericalDistribution",
    "AbstractHyperhemisphericalDistribution",
    "AbstractHypersphereSubsetDiracDistribution",
    "AbstractHypersphereSubsetDistribution",
    "AbstractHypersphereSubsetUniformDistribution",
    "AbstractHypersphericalDistribution",
    "AbstractSphereSubsetDistribution",
    "AbstractSphericalDistribution",
    "AbstractSphericalHarmonicsDistribution",
    "BinghamDistribution",
    "CustomHemisphericalDistribution",
    "CustomHyperhemisphericalDistribution",
    "HyperhemisphericalDiracDistribution",
    "HyperhemisphericalUniformDistribution",
    "HyperhemisphericalWatsonDistribution",
    "HypersphericalMixture",
    "HypersphericalUniformDistribution",
    "SphericalHarmonicsDistributionComplex",
    "SphericalHarmonicsDistributionReal",
    "VonMisesFisherDistribution",
    "WatsonDistribution",
    "AbstractHypertoroidalDistribution",
    "AbstractToroidalDistribution",
    "CustomHypertoroidalDistribution",
    "CustomToroidalDistribution",
    "HypertoroidalDiracDistribution",
    "HypertoroidalMixture",
    "HypertoroidalUniformDistribution",
    "HypertoroidalWrappedNormalDistribution",
    "ToroidalDiracDistribution",
    "ToroidalMixture",
    "ToroidalUniformDistribution",
    "ToroidalVonMisesSineDistribution",
    "ToroidalWrappedNormalDistribution",
    "AbstractHyperrectangularDistribution",
    "AbstractLinearDistribution",
    "CustomLinearDistribution",
    "GaussianDistribution",
    "GaussianMixture",
    "HyperrectangularUniformDistribution",
    "LinearDiracDistribution",
    "LinearMixture",
    "SE3CartProdStackedDistribution",
    "SE3DiracDistribution",
]
