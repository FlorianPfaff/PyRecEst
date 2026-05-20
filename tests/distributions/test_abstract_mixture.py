import unittest

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import allclose, array, eye, linalg, ones
from pyrecest.distributions import VonMisesFisherDistribution
from pyrecest.distributions.hypersphere_subset.custom_hyperhemispherical_distribution import (
    CustomHyperhemisphericalDistribution,
)
from pyrecest.distributions.hypersphere_subset.hyperspherical_mixture import (
    HypersphericalMixture,
)
from pyrecest.distributions.hypertorus.hypertoroidal_mixture import HypertoroidalMixture
from pyrecest.distributions.hypertorus.toroidal_wrapped_normal_distribution import (
    ToroidalWrappedNormalDistribution,
)


class AbstractMixtureTest(unittest.TestCase):
    def _test_sample(self, mix, n):
        for sampling_method in [mix.sample_metropolis_hastings, mix.sample]:
            s = sampling_method(n)
            self.assertEqual(s.shape, (n, mix.input_dim))
        return s

    def test_negative_weights_rejected(self):
        vmf = ToroidalWrappedNormalDistribution(array([1.0, 0.0]), eye(2))
        shifted_vmf = vmf.shift(array([1.0, 1.0]))

        with self.assertRaisesRegex(ValueError, "nonnegative"):
            HypertoroidalMixture(
                [vmf, shifted_vmf],
                array([1.2, -0.2]),
            )

    def test_empty_mixture_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one distribution"):
            HypertoroidalMixture([], array([]))

    def test_nonfinite_weights_rejected(self):
        vmf = ToroidalWrappedNormalDistribution(array([1.0, 0.0]), eye(2))
        shifted_vmf = vmf.shift(array([1.0, 1.0]))

        with self.assertRaisesRegex(ValueError, "finite"):
            HypertoroidalMixture(
                [vmf, shifted_vmf],
                array([float("nan"), 1.0]),
            )

    def test_sample_metropolis_hastings_basics_only_t2(self):
        vmf = ToroidalWrappedNormalDistribution(array([1.0, 0.0]), eye(2))
        mix = HypertoroidalMixture(
            [vmf, vmf.shift(array([1.0, 1.0]))], array([0.5, 0.5])
        )
        self._test_sample(mix, 10)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch",),
        reason="Not supported on this backend",
    )
    def test_sample_metropolis_hastings_basics_only_s2(self):
        vmf1 = VonMisesFisherDistribution(array([1.0, 0.0, 0.0]), 2.0)
        vmf2 = VonMisesFisherDistribution(array([0.0, 1.0, 0.0]), 2.0)
        mix = HypersphericalMixture([vmf1, vmf2], array([0.5, 0.5]))
        s = self._test_sample(mix, 10)
        self.assertTrue(allclose(linalg.norm(s, axis=1), ones(10), rtol=5e-7))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch",),
        reason="Not supported on this backend",
    )
    def test_sample_metropolis_hastings_basics_only_h2(self):
        vmf = VonMisesFisherDistribution(array([1.0, 0.0, 0.0]), 2.0)
        mix = CustomHyperhemisphericalDistribution(
            lambda x: vmf.pdf(x) + vmf.pdf(-x), 2
        )
        s = self._test_sample(mix, 10)
        self.assertTrue(allclose(linalg.norm(s, axis=1), ones(10), rtol=5e-7))


if __name__ == "__main__":
    unittest.main()
