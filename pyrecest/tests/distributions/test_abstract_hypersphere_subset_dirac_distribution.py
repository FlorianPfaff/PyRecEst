import unittest

import numpy.testing as npt
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import allclose, array, linalg, random, spatial, stack
from pyrecest.distributions import BinghamDistribution
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_bingham_distribution import (
    HyperhemisphericalBinghamDistribution,
)
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_dirac_distribution import (
    HyperhemisphericalDiracDistribution,
)
from pyrecest.distributions.hypersphere_subset.hyperspherical_dirac_distribution import (
    HypersphericalDiracDistribution,
)


class AbstractHypersphereSubsetDiracDistributionTest(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    # pylint: disable=too-many-locals
    def test_mean_axis(self):
        random.seed(0)

        # Original quaternion components (w, x, y, z)
        q_components = array([1.0, 2.0, 3.0, 4.0], dtype=float)
        q_components /= linalg.norm(q_components)  # normalize
        w, x, y, z = q_components

        # Rotation uses scalar-last (x, y, z, w)
        q_xyzw = array([x, y, z, w], dtype=float)
        q_rot = spatial.Rotation.from_quat(q_xyzw)

        # Basis quaternions e0..e3 in (x, y, z, w) form:
        # e0 = (1, 0, 0, 0), e1 = (0, 1, 0, 0), e2 = (0, 0, 1, 0), e3 = (0, 0, 0, 1)
        basis_xyzw = array(
            [
                [0.0, 0.0, 0.0, 1.0],  # (1, 0, 0, 0)
                [1.0, 0.0, 0.0, 0.0],  # (0, 1, 0, 0)
                [0.0, 1.0, 0.0, 0.0],  # (0, 0, 1, 0)
                [0.0, 0.0, 1.0, 0.0],  # (0, 0, 0, 1)
            ]
        )

        # Emulate q * e_i via rotation composition and collect the resulting quaternions
        rows = []
        for b in basis_xyzw:
            e_rot = spatial.Rotation.from_quat(b)
            qe_rot = q_rot * e_rot  # composition of rotations
            # Get quaternion back in scalar-first (w, x, y, z) form
            qe_wxyz = qe_rot.as_quat(scalar_first=True)
            rows.append(qe_wxyz)

        # Orientation matrix M for the 4D Bingham distribution
        M = stack(rows, axis=0)  # shape (4, 4)

        Z = array([-10.0, -2.0, -1.0, 0.0])

        bd = BinghamDistribution(Z, M)
        bdHemi = HyperhemisphericalBinghamDistribution(Z, M)
        bd.F = 2.98312
        bdHemi.distFullSphere.F = bd.F

        wdFull = HypersphericalDiracDistribution.from_distribution(bd, 2001)
        wdHemi = HyperhemisphericalDiracDistribution.from_distribution(bdHemi, 2001)

        self.assertTrue(
            allclose(wdFull.mean_axis(), bd.mean_axis(), atol=0.2)
            or allclose(wdFull.mean_axis(), -bd.mean_axis(), atol=0.2)
        )
        npt.assert_allclose(wdHemi.mean_axis(), bdHemi.mean_axis(), atol=0.2)


if __name__ == "__main__":
    unittest.main()
