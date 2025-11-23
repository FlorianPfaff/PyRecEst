import unittest
import numpy as np
import quaternion
from pyrecest.distributions import BinghamDistribution, VonMisesFisherDistribution
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_bingham_distribution import HyperhemisphericalBinghamDistribution
from pyrecest.distributions.hypersphere_subset.hyperspherical_grid_distribution import HypersphericalGridDistribution
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_grid_distribution import HyperhemisphericalGridDistribution
import sys

class AbstractHypersphereSubsetGridDistributionTest(unittest.TestCase):

    def test_mean_axis_S2(self):
        np.random.seed(0)
        v = np.array([1,1,0])/np.sqrt(2)
        vmf = VonMisesFisherDistribution(v, 1)
        vmf.mean_axis()
        
    @unittest.skipIf(sys.platform.startswith("win"), "requires Unix-based system")
    def test_mean_axis_S2(self):
        np.random.seed(0)
        q = np.quaternion(1, 2, 3, 4)
        q = q.normalized()
        qs = [q * np.quaternion(1, 0, 0, 0),
                             q * np.quaternion(0, 1, 0, 0),
                             q * np.quaternion(0, 0, 1, 0),
                             q * np.quaternion(0, 0, 0, 1)]
        Z = np.array([-10, -2, -1, 0])
        M = np.array([np.array([q.w, q.x, q.y, q.z]) for q in qs])

        bd = BinghamDistribution(Z, M)
        bdHemi = HyperhemisphericalBinghamDistribution(Z, M)
        wdFull = HypersphericalGridDistribution.from_distribution(bd, 100001)
        wdHemi = HyperhemisphericalGridDistribution.from_distribution(bdHemi, 100001)

        self.assertTrue(np.allclose(wdFull.mean_axis(), bd.mean_axis(), atol=0.01))
        self.assertTrue(np.allclose(wdHemi.mean_axis(), bd.mean_axis(), atol=0.01))

if __name__ == '__main__':
    unittest.main()
