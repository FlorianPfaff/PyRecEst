import unittest
import numpy as np
# Import other necessary packages, classes, or functions here.

class AbstractSE3LinVelDistributionTest(unittest.TestCase):

    def test_plot_trajectory(self):
        offsets = np.arange(10)
        quats = np.array([1, 1, 1, 1])[:, None] + np.array([offsets, np.zeros(3, 10)])
        quats = quats / np.linalg.norm(quats, axis=0)

        # TODO
        # AbstractSE3LinVelDistribution.plot_trajectory(quats,
        # AbstractSE3LinVelDistribution.plot_trajectory(quats, ..., True, 0.05)

# If you want to run the tests.
if __name__ == '__main__':
    unittest.main()
