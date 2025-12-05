import numpy as np
from abc import abstractmethod
from .abstract_se3_distribution import AbstractSE3Distribution
from typing import Union

class AbstractSE3LinVelDistribution(AbstractSE3Distribution):

    @abstractmethod
    def plot_mode(self):
        # Visualize the mode in SE(3) space
        mode = self.mode()
        h1 = AbstractSE3Distribution.plot_point(mode[:7])
        h2 = farrow(mode[4], mode[5], mode[6], mode[4] + mode[7], mode[5] + mode[8], mode[6] + mode[9], 'k')
        h = h1 + h2
        return h

    @abstractmethod
    def plot_state(self, orientation_samples=10, show_marginalized=True):
        # The shown uncertainties are shown marginalized
        samples = self.sample(orientation_samples)
        mode = self.mode()
        gauss_pos_vel = self.marginalize_periodic()

        gauss_pos = gauss_pos_vel.marginalize_out(range(3, 6))
        gauss_vel = gauss_pos_vel.marginalize_out(range(0, 3))
        
        if show_marginalized:
            hold_status = ishold()
            h = gauss_pos.plot_state()
            hold(True)
            h.extend(gauss_vel.plot_state())
            if hold_status:
                hold(True)
        else:
            h = []

        for i in range(samples.shape[1]):
            if show_marginalized:
                linear_part = mode[4:]
            else:
                linear_part = samples[4:, i]

            h.extend(AbstractSE3LinVelDistribution.plot_point(np.concatenate([samples[:4, i], linear_part])))

        return h

    @staticmethod
    def plot_point(se3with_vel_point):
        # Visualize just a point in the SE(3) domain with velocities
        # (no uncertainties are considered)
        h = AbstractSE3Distribution.plot_point(se3with_vel_point[:7])
        h.extend(farrow(se3with_vel_point[4], se3with_vel_point[5], se3with_vel_point[6],
                        se3with_vel_point[4] + se3with_vel_point[7], se3with_vel_point[5] + se3with_vel_point[8],
                        se3with_vel_point[6] + se3with_vel_point[9], 'k'))
        return h

    @staticmethod
    def plot_trajectory(periodic_states, lin_states, animate=False, delay=0.05):
        assert periodic_states.shape[1] == lin_states.shape[1]
        h = []
        for i in range(periodic_states.shape[1]):
            h.extend(AbstractSE3LinVelDistribution.plot_point(np.concatenate([periodic_states[:, i], lin_states[:, i]])))
            if animate:
                pause(delay)
        return h


"""

import numpy as np
from .abstract_lin_hemispherical_distribution import AbstractLinHemisphericalDistribution
from .abstract_se3_distribution import AbstractSE3Distribution

class AbstractSE3LinVelDistribution(AbstractLinHemisphericalDistribution):

    def plot_mode(self):
        mode = self.mode()
        h1 = AbstractSE3Distribution.plot_point(mode[:7])
        h2 = farrow(mode[4], mode[5], mode[6], mode[4] + mode[7], mode[5] + mode[8], mode[6] + mode[9], 'k')
        return h1 + h2

    def plot_state(self, orientation_samples=10, show_marginalized=True):
        samples = self.sample(orientation_samples)
        mode = self.mode()
        gauss_pos_vel = self.marginalize_periodic()

        gauss_pos = gauss_pos_vel.marginalize_out(np.array([3, 4, 5]))
        gauss_vel = gauss_pos_vel.marginalize_out(np.array([0, 1, 2]))

        if show_marginalized:
            h = gauss_pos.plot_state()
            h.extend(gauss_vel.plot_state())
        else:
            h = []

        for i in range(samples.shape[1]):
            if show_marginalized:
                linear_part = mode[4:]
            else:
                linear_part = samples[4:, i]

            h.extend(AbstractSE3LinVelDistribution.plot_point(np.hstack([samples[:4, i], linear_part])))

        return h

    @staticmethod
    def plot_point(se3_with_vel_point):
        h = AbstractSE3Distribution.plot_point(se3_with_vel_point[:7])
        h.extend(farrow(se3_with_vel_point[4], se3_with_vel_point[5], se3_with_vel_point[6],
                        se3_with_vel_point[4] + se3_with_vel_point[7], se3_with_vel_point[5] + se3_with_vel_point[8],
                        se3_with_vel_point[6] + se3_with_vel_point[9], 'k'))
        return h

    @staticmethod
    def plot_trajectory(periodic_states, lin_states, animate=False, delay=0.05):
        assert periodic_states.shape[1] == lin_states.shape[1]
        h = []
        for i in range(periodic_states.shape[1]):
            h.extend(AbstractSE3LinVelDistribution.plot_point(np.hstack([periodic_states[:, i], lin_states[:, i]])))
            if animate:
                time.sleep(delay)
        return h

"""


"""
import numpy as np
from abc import abstractmethod
from .abstract_lin_hemispherical_distribution import AbstractLinHemisphericalDistribution
from .abstract_se3_distribution import AbstractSE3Distribution

class AbstractSE3LinVelDistribution(AbstractLinHemisphericalDistribution):

    @abstractmethod
    def plot_mode(self):
        # Visualize the mode in SE(3) space
        mode = self.mode()
        h1 = AbstractSE3Distribution.plot_point(mode[:7])
        h2 = farrow(mode[4], mode[5], mode[6], mode[4] + mode[7], mode[5] + mode[8], mode[6] + mode[9], 'k')
        h = h1 + h2
        return h

    @abstractmethod
    def plot_state(self, orientation_samples=10, show_marginalized=True):
        # The shown uncertainties are shown marginalized
        samples = self.sample(orientation_samples)
        mode = self.mode()
        gauss_pos_vel = self.marginalize_periodic()

        gauss_pos = gauss_pos_vel.marginalize_out(range(3, 6))
        gauss_vel = gauss_pos_vel.marginalize_out(range(0, 3))
        
        if show_marginalized:
            hold_status = ishold()
            h = gauss_pos.plot_state()
            hold(True)
            h.extend(gauss_vel.plot_state())
            if hold_status:
                hold(True)
        else:
            h = []

        for i in range(samples.shape[1]):
            if show_marginalized:
                linear_part = mode[4:]
            else:
                linear_part = samples[4:, i]

            h.extend(AbstractSE3LinVelDistribution.plot_point(np.concatenate([samples[:4, i], linear_part])))

        return h

    @staticmethod
    def plot_point(se3with_vel_point):
        # Visualize just a point in the SE(3) domain with velocities
        # (no uncertainties are considered)
        h = AbstractSE3Distribution.plot_point(se3with_vel_point[:7])
        h.extend(farrow(se3with_vel_point[4], se3with_vel_point[5], se3with_vel_point[6],
                        se3with_vel_point[4] + se3with_vel_point[7], se3with_vel_point[5] + se3with_vel_point[8],
                        se3with_vel_point[6] + se3with_vel_point[9], 'k'))
        return h

    @staticmethod
    def plot_trajectory(periodic_states, lin_states, animate=False, delay=0.05):
        assert periodic_states.shape[1] == lin_states.shape[1]
        h = []
        for i in range(periodic_states.shape[1]):
            h.extend(AbstractSE3LinVelDistribution.plot_point(np.concatenate([periodic_states[:, i], lin_states[:, i]])))
            if animate:
                pause(delay)
        return h

"""