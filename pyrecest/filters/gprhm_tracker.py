# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import (
    abs,
    arctan2,
    array,
    cos,
    dot,
    exp,
    eye,
    hstack,
    linalg,
    linspace,
    min,
    ndim,
    pi,
    reshape,
    sin,
    stack,
    zeros,
    zeros_like,
)

from .abstract_extended_object_tracker import AbstractExtendedObjectTracker


def pol2cart(phi, r=1.0):
    if ndim(phi) > 1:
        r = reshape(r, (1, -1))
    return r * stack((cos(phi), sin(phi)))


def angle_between_two_vectors(x, y):
    dot_prod = dot(x, y)
    cross_prod = x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]
    return -arctan2(cross_prod, dot_prod) % (2 * pi)


# pylint: disable=too-many-instance-attributes
class GPRHMTracker(AbstractExtendedObjectTracker):
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def __init__(
        self,
        n_base_points,
        dimension: int = 2,
        velocities=False,
        kernel_params=(2.0, pi / 4),
        log_prior_estimates=True,
        log_posterior_estimates=True,
        log_prior_extents=True,
        log_posterior_extents=True,
    ):
        super().__init__(
            log_prior_estimates=log_prior_estimates,
            log_posterior_estimates=log_posterior_estimates,
            log_prior_extents=log_prior_extents,
            log_posterior_extents=log_posterior_extents,
        )

        def sin_kernel(
            phi_1, phi_2, sigma_squared=kernel_params[0], kernel_width=kernel_params[1]
        ):
            d = min(array((abs(phi_1 - phi_2), 2 * pi - abs(phi_1 - phi_2))))
            exponent = 2 * sin((0.5) * d) ** 2 / kernel_width**2
            return sigma_squared * exp(-exponent)

        self.kernel = sin_kernel
        self.phi_pts = linspace(0.0, 2 * pi, n_base_points, endpoint=False)
        if dimension == 2 and not velocities:
            self.m = zeros(2)
            self.H = eye(2)
            self.C_m = 0.1 * eye(2)
        else:
            raise NotImplementedError(
                "Only 2-D scenarios without velocity estimation supported"
            )

        self.p = zeros_like(self.phi_pts)
        self.C_p = 0.1 * eye(self.phi_pts.shape[0])

        def K_fun(phi):
            return array([[self.kernel(phi, phi_n) for phi_n in self.phi_pts]])

        K_p = array(
            [
                [
                    self.kernel(self.phi_pts[i], self.phi_pts[j])
                    for i in range(len(self.phi_pts))
                ]
                for j in range(len(self.phi_pts))
            ]
        )
        self.A_fun = lambda phi: linalg.solve(K_p, K_fun(phi).T).T
        self.C_e_fun = (
            lambda phi: self.kernel(phi, phi)
            - K_fun(phi) @ linalg.solve(K_p, K_fun(phi).T).T
        )

    def get_point_estimate(self):
        # Return the state concatenated with the flattened extent
        return hstack([self.m, self.p.flatten()])

    def get_point_estimate_kinematics(self):
        # Return only the kinematic state
        return self.m

    def get_point_estimate_extent(self, flatten_matrix=False):
        # Return the extent matrix, optionally flattened
        return self.p.flatten() if flatten_matrix else self.p

    def get_extents_on_grid(self, n: int = 100):
        angles = linspace(0.0, 2 * pi, n, endpoint=False)
        extents = array([linalg.norm(self.A_fun(phi) @ self.p) for phi in angles])
        return extents

    def get_contour_points(self, n: int = 100):
        angles = linspace(0.0, 2 * pi, n, endpoint=False)
        star_point = self.H @ self.m
        extents = self.get_extents_on_grid()
        coords = pol2cart(angles, extents) + reshape(star_point, (-1, 1))
        return coords.T

    def update(
        self,
        z,
        R,
        # s_hat=1 and sigma_squared_s=0 means that measurements only come from the contour
        s_hat=1,
        sigma_squared_s=0,
    ):
        phi = angle_between_two_vectors(z - self.H @ self.m, pol2cart(array(0)))
        B_phi = s_hat * pol2cart(phi)[:, None] @ self.A_fun(phi)

        if sigma_squared_s == 0:  # Use simpler formula if sigma_squared_s is zero
            C_w = R
        else:
            C_w = (
                sigma_squared_s
                * (
                    (pol2cart(phi) @ self.A_fun(phi))
                    @ self.C_p
                    @ (pol2cart(phi) @ self.A_fun(phi)).T
                    + pol2cart(phi) @ self.C_e_fun(phi) @ pol2cart(phi).T
                )
                + R
            )

        # Compute covariance matrix for the measurement z
        C_z = B_phi @ self.C_p @ B_phi.T + self.H @ self.C_m @ self.H.T + C_w

        # Compute cross-covariance matrices
        C_mz = self.C_m @ self.H.T
        C_pz = self.C_p @ B_phi.T

        # Mean of the expected measurement
        z_bar = B_phi @ self.p + self.H @ self.m

        # Compute the vector difference between z and z_bar
        delta_z = z - z_bar

        # Solve for the Kalman gain for the state update
        K_m = linalg.solve(C_z, C_mz.T).T

        # Update the state vector and covariance matrix
        self.m = self.m + K_m @ delta_z
        self.C_m = self.C_m - K_m @ C_mz.T

        # Solve for the Kalman gain for the extent parameters update
        K_p = linalg.solve(C_z, C_pz.T).T

        # Update the extent parameters and covariance matrix
        self.p = self.p + K_p @ delta_z
        self.C_p = self.C_p - K_p @ C_pz.T

        if self.log_posterior_estimates:
            self.store_posterior_estimates()
        if self.log_posterior_extents:
            self.store_posterior_extents()
