# MEM-EKF* for 3-D ellipsoids

`MEMEKFStar3DTracker` estimates the center, linear kinematics, orientation, and
three semi-axis lengths of one ellipsoidal object from 3-D point measurements.
It is an engineering generalization of the planar MEM-EKF* moment equations in
[Yang and Baum, *Tracking the Orientation and Axes Lengths of an Elliptical
Extended Object* (2019)](https://arxiv.org/abs/1805.03276).
The cited paper describes the 2-D method; the rotation-matrix state and 3-D
moment calculations described here are this implementation's extension.

## State and measurement conventions

The measurement model is

```text
z = H x + R diag(a) h + v
```

Here `x` is the kinematic state, `R` is a proper 3-by-3 rotation matrix, and
`a = [a1, a2, a3]` contains positive semi-axis lengths. Each column of `R`
identifies the corresponding body axis in world coordinates. The default
measurement matrix selects the first three state entries as the center. A
6-D constant-velocity state can therefore use `[x, y, z, vx, vy, vz]`.

The 6-by-6 shape covariance uses the ordering
`[dtheta_x, dtheta_y, dtheta_z, da1, da2, da3]`. Rotation errors are local
body-frame rotation vectors in radians:

```text
R_true = R_estimate Exp([dtheta]_x)
```

The filter linearizes in these local coordinates, applies each orientation
correction through the rotation exponential, and transports the covariance to
the corrected tangent frame. The rotation matrix remains on `SO(3)`.
Orientation uncertainty should be local: a single Gaussian tangent
approximation does not represent multiple possible orientations well. The
semi-axis correction is additive and uses `minimum_axis_length` to keep
lengths positive. Repeated clipping indicates that the initialization or
shape process covariance needs adjustment.

Axis lengths are not automatically sorted or relabelled. A sign reversal of
two orientation columns represents the same ellipsoid, and permuting columns
with their axis lengths also preserves the extent. Nearly equal axes make
some orientation components weakly observable; compare extent matrices when
evaluating such targets.

`update` accepts a single 3-D point, a `3 x N` array of column measurements, or
an `N x 3` array of row measurements. A `3 x 3` array is interpreted as columns.
An empty set leaves the state unchanged. Measurements are processed
sequentially; no association or clutter model is included.

## Noise and extent conventions

The zero-mean multiplicative noise `h` controls where measurements are
distributed within the ellipsoid. Its covariance is supplied through
`multiplicative_noise_cov`; the default is `I3 / 5`, matching points sampled
uniformly in the volume of the unit ball and transformed by `R diag(a)`.
Points uniformly sampled on the unit sphere instead have covariance `I3 / 3`.
The tracker uses Gaussian fourth-moment closure for the quadratic update, so
uniform-volume or uniform-surface data still use an approximate update.

Additive sensor covariance belongs in `meas_noise_cov`. It is separate from
the multiplicative covariance and is not absorbed into the estimated axes.
The estimator uses six quadratic residual components:
`[dx**2, dy**2, dz**2, dx*dy, dx*dz, dy*dz]`, including the shape-uncertainty
contribution to the linear measurement covariance.

`tracker.extent` and `tracker.get_point_estimate_extent()` return
`R @ diag(a**2) @ R.T`. This is the geometric extent matrix. The measurement
spread from the nominal ellipsoid is instead
`R @ diag(a) @ C_h @ diag(a) @ R.T`; with the default noise convention it is
`extent / 5`. Kinematic uncertainty, shape uncertainty, and additive sensor
noise also contribute to the innovation covariance.

## Minimal filter loop

```python
import numpy as np
from pyrecest.filters import MEMEKFStar3DTracker

tracker = MEMEKFStar3DTracker(
    kinematic_state=np.zeros(6),
    covariance=np.diag([1.0] * 3 + [0.1] * 3),
    orientation=np.eye(3),
    semi_axes=[4.0, 2.5, 1.4],
    shape_covariance=np.diag([0.05**2] * 3 + [0.3**2] * 3),
)

dt = 0.2
transition = np.eye(6)
transition[:3, 3:] = dt * np.eye(3)

# Replace this small illustrative scan with points from the sensor.
points = np.array([[1.0, -0.5, 0.2], [-1.0, 0.3, -0.1], [0.2, 1.0, 0.4], [0.0, -1.0, -0.3]])
tracker.predict_linear(
    transition,
    sys_noise=1e-4 * np.eye(6),
    shape_sys_noise=1e-6 * np.eye(6),
)
tracker.update(points, meas_noise_cov=0.08**2 * np.eye(3))

center = tracker.kinematic_state[:3]
orientation = tracker.orientation
semi_axes = tracker.semi_axes
extent = tracker.get_point_estimate_extent()
```

The full point estimate contains the kinematic state followed by the nine
row-major entries of the extent matrix. Shape uncertainty is available
separately as `shape_covariance`.

## Rotation prediction

`predict_linear` predicts linear kinematics and preserves the mean axes.
Its optional `rotation_increment` is a known body-frame rotation vector in
radians, applied on the right:

```text
R_next = R_current Exp([rotation_increment]_x)
```

Omitting the increment keeps the mean orientation unchanged. A known
increment is not an estimated angular-velocity state. The prediction
transports the rotation covariance and rotation/axis cross-covariances into
the next body tangent frame. `shape_sys_noise` is a 6-by-6 process covariance
in the predicted frame, with the same ordering as `shape_covariance`.

## Seeded rotating-object example

From the repository root after installing the checkout, run:

```bash
python examples/basic/mem_ekf_star_3d_tracking.py
```

The example tracks a triaxial ellipsoid with an arbitrary initial 3-D
orientation, constant linear velocity, and known body rotation increments.
It generates 60 scans of 50 uniform-volume points with additive Gaussian
sensor noise. The printed center, velocity, orientation, and relative extent
errors demonstrate the estimate for this seeded scenario. The orientation
metric allows equivalent axis sign flips, and the extent metric is invariant
to axis labelling.

To save a figure containing the final ellipsoid and error histories, use:

```bash
python examples/basic/mem_ekf_star_3d_tracking.py --output tracking-3d.png
```

The simulation and plotting example uses the NumPy backend. Its seeded result
is a reproducible demonstration, not a general convergence guarantee.
