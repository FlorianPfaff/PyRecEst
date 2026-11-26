# Track a 3D ellipsoid with MEM-QKF or MEM-RBPF

`MEMQKF3DTracker` and `MEMRBPF3DTracker` estimate the center, orientation and
three semi-axis lengths of one ellipsoid from associated 3D point returns.
The QKF represents orientation error by a local Gaussian in three dimensions.
The RBPF represents orientation by a weighted set of rotations and estimates
three semi-axes separately for each rotation. Both are experimental spatial
extensions of the planar algorithms. See the authors' implementations of
[MEM-QKF](https://github.com/Fusion-Goettingen/SDF_2025_Steuernagel_QuadraticEOT)
and [MEM-RBPF](https://github.com/Fusion-Goettingen/TSP_2025_Steuernagel_RBPF).

## Measurement model

Each point follows `z = H x + R diag(a) h + v`. Here `R` rotates body
coordinates into the world frame, `a` contains semi-axis lengths, and `v`
has the additive sensor covariance supplied to `update`. The default source
covariance is `Cov(h) = I / 5`, the second moment of uniform returns throughout
a solid unit ball. Both filters evaluate fourth moments under a Gaussian source
model. Sensors that return surface points or favor visible faces can have
different source statistics.

## Filter updates

The QKF processes each point about its predicted center. Its kinematic update
uses the nominal ellipsoid. For orientation it forms the six distinct products
of the world-frame residual components and integrates their conditional moments
over 27 Gauss–Hermite nodes in the local rotation coordinates. For semi-axis
lengths it squares the three body-frame components and evaluates their moments
analytically. Both shape updates use the orientation before that point's
correction. The orientation and axis covariance blocks are maintained separately.
Cross-block entries in the initial and process covariance are discarded. Each
point's shape update includes projected center uncertainty. A biased initial
center can affect the inferred shape until the kinematic estimate settles.
A locally plausible starting orientation is important for this approximation.

The RBPF updates a common kinematic estimate from the scan centroid. A local
random generator samples body rotations, and each particle carries a Gaussian
estimate for the semi-axes. Initial and process correlations between axes and
rotations shift the conditional axis means. The filter splits a scan into its
normalized centroid innovation and orthogonal differences between points.
Only the centroid depends on uncertainty in the center. Moment-matched Gaussian
likelihoods weight the rotations, and squared body-frame observations update
the axes conditionally. The order of these approximate updates can affect the
result. The reported extent
averages physical ellipsoids across particles, so equivalent axis sign choices
produce the same estimate. Its principal directions and descending semi-axis
lengths are available as `orientation` and `semi_axes`. Directions within a
subspace of repeated axis lengths are ambiguous for an isotropic source.

## Use the trackers

The constructor accepts a body-to-world rotation and a 6-by-6 shape covariance
ordered as three body rotation errors in radians followed by three additive
semi-axis errors. By default the first three kinematic coordinates give the
center. A custom `measurement_matrix` may select a different position model.

```python
import numpy as np
from pyrecest.filters import MEMQKF3DTracker, MEMRBPF3DTracker

prior = dict(
    kinematic_state=np.zeros(6),
    covariance=np.diag([0.5] * 3 + [0.1] * 3),
    orientation=np.eye(3),
    semi_axes=[3.0, 2.0, 1.0],
    shape_covariance=np.diag([0.15**2] * 3 + [0.3**2] * 3),
)
qkf = MEMQKF3DTracker(**prior)
rbpf = MEMRBPF3DTracker(**prior, n_particles=128, rng=7)

transition = np.eye(6)
transition[:3, 3:] = 0.2 * np.eye(3)
scan = np.array([[1.0, -0.4, 0.7, 1.1],
                 [0.3,  0.9, 0.2, -0.5],
                 [0.1, -0.2, 0.6, 0.0]])
for tracker in (qkf, rbpf):
    tracker.predict_linear(transition, rotation_increment=[0.02, -0.01, 0.03])
    tracker.update(scan, meas_noise_cov=0.08**2 * np.eye(3))
    extent = tracker.get_point_estimate_extent()
```

A scan may contain a single vector or points in columns or rows. A 3-by-3
array means three points in columns. The optional prediction increment is a
known right-body rotation in radians. The reported 3-by-3 extent is a geometric
matrix with eigenvalues equal to squared semi-axis lengths.

The QKF supports NumPy and eager CPU use with JAX and PyTorch. The RBPF uses
NumPy and accepts an integer seed or a NumPy random generator. Its
`resampling_threshold` is an effective particle count. The default triggers
systematic resampling at half the particle count, and zero disables automatic
resampling.

To compare both trackers on the same moving ellipsoid and observations, run
the seeded example twice from the repository root.

```bash
python examples/basic/mem_ekf_star_3d_tracking.py --tracker qkf --seed 7
python examples/basic/mem_ekf_star_3d_tracking.py --tracker rbpf --seed 7
```
