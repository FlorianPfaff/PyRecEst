# Shapes And Conventions

PyRecEst follows mathematical filtering notation, but most vectors are passed as
one-dimensional backend arrays in Python. The examples use `pyrecest.backend`
instead of importing NumPy directly so the same code can run on supported
backends where possible.

```python
from pyrecest.backend import array, diag

mean = array([0.0, 1.0])
covariance = diag(array([1.0, 1.0]))
```

## Backend Arrays

Use functions from `pyrecest.backend` for arrays, linear algebra, random values,
and elementary operations in reusable code. The default backend is NumPy, and
optional PyTorch or JAX backends can be selected with `PYRECEST_BACKEND` before
importing `pyrecest`.

Some advanced APIs are backend-specific. The examples and tests are the best
current reference when an API has backend limitations.

## Linear Gaussian States

For Euclidean Gaussian states, the common convention is:

| Quantity            | Shape                    | Meaning                        |
|---------------------|--------------------------|--------------------------------|
| `mean` or `mu`      | `(state_dim,)`           | State mean vector              |
| `covariance` or `C` | `(state_dim, state_dim)` | State covariance matrix        |
| `system_matrix`     | `(pred_dim, state_dim)`  | Linear transition matrix       |
| `sys_noise_cov`     | `(pred_dim, pred_dim)`   | Process-noise covariance       |
| `sys_input`         | `(pred_dim,)`            | Optional additive system input |

`GaussianDistribution(mu, C)` expects a one-dimensional mean and a
two-dimensional covariance matrix. A scalar one-dimensional Gaussian is also
accepted and is reshaped internally.

Linear prediction follows:

```text
x_k = F x_{k-1} + u + w
w ~ N(0, Q)
```

In code, this is:

```python
filter.predict_linear(system_matrix, sys_noise_cov, sys_input=None)
```

## Linear Measurements

Linear updates follow:

```text
z_k = H x_k + v
v ~ N(0, R)
```

For a single target, use:

| Quantity             | Shape                   | Meaning                      |
|----------------------|-------------------------|------------------------------|
| `measurement`        | `(meas_dim,)`           | One measurement vector       |
| `measurement_matrix` | `(meas_dim, state_dim)` | Linear measurement matrix    |
| `meas_noise`         | `(meas_dim, meas_dim)`  | Measurement-noise covariance |

For a scalar measurement, pass a one-element vector:

```python
measurement = array([position_measurement])
measurement_matrix = array([[1.0, 0.0]])
measurement_noise_cov = array([[0.25]])

filter.update_linear(measurement, measurement_matrix, measurement_noise_cov)
```

## Measurement Sets For Trackers

Multi-target tracking APIs use a column-oriented measurement-set convention:
each column is one measurement.

| Quantity             | Shape                                    | Meaning                       |
|----------------------|------------------------------------------|-------------------------------|
| `measurements`       | `(meas_dim, num_measurements)`           | Measurement set               |
| `measurement_matrix` | `(meas_dim, state_dim)`                  | Shared measurement model      |
| `cov_mats_meas`      | `(meas_dim, meas_dim)`                   | Shared measurement covariance |
| `cov_mats_meas`      | `(meas_dim, meas_dim, num_measurements)` | Per-measurement covariances   |

For example, three one-dimensional detections are represented as:

```python
measurements = array([[1.10, 9.15, 5.00]])
```

Extracted labeled target states are returned with one target per column:

```python
labels, estimates = tracker.get_labeled_point_estimate(number_of_targets=2)
```

Here `estimates` has shape `(state_dim, num_targets)`, and the column order
matches the order of `labels`.

## Distribution Inputs

Many distribution `pdf` methods accept either one point or a batch of points.
For Euclidean and hyperspherical distributions, batched points commonly use the
last axis for coordinates:

| Domain                      | Single point | Batch of points     |
|-----------------------------|--------------|---------------------|
| Euclidean `dim`             | `(dim,)`     | `(num_points, dim)` |
| Hypersphere embedded in R^d | `(d,)`       | `(num_points, d)`   |

Dirac distributions store support locations by row:

| Quantity | Shape               | Meaning                 |
|----------|---------------------|-------------------------|
| `d`      | `(num_diracs, dim)` | Dirac support locations |
| `w`      | `(num_diracs,)`     | Dirac weights           |

## Angles And Manifold Coordinates

Circular and toroidal APIs use radians. Functions that normalize angles usually
map them modulo `2 * pi`, so example code should avoid mixing degrees and
radians.

Hyperspherical distributions use embedded coordinates. For example,
`VonMisesFisherDistribution(mu, kappa)` expects `mu` to be a unit vector with
shape `(input_dim,)`, where `input_dim` is one larger than the intrinsic sphere
dimension. Samples and evaluation points use the same embedded coordinate
dimension.

For Cartesian-product and rigid-body state spaces, keep component blocks in the
order expected by the specific distribution or filter class. When in doubt, look
at the class tests for a runnable example of the expected coordinate ordering.

## Protocol-Level State-Space Metadata

The public manifold and state-space protocols use `dim` for the intrinsic state
dimension and `input_dim` for the trailing coordinate dimension accepted by
public methods. For Euclidean and toroidal coordinates these values often match.
For embedded manifolds, such as hyperspheres, `input_dim` can be larger than
`dim`.

Finite-measure domains can additionally expose `get_manifold_size()` and
`get_ln_manifold_size()`. These methods should describe the total measure of the
state space with respect to the base measure used by the density, for example
the circumference of a circle or the surface area of a hypersphere. Non-compact
state spaces should not fake a finite manifold size just to satisfy a protocol.


## Practical Checklist

- Use one-dimensional arrays for state and measurement vectors.
- Use square two-dimensional arrays for covariance matrices.
- Wrap scalar measurements as one-element arrays.
- Put multi-target measurement sets in columns.
- Keep batch point coordinates on the final axis.
- Use radians for angular quantities.
- Use `pyrecest.backend` in examples intended to be backend-portable.
