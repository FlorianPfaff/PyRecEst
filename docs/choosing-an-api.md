# Choosing an API

Use this page as a first-pass map from an estimation task to the most relevant
PyRecEst components. The exact class names and argument conventions are covered
in the API overview, examples, and reference pages.

| Task                                | Start with                               | Notes                                                                                            |
|-------------------------------------|------------------------------------------|--------------------------------------------------------------------------------------------------|
| Linear Euclidean Gaussian filtering | `KalmanFilter`                           | Use `predict_linear()` and `update_linear()` for matrix-based transition and measurement models. |
| Nonlinear Euclidean filtering       | Unscented or particle-filter examples    | Prefer model-object examples when the transition and measurement models should be reusable.      |
| Circular or angular states          | Circular distributions and filters       | Use circular/hypertoroidal distributions when wrap-around behavior matters.                      |
| Spherical or directional states     | Hyperspherical distributions and filters | Use directional distributions for unit-vector or sphere-valued states.                           |
| Multi-target tracking               | Multi-target tracker examples            | Check the measurement-set shape convention before wiring detections into a tracker.              |
| Extended object tracking            | EOT-oriented filters and trackers        | Start with NumPy unless the target API is documented and tested for another backend.             |
| Backend-portable code               | `pyrecest.backend` imports               | Set `PYRECEST_BACKEND` before importing PyRecEst and run focused tests on each backend you need. |
| Custom estimators or distributions  | Public protocol docs                     | Use the protocol examples when you do not need to inherit from a PyRecEst base class.            |

## Recommended First Examples

1. Run `examples/basic/kalman_filter.py` to verify installation and learn the
   linear Gaussian conventions.
2. Run `examples/basic/kalman_filter_with_models.py` or
   `examples/basic/ukf_with_models.py` when you want reusable model objects.
3. Run `examples/basic/multi_target_tracking.py` when your measurements are sets
   of detections rather than one detection per time step.
4. Compare the same small workflow under `PYRECEST_BACKEND=numpy`, `pytorch`, and
   `jax` before relying on backend portability.
