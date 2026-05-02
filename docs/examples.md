# Examples

The source repository includes executable examples in `examples/basic/`.

Run examples from the repository root after installing PyRecEst or after
installing a development checkout.

## Basic Examples

### Gaussian Multiplication

`examples/basic/gaussian_multiplication.py` multiplies several two-dimensional
Gaussian distributions and checks the result against the closed-form
information representation.

```bash
python examples/basic/gaussian_multiplication.py
```

### Kalman Filter

`examples/basic/kalman_filter.py` runs a one-dimensional constant-velocity
Kalman filter.

```bash
python examples/basic/kalman_filter.py
```

### Multi-Target Tracking

`examples/basic/multi_target_tracking.py` runs a small linear/Gaussian
multi-Bernoulli-tracker scenario with two labeled targets, missed detections,
and clutter measurements.

```bash
python examples/basic/multi_target_tracking.py
```

This example currently requires the NumPy backend.

### von Mises-Fisher Multiplication

`examples/basic/von_mises_fisher_multiplication.py` multiplies two
von Mises-Fisher distributions on the unit sphere and verifies the analytic
product relation.

```bash
python examples/basic/von_mises_fisher_multiplication.py
```

### Custom Distribution Protocol

`examples/basic/custom_distribution_protocol.py` shows a small user-defined
scalar distribution that satisfies the public dimension protocols without
inheriting from a PyRecEst distribution base class.

```bash
python examples/basic/custom_distribution_protocol.py
```

### Custom Filter Protocol

`examples/basic/custom_filter_protocol.py` shows a small user-defined recursive
filter that satisfies the public dimension protocol and follows common PyRecEst
filter naming conventions.

```bash
python examples/basic/custom_filter_protocol.py
```

## Backend Selection

Select a non-default backend by setting `PYRECEST_BACKEND` before running the
script. For example, on a bash-compatible shell:

```bash
PYRECEST_BACKEND=pytorch python examples/basic/kalman_filter.py
```

Install the matching optional dependency extra before using a non-default
backend.
