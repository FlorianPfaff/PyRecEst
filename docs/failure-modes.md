# Common Failure Modes

This page collects common symptoms and debugging directions for recursive
Bayesian estimation workflows.

## Covariance Is Not Positive Semidefinite

Check model dimensions, process-noise covariance, and update equations. For
linear Gaussian workflows, inspect innovation covariance and covariance trace.
Symmetrizing a covariance can hide numerical issues, so prefer finding the first
step where definiteness is lost.

## Particle Filter Degeneracy

Inspect effective sample size, weight entropy, resampling count, and proposal
coverage. Increase particle count or improve the proposal distribution before
assuming an implementation bug.

## Circular Mean Jumps Near `pi`

Angles wrap. Compare circular distances and trigonometric moments rather than raw
Euclidean differences around the branch cut.

## Backend Changed After Import

`PYRECEST_BACKEND` is read before `pyrecest.backend` is imported. Start a fresh
Python process when switching the process-wide facade backend. Use
`pyrecest.backends.get_backend(name)` for explicit backend inspection without
changing the facade.

## PyTorch Gradients Disappear

Some PyTorch backend helpers use NumPy/SciPy fallback paths. Treat capability
rows marked `partial` as numerically useful but not necessarily differentiable or
GPU-preserving.

## JAX Fails In Assignment-Heavy Code

JAX works best with functional array updates and pure transformations. APIs that
rely on mutation, SciPy-only functionality, or global random state may be marked
partial or unsupported for JAX.

## Tracker Births Or Deaths Look Unexpected

Inspect association costs, gating thresholds, missed-detection probabilities,
clutter assumptions, and track-management thresholds. Association diagnostics
should be added to new tracker workflows where possible.
