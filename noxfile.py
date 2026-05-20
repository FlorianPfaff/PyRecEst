"""Local development sessions mirroring the main CI entry points."""

from __future__ import annotations

import nox

PYTHON_DEFAULT = "3.13"


def _poetry_install(session: nox.Session, *args: str) -> None:
    session.run("poetry", "install", *args, external=True)


@nox.session(python=PYTHON_DEFAULT)
def tests_numpy(session: nox.Session) -> None:
    """Run the default NumPy test suite."""
    _poetry_install(session, "--with", "dev", "--extras", "healpy_support")
    session.env["PYRECEST_BACKEND"] = "numpy"
    session.run(
        "poetry",
        "run",
        "python",
        "-m",
        "pytest",
        "--rootdir",
        ".",
        "-v",
        "./tests",
        external=True,
    )


@nox.session(python=PYTHON_DEFAULT)
def tests_pytorch(session: nox.Session) -> None:
    """Run the PyTorch backend test suite."""
    _poetry_install(
        session, "--with", "dev", "--extras", "healpy_support pytorch_support"
    )
    session.env["PYRECEST_BACKEND"] = "pytorch"
    session.run(
        "poetry",
        "run",
        "python",
        "-m",
        "pytest",
        "--rootdir",
        ".",
        "-v",
        "./tests",
        external=True,
    )


@nox.session(python=PYTHON_DEFAULT)
def tests_jax(session: nox.Session) -> None:
    """Run the JAX backend test suite."""
    _poetry_install(session, "--with", "dev", "--extras", "healpy_support jax_support")
    session.env["PYRECEST_BACKEND"] = "jax"
    session.env["JAX_ENABLE_X64"] = "True"
    session.run(
        "poetry",
        "run",
        "python",
        "-m",
        "pytest",
        "--rootdir",
        ".",
        "-v",
        "./tests",
        external=True,
    )


@nox.session(python=PYTHON_DEFAULT)
def docs(session: nox.Session) -> None:
    """Build documentation in strict mode."""
    _poetry_install(session, "--with", "docs", "--without", "dev")
    session.env["PYTHONPATH"] = "src"
    session.run("poetry", "run", "mkdocs", "build", "--strict", external=True)


@nox.session(python=PYTHON_DEFAULT)
def numerical_stress(session: nox.Session) -> None:
    """Run slower numerical-stability tests."""
    _poetry_install(session, "--with", "dev", "--extras", "healpy_support")
    session.env["PYRECEST_BACKEND"] = "numpy"
    session.run(
        "poetry",
        "run",
        "python",
        "-m",
        "pytest",
        "--rootdir",
        ".",
        "-v",
        "-m",
        "numerical_stress",
        "./tests",
        external=True,
    )


@nox.session(python=PYTHON_DEFAULT)
def benchmarks(session: nox.Session) -> None:
    """Run optional benchmark tests."""
    _poetry_install(session, "--with", "dev")
    session.env["PYRECEST_BACKEND"] = "numpy"
    session.run(
        "poetry",
        "run",
        "python",
        "-m",
        "pytest",
        "-m",
        "benchmark",
        "./benchmarks",
        external=True,
    )
