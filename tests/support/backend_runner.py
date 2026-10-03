"""Subprocess helpers for import-time backend portability tests."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_PATH = str(_REPO_ROOT / "src")


@dataclass(frozen=True)
class BackendRunResult:
    backend: str
    returncode: int
    stdout: str
    stderr: str


def _subprocess_pythonpath(existing_pythonpath: str | None) -> str:
    """Return a subprocess PYTHONPATH that prefers this checkout's sources."""

    if not existing_pythonpath:
        return _SRC_PATH

    paths = existing_pythonpath.split(os.pathsep)
    return os.pathsep.join([_SRC_PATH, *(path for path in paths if path != _SRC_PATH)])


def run_backend_code(
    backend: str, code: str, *, timeout: float = 30.0
) -> BackendRunResult:
    env = os.environ.copy()
    env["PYRECEST_BACKEND"] = backend
    env["PYTHONPATH"] = _subprocess_pythonpath(env.get("PYTHONPATH"))
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )
    return BackendRunResult(
        backend=backend,
        returncode=int(completed.returncode),
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
