#!/usr/bin/env python3
"""Run a PyRecEst scenario from a source checkout."""

from __future__ import annotations

from pyrecest.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["run-scenario", *(__import__("sys").argv[1:])]))
