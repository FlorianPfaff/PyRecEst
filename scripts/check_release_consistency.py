#!/usr/bin/env python3
"""Check local and, optionally, remote release metadata consistency."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import tomllib
import urllib.error
import urllib.parse
import urllib.request

REPOSITORY = "FlorianPfaff/PyRecEst"
PYPI_PROJECT = "pyrecest"
ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_local_version() -> str:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return pyproject["tool"]["poetry"]["version"]


def _normalise_tag(tag: str) -> str:
    return tag.removeprefix("v")


def _load_json(url: str) -> dict:
    headers = {"Accept": "application/json"}
    token = os.environ.get("GITHUB_TOKEN")
    parsed_url = urllib.parse.urlparse(url)
    if token and parsed_url.scheme == "https" and parsed_url.netloc == "api.github.com":
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def _check_citation_metadata() -> list[str]:
    errors: list[str] = []
    citation_path = ROOT / "CITATION.cff"
    if not citation_path.exists():
        return ["CITATION.cff is missing"]

    citation = citation_path.read_text(encoding="utf-8")
    expected_pairs = {
        "title": 'title: "PyRecEst: Recursive Bayesian Estimation for Python"',
        "repository-code": 'repository-code: "https://github.com/FlorianPfaff/PyRecEst"',
    }
    for label, expected in expected_pairs.items():
        if expected not in citation:
            errors.append(f"CITATION.cff does not contain expected {label}: {expected}")
    return errors


def _check_remote(local_version: str) -> list[str]:
    errors: list[str] = []
    try:
        latest_release = _load_json(
            f"https://api.github.com/repos/{REPOSITORY}/releases/latest"
        )
        github_version = _normalise_tag(str(latest_release.get("tag_name", "")))
        if github_version != local_version:
            errors.append(
                f"GitHub latest release is {github_version!r}, but pyproject.toml is {local_version!r}"
            )
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        errors.append(f"Could not read GitHub latest release metadata: {exc}")

    try:
        pypi = _load_json(f"https://pypi.org/pypi/{PYPI_PROJECT}/json")
        pypi_version = str(pypi.get("info", {}).get("version", ""))
        if pypi_version != local_version:
            errors.append(
                f"PyPI latest version is {pypi_version!r}, but pyproject.toml is {local_version!r}"
            )
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        errors.append(f"Could not read PyPI metadata: {exc}")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expect-version",
        help="Require pyproject.toml to contain this exact version.",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Only check files in the source tree.",
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Also compare against GitHub latest release and PyPI.",
    )
    args = parser.parse_args(argv)

    local_version = _load_local_version()
    errors = _check_citation_metadata()

    if args.expect_version and local_version != args.expect_version:
        errors.append(
            f"pyproject.toml version is {local_version!r}, expected {args.expect_version!r}"
        )

    if args.remote and not args.local_only:
        errors.extend(_check_remote(local_version))

    if errors:
        for error in errors:
            print(f"release-consistency: {error}", file=sys.stderr)
        return 1

    mode = "local" if args.local_only or not args.remote else "local+remote"
    print(f"release-consistency: {mode} checks passed for {local_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
