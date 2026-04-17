"""
Environment helpers for local smoke-test scripts.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional


def load_env(override: bool = True, env_path: Optional[str] = None) -> None:
    """
    Load the first available .env file and normalize proxy variables.

    The client code reads credentials from process env, so this keeps the loader
    dependency-free and predictable for small standalone repos.
    """
    for candidate in _candidate_env_paths(env_path):
        if not candidate.is_file():
            continue
        _load_env_file(candidate, override=override)
        break
    _sync_proxy_env()


def _candidate_env_paths(env_path: Optional[str]) -> Iterable[Path]:
    if env_path:
        yield Path(env_path).expanduser()
        return

    current = Path.cwd().resolve()
    for directory in (current, *current.parents):
        yield directory / ".env"


def _load_env_file(path: Path, override: bool) -> None:
    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                continue
            if not override and key in os.environ:
                continue
            os.environ[key] = value


def _sync_proxy_env() -> None:
    mappings = [
        ("http_proxy", "HTTP_PROXY"),
        ("https_proxy", "HTTPS_PROXY"),
        ("all_proxy", "ALL_PROXY"),
        ("no_proxy", "NO_PROXY"),
    ]
    for lower, upper in mappings:
        if os.getenv(upper) is None:
            value = os.getenv(lower)
            if value:
                os.environ[upper] = value
