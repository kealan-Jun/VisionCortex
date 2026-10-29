"""Load an explicitly selected local configuration without executing shell code."""

import os
from pathlib import Path


def read_environment_file(path: str | Path) -> dict[str, str]:
    """Read literal values without expanding variables or executing commands."""
    from dotenv import dotenv_values

    path = Path(path).expanduser()
    if not path.is_file():
        raise ValueError(f"Environment file does not exist: {path}")
    return {
        key: value
        for key, value in dotenv_values(path, interpolate=False, encoding="utf-8-sig").items()
        if value is not None
    }


def load_environment_file(path: str | Path) -> None:
    """Keep process variables authoritative; never search for another project's file.

    Use literal values so the same simple KEY=value profile can be used by
    systemd's EnvironmentFile. Shell expansion and commands are not supported.
    """
    for key, value in read_environment_file(path).items():
        os.environ.setdefault(key, value)
