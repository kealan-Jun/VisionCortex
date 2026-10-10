"""Guarded entrypoint for the existing worker; no new model pipeline."""

import argparse
import os
import re
import stat
from pathlib import Path

from .config import load_config, require_configured_site


def _guard_environment(name):
    return os.environ.get("VISIONCORTEX_" + name, os.environ.get("REALITYLOOP_" + name))


def drain_requested():
    path = _guard_environment("DRAIN_FILE")
    return bool(path and Path(path).exists())


def validate_gpu_guard():
    """Validate and retain the inherited shared lock before worker startup."""
    if os.name != "posix":
        raise RuntimeError("Start through deployment/inference/gpu_guard.py on Linux/WSL2")
    import fcntl

    try:
        gpu = _guard_environment("GPU_UUID") or ""
        if not re.fullmatch(r"GPU-[A-Za-z0-9-]+", gpu):
            raise ValueError("Invalid GPU UUID")
        fd = int(_guard_environment("GPU_GUARD_FD") or "-1")
        inherited = os.fstat(fd)
        if not stat.S_ISREG(inherited.st_mode):
            raise ValueError("GPU guard requires a regular lock file")
        configured_path = _guard_environment("GPU_LOCK_PATH")
        # Historical launchers export only FD/UUID; Linux identifies their
        # already-open file without trusting a newly chosen lock location.
        lock_path = Path(configured_path or os.readlink(f"/proc/self/fd/{fd}"))
        if (
            not lock_path.is_absolute()
            or lock_path.name != gpu + ".lock"
            or lock_path.is_symlink()
        ):
            raise ValueError("GPU guard lock identity is invalid")
        expected = lock_path.stat()
        if (inherited.st_dev, inherited.st_ino) != (expected.st_dev, expected.st_ino):
            raise ValueError("GPU guard FD does not match its lock file")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, ValueError) as exc:
        raise RuntimeError("Start through deployment/inference/gpu_guard.py") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    settings = load_config(args.config)
    require_configured_site(settings)
    for name in ("device_day", "collection_ingest"):
        if settings.get(name, {}).get("enabled"):
            raise RuntimeError(
                "Machine worker requires automatic NAS producers disabled"
            )
    validate_gpu_guard()
    from .cli import worker_command

    worker_command(args.config)


if __name__ == "__main__":
    main()
