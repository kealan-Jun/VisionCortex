"""Compatibility entry point; implementation lives in tools/evaluation/replay_archive_acceptance.py."""
from importlib import import_module
from pathlib import Path
import sys

# Also supports script execution from a portable package without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_implementation = import_module("tools.evaluation.replay_archive_acceptance")


def __getattr__(name):
    return getattr(_implementation, name)


if __name__ == "__main__":
    raise SystemExit(_implementation.main())
