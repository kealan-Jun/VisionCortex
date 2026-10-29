"""Repository paths and explicit optional research/private-fixture gates."""
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def research_helpers():
    """Research scripts are optional, separately supplied source, never site data."""
    raw = os.environ.get("VC_LAB_RESEARCH_HELPERS")
    if not raw:
        pytest.skip("Optional research helpers require VC_LAB_RESEARCH_HELPERS", allow_module_level=True)
    path = Path(raw)
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise ValueError("Research helpers require an explicit existing local directory")
    return path


def private_fixture(name):
    """Real source checks stay distinct from generated algorithm fixtures."""
    raw = os.environ.get("VC_LAB_PRIVATE_FIXTURE_ROOT")
    if not raw:
        pytest.skip("Real-source acceptance requires VC_LAB_PRIVATE_FIXTURE_ROOT")
    root = Path(raw)
    path = root / name
    if not root.is_absolute() or root.is_symlink() or not path.is_file():
        pytest.skip("Requested private source fixture is not supplied")
    return path


def decoder_python():
    """Use an explicitly selected decoder interpreter; never a site prefix."""
    return os.environ.get("VC_LAB_DECODER_PYTHON", sys.executable)
