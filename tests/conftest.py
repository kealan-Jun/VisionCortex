import pytest

from visioncortex.config import load_config
from repo_paths import ROOT


@pytest.fixture
def default_config():
    return load_config(ROOT / "configs/default.yaml")
