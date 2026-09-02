from pathlib import Path

import pytest

from visioncortex.config import load_config


@pytest.fixture
def default_config():
    return load_config(Path("configs/default.yaml"))

