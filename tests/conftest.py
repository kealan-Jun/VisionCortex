from pathlib import Path

import pytest

from labvision_evidence.config import load_config


@pytest.fixture
def default_config():
    return load_config(Path("configs/default.yaml"))

