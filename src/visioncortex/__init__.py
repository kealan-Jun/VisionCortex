"""Multi-view wet-lab video evidence pipeline."""

from .runtime_environment import configure_third_party_runtime


ULTRALYTICS_CONFIG_DIR = configure_third_party_runtime()

__version__ = "0.1.0"
