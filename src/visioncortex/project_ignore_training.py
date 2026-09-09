"""Lazy public entry point for the optional partial-supervision training adapter.

Importing this module must work in a CPU-only core installation. Accessing an
adapter explicitly loads the optional training dependencies.
"""
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .training_runtime.ignore import (
        IgnoreDataset, IgnoreDetectionLoss, IgnoreLetterBox, IgnoreNegativeBCE,
        IgnoreTrainer, ObservedE2ELoss, ignored_anchor_mask, loss_usage,
        make_ignore_criterion,
    )

__all__ = [
    "ignored_anchor_mask", "IgnoreNegativeBCE", "IgnoreDetectionLoss",
    "ObservedE2ELoss", "make_ignore_criterion", "loss_usage",
    "IgnoreLetterBox", "IgnoreDataset", "IgnoreTrainer",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    implementation = import_module(".training_runtime.ignore", __package__)
    value = getattr(implementation, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
