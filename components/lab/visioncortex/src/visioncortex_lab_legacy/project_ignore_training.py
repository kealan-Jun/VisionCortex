"""Lazy compatibility entry point for the optional partial-supervision adapter.

Application and evidence tooling can import this module without installing or
initializing the training stack. Explicit access to a training symbol loads the
implementation; historical import paths remain valid for callers/checkpoints.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .training.ignore import (
        IgnoreDataset, IgnoreDetectionLoss, IgnoreLetterBox, IgnoreNegativeBCE,
        IgnoreTrainer, ObservedE2ELoss, ignored_anchor_mask, loss_usage,
        make_ignore_criterion,
    )

__all__ = [
    "IgnoreDataset", "IgnoreDetectionLoss", "IgnoreLetterBox",
    "IgnoreNegativeBCE", "IgnoreTrainer", "ObservedE2ELoss",
    "ignored_anchor_mask", "loss_usage", "make_ignore_criterion",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    implementation = import_module(".training.ignore", __package__)
    value = getattr(implementation, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
