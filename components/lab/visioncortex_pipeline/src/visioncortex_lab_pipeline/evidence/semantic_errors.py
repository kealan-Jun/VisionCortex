"""Shared failure types for semantic evidence stages."""

from __future__ import annotations


class SemanticAnalysisUnavailable(RuntimeError):
    """A resumable semantic stage failed before it could adjudicate evidence."""
