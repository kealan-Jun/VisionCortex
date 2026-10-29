"""Shared collection preparation, analysis and guarded publication order."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml


@dataclass(frozen=True)
class CollectionPolicy:
    """Entry-specific acceptance gates, explicit rather than inherited from HTTP."""

    require_cold_execution: bool = False
    configure_scan: Callable | None = None
    before_publish: Callable | None = None


@dataclass(frozen=True)
class CollectionResult:
    output: Path
    ingest: dict[str, Any]
    partial: bool
    promotion: dict[str, Any] | None
    policy_receipts: dict[str, Any]


class CollectionWorkflow:
    """One order for every entry; adapters supply their validation and receipts."""

    def __init__(self, *, prepare, analyze, is_partial, promote):
        self.prepare = prepare
        self.analyze = analyze
        self.is_partial = is_partial
        self.promote = promote

    def execute(
        self,
        *,
        settings,
        experiment_id,
        staging_root,
        fixed_root,
        history_root,
        policy: CollectionPolicy,
        ingest_progress=None,
        on_prepared=None,
        on_partial=None,
        before_promotion=None,
        promotion_metadata=None,
    ) -> CollectionResult:
        if (
            policy.require_cold_execution
            and settings.get("project", {}).get("cache_mode") != "cold"
        ):
            raise ValueError("Formal collection policy requires cold execution")
        manifest, manifest_path, ingest = self.prepare(
            settings, experiment_id, ingest_progress
        )
        receipts = {}
        if policy.configure_scan is not None:
            receipts["scan_policy"] = policy.configure_scan(settings, ingest)
        manifest_path.write_text(
            yaml.safe_dump(
                manifest.model_dump(mode="json"), allow_unicode=True, sort_keys=False
            ),
            encoding="utf-8",
        )
        if on_prepared is not None:
            on_prepared(manifest, manifest_path, ingest)
        output = Path(self.analyze(manifest))
        if self.is_partial(output):
            if on_partial is not None:
                on_partial(output)
            return CollectionResult(output, ingest, True, None, receipts)
        if policy.before_publish is not None:
            receipts.update(policy.before_publish(output) or {})
        if before_promotion is not None:
            before_promotion(output)
        args = (staging_root, fixed_root, history_root)
        if promotion_metadata is not None:
            args += (promotion_metadata,)
        promotion = self.promote(*args)
        return CollectionResult(output, ingest, False, promotion, receipts)
