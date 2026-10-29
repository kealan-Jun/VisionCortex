"""Compatibility publication entry points preserve every historical path."""


def reconcile_outputs(runner):
    """Keep old callers compatible without relocating any published material.

    Current selection belongs to the verified interval/file-index publisher.
    An absent current candidate does not authorize moving its historical
    directory or breaking its old references. Capture catalogs retain their
    existing independent owner in ``device_day_capture_files``.
    """
    return {"relocated": [], "source_media_changed": False,
            "automatic_relocation": False, "historical_paths": "preserved",
            "superseded_detection": "not_performed"}


def write_archive_guide(runner):
    """CLI and service publish the same user-facing overview."""
    from .device_day_overview import ArchiveOverview
    return ArchiveOverview().publish(runner)
