"""Pin uploaded originals in MetaData while preserving legacy source references."""

from pathlib import Path
from field_recognition.capture.photo_sources import photo_location


def source_relative(root, source, *, source_roots=()):
    external = source.get("external_photo") or {}
    reference = external.get("source_ref")
    camera = source.get("camera_id") or external.get("camera_id")
    if not reference or not camera:
        return None
    path = Path(reference)
    for prefix in (root, *source_roots):
        try:
            relative = path.relative_to(prefix)
            location = photo_location(relative.as_posix())
        except ValueError:
            continue
        if location.camera != camera:
            return None
        if Path(prefix) == Path(root):
            # During handover keep capture-owned old-layout files in place.
            # If a verified migration moved them, use their MetaData location.
            metadata = Path(root) / location.metadata_path
            if not path.exists() and metadata.is_file():
                return location.metadata_path
            return relative.as_posix()
        return location.metadata_path
    return None
