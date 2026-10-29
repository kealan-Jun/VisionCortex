"""Retire only recognized generated views; immutable evidence is outside this scope."""

import json
from field_recognition.archive.archive_integrity import relative_file

OLD_CATEGORIES = {
    "Bindings",
    "InstrumentReadings",
    "VoicePhotos",
    "PhotoDrafts",
    "Photos",
    "Handoffs",
}


def retire_generated_views(root, current, previous=()):
    """Return legacy URL -> canonical URL mappings, with no business payload copies.

    Unknown/user files are never removed. An old generated JSON must declare its
    own schema and entity before its companions can be retired.
    """
    mappings = {}
    candidates = {p for p in previous if p.endswith(".json")}
    for category in OLD_CATEGORIES | {"ExperimentRecords", "WorkbenchReadings"}:
        folder = root / "Browse" / category
        if folder.is_dir():
            candidates.update(p.relative_to(root).as_posix() for p in folder.glob("**/Record.json"))
    for relative in sorted(candidates):
        if relative in current or not relative.startswith(("01_业务数据/", "Browse/")):
            continue
        source = relative_file(root, relative)
        try:
            if source.stat().st_size > 16 * 1024 * 1024:
                continue
            data = json.loads(source.read_text())
        except (OSError, ValueError):
            continue
        if data.get("schema") not in {
            "field-recognition-readable/1",
            "field-recognition-browse/1",
        } or not data.get("derived_view"):
            continue
        target = (
            "Records/"
            + "".join(w.title() for w in data["entity"].split("_"))
            + "/"
            + str(data["entity_id"])
            + "/Record.json"
        )
        if target not in current:
            continue
        directory = source.parent
        mappings[relative] = target
        for filename in ("Readme.html", "查看记录.html", "Measurement.json"):
            old = directory / filename
            if not old.is_file():
                continue
            text = old.read_text()
            generated = (
                ("原始回执版本" in text or "完整证据与修订历史" in text)
                if old.suffix == ".html"
                else ('"photo_time"' in text and '"values"' in text)
            )
            if generated:
                mappings[old.relative_to(root).as_posix()] = (
                    target.replace("Record.json", "Readme.html")
                    if old.suffix == ".html"
                    else target
                )
                old.unlink()
        source.unlink()
        stop = root / ("01_业务数据" if relative.startswith("01_业务数据/") else "Browse")
        parent = directory
        while parent != stop:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
    category_map = {
        "Bindings": "BindingEvents",
        "InstrumentReadings": "PanelReadings",
        "VoicePhotos": "VoicePhotoReadings",
        "PhotoDrafts": "MeasurementDrafts",
        "Photos": "SourceMaterials",
        "Handoffs": "DeviceHandoffs",
    }
    for old, new in category_map.items():
        relative = "Browse/" + old + "/Readme.html"
        path = root / relative
        if (
            path.is_file()
            and not list(path.parent.glob("**/Record.json"))
            and "返回分类入口" in path.read_text()
        ):
            mappings[relative] = "Browse/" + new + "/Readme.html"
            path.unlink()
            try:
                path.parent.rmdir()
            except OSError:
                pass
    # Retire recognized category pages after their records are gone.
    for relative in previous:
        if (
            relative in current
            or not relative.startswith("01_业务数据/")
            or not relative.endswith("打开目录.html")
        ):
            continue
        path = relative_file(root, relative)
        if path.is_file() and "返回归档导航" in path.read_text():
            path.unlink()
            mappings[relative] = "Readme.html"
            try:
                path.parent.rmdir()
            except OSError:
                pass
    try:
        (root / "01_业务数据").rmdir()
    except OSError:
        pass
    return mappings
