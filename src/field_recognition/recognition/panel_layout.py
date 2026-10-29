"""Explicit model layout contracts. A visual class alone never grants occupancy."""

LEGACY_LAYOUTS = {
    "0": {"name": "panel_a", "layout": "horizontal_pair", "measurements": ["温度", "转速"]},
    "1": {"name": "panel_b", "layout": "single", "measurements": ["质量"]},
}


def normalize_classes(classes):
    if not classes or any(not str(key).isdigit() for key in classes):
        raise ValueError("Invalid model classes")
    result = {}
    for key, value in classes.items():
        config = LEGACY_LAYOUTS.get(str(key), {}) | value
        if not config.get("name") or config.get("layout") not in {"single", "horizontal_pair"}:
            raise ValueError("Each class requires an explicit name and display layout")
        count = 2 if config["layout"] == "horizontal_pair" else 1
        if len(config.get("measurements", [])) != count:
            raise ValueError("Display layout and fields do not match")
        if not config.get("instrument_id") and not config.get("type_id"):
            raise ValueError("Class requires an asset or type mapping")
        result[str(key)] = config
    return result


def resolve_type_boxes(boxes, allowed, get_instrument):
    result = []
    for box in boxes:
        item = dict(box)
        if not item.get("instrument_id") and item.get("type_id"):
            matching = [
                ident
                for ident in allowed
                if (get_instrument(ident) or {}).get("type_id") == item["type_id"]
            ]
            # Occupancy narrows eligibility, but does not identify the physical object.
            # Generic type models need QR/spatial instance evidence before attribution.
            item.update(
                instrument_id=None,
                association_issue="instance_evidence_required" if matching else "unbound_type",
            )
        result.append(item)
    return result


def roles_for_boxes(boxes):
    roles = {}
    groups = {}
    for box in boxes:
        class_id = str(box["class_id"])
        layout = LEGACY_LAYOUTS.get(class_id, {}) | box
        group = (class_id, box.get("instrument_id"))
        groups.setdefault(group, []).append((box, layout))
    for group, items in groups.items():
        items.sort(key=lambda pair: pair[0]["xyxy"][0])
        contract = items[0][1]
        names = contract.get("measurements", [])
        if contract.get("layout") == "single":
            for box, _ in items:
                roles[(group, tuple(box["xyxy"]))] = names[0] if names else None
        elif len(items) == 2 and len(names) == 2:
            a, b = [pair[0]["xyxy"] for pair in items]
            overlap = min(a[3], b[3]) - max(a[1], b[1])
            # Small detector borders can overlap although the display centers
            # and digits are distinct. Do not lose both field names for that.
            width = min(a[2] - a[0], b[2] - b[0])
            if a[2] - b[0] <= 0.15 * width and overlap >= 0.4 * min(a[3] - a[1], b[3] - b[1]):
                roles[(group, tuple(a))] = names[0]
                roles[(group, tuple(b))] = names[1]
    return roles
