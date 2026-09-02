from __future__ import annotations

import re
from typing import Any

from .schemas import EvidenceEvent


ACTION_SLUGS = {
    "hand_object_contact": "Hand-Object-Contact",
    "object_movement": "Object-Movement",
    "liquid_movement": "Liquid-Movement",
    "container_state_change": "Container-State-Change",
    "device_panel_operation": "Device-Panel-Operation",
    # Append instead of renumbering the established five archive categories.
    "pipette_transfer_operation": "Pipette-Transfer-Operation",
}

ACTION_CATEGORY_FOLDERS = {
    action_type: f"{index:02d}-{slug}"
    for index, (action_type, slug) in enumerate(ACTION_SLUGS.items(), 1)
}

OBJECT_NAME_ALIASES = {
    "hand": "Hand",
    "gloved_hand": "Gloved-Hand",
    "paper": "Weighing-Paper",
    "weighing_paper": "Weighing-Paper",
    "pipette": "Pipette",
    "spearhead": "Pipette-Tip",
    "spatula": "Spatula",
    "sample_bottle": "Sample-Bottle",
    "reagent_bottle": "Reagent-Bottle",
    "reagent_bottle_open": "Open-Reagent-Bottle",
    "tube": "Tube",
    "tube_rack": "Tube-Rack",
    "balance": "Analytical-Balance",
    "magnetic_stirrer": "Magnetic-Stirrer",
    "panel": "Device-Panel",
    "cap": "Container-Cap",
    "移液器": "Pipette",
    "移液枪": "Pipette",
    "枪头": "Pipette-Tip",
    "吸头": "Pipette-Tip",
    "药勺": "Spatula",
    "称量纸": "Weighing-Paper",
    "试剂瓶": "Reagent-Bottle",
    "离心管": "Centrifuge-Tube",
    "分析天平": "Analytical-Balance",
}

HAND_OBJECT_NAMES = {"hand", "gloved_hand", "手", "戴手套的手"}
LIQUID_TRANSFER_TOOL_NAMES = {
    "pipette",
    "spearhead",
    "pipette_tip",
    "spatula",
    "dropper",
    "移液器",
    "移液枪",
    "枪头",
    "吸头",
    "药勺",
    "滴管",
}


def key_material_action_folder(action_type: Any) -> str:
    value = str(getattr(action_type, "value", action_type))
    try:
        return ACTION_CATEGORY_FOLDERS[value]
    except KeyError as error:
        raise ValueError(
            f"Unsupported key-material action type: {value}"
        ) from error


def readable_object_label(value: str) -> str:
    normalized = str(value).strip()
    lowered = normalized.lower().replace("-", "_").replace(" ", "_")
    if normalized in OBJECT_NAME_ALIASES:
        return OBJECT_NAME_ALIASES[normalized]
    if lowered in OBJECT_NAME_ALIASES:
        return OBJECT_NAME_ALIASES[lowered]
    ascii_slug = re.sub(
        r"[^A-Za-z0-9_.-]+", "-", normalized.replace("_", "-")
    ).strip("-_")
    if ascii_slug:
        return "-".join(
            part.capitalize() for part in ascii_slug.split("-") if part
        )
    return "Unknown-Object"


def key_material_semantic_name(event: EvidenceEvent) -> dict[str, Any]:
    raw_objects = list(
        dict.fromkeys(
            str(item) for item in event.objects if str(item).strip()
        )
    )
    non_hand_objects = [
        item
        for item in raw_objects
        if item.strip().lower().replace("-", "_").replace(" ", "_")
        not in HAND_OBJECT_NAMES
    ]
    object_labels = [
        readable_object_label(item) for item in non_hand_objects
    ]
    if not object_labels:
        object_labels = ["Unknown-Object"]
    primary = object_labels[0]
    semantic_primary = primary
    action_type = event.action_type.value
    if action_type == "hand_object_contact":
        descriptor = f"Contact-Hand-With-{primary}"
    elif action_type == "object_movement":
        descriptor = f"Move-{primary}"
    elif action_type in {"liquid_movement", "pipette_transfer_operation"}:
        liquid_objects = [
            readable_object_label(item)
            for item in non_hand_objects
            if item.strip().lower().replace("-", "_").replace(" ", "_")
            not in LIQUID_TRANSFER_TOOL_NAMES
        ]
        source = liquid_objects[0] if liquid_objects else primary
        target = liquid_objects[1] if len(liquid_objects) > 1 else None
        semantic_primary = source
        descriptor = (
            f"Transfer-Liquid-{source}"
            if action_type == "liquid_movement"
            else f"Operate-Pipette-{source}"
        )
        if target:
            descriptor += f"-To-{target}"
    elif action_type == "container_state_change":
        descriptor = f"Change-State-{primary}"
    elif action_type == "device_panel_operation":
        descriptor = f"Operate-{primary}"
    else:
        descriptor = f"{ACTION_SLUGS.get(action_type, 'Action')}-{primary}"
    return {
        "file_stem": f"{descriptor}_{event.event_id}",
        "display_name_en": descriptor.replace("-", " "),
        "action_label_en": ACTION_SLUGS.get(action_type, action_type),
        "primary_object": semantic_primary,
        "object_labels": object_labels,
        "raw_object_labels": raw_objects,
    }
