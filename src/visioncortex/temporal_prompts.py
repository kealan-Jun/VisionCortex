"""Bounded, class-diverse geometric prompts; conflicting labels remain unresolved."""
from __future__ import annotations

import math

from .temporal_segmentation import _box_iou


def select_temporal_prompts(proposals: list[dict], *, max_objects: int = 8,
                            overlap_threshold: float = .85) -> tuple[list[dict], dict]:
    """Cluster only mutually near-identical boxes; do not claim physical identity.

    Complete-link membership prevents A-B-C overlap chains from merging A and C.
    The most confident member supplies geometry and a provisional label. Every
    original hypothesis remains in the group audit, including discarded groups.
    """
    if type(max_objects) is not int or not 1 <= max_objects <= 8:
        raise ValueError("Expected 1–8 temporal prompts")
    if not math.isfinite(overlap_threshold) or not .8 <= overlap_threshold <= 1:
        raise ValueError("Invalid near-identical geometry threshold")
    ids = set()
    for p in proposals:
        box = p["box"]
        if (not p["id"] or p["id"] in ids or not p["label"]
                or not math.isfinite(p["confidence"]) or not 0 <= p["confidence"] <= 1
                or len(box) != 4 or not all(math.isfinite(x) for x in box)
                or not 0 <= box[0] < box[2] or not 0 <= box[1] < box[3]):
            raise ValueError("Invalid temporal seed proposal")
        ids.add(p["id"])
    groups: list[list[dict]] = []
    for p in sorted(proposals, key=lambda p: (-p["confidence"], p["id"])):
        if p["label"] in {"hand", "gloved_hand"}:
            continue
        for group in groups:
            if all(_box_iou(p["box"], other["box"]) >= overlap_threshold for other in group):
                group.append(p)
                break
        else:
            groups.append([p])
    diverse, repeated, labels = [], [], set()
    for group in groups:
        label = group[0]["label"]
        (repeated if label in labels else diverse).append(group)
        labels.add(label)
    chosen = (diverse + repeated)[:max_objects]
    selected = []
    audit_groups = []
    for group in groups:
        record = {"representative_id": group[0]["id"],
                  "members": [{k: p[k] for k in ("id", "label", "confidence", "box")} for p in group],
                  "label_status": "ambiguous_model_proposals" if len({p["label"] for p in group}) > 1 else "model_proposal",
                  "physical_identity_confirmed": False,
                  "selected": any(group is g for g in chosen)}
        audit_groups.append(record)
    records = {g["representative_id"]: g for g in audit_groups}
    for group in chosen:
        p = group[0]
        selected.append({**p, "proposal_group": records[p["id"]]})
    return selected, {"schema_version": "visioncortex-temporal-prompt-selection/1",
                      "policy": "complete_link_geometry_then_class_diversity",
                      "overlap_threshold": overlap_threshold, "maximum_objects": max_objects,
                      "excluded_labels": ["hand", "gloved_hand"],
                      "groups": audit_groups, "classification_resolved": False}
