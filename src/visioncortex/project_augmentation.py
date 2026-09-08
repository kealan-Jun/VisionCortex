"""Validate explicit offline partial-supervision derivatives before model startup.

Original annotations stay frozen. This checks geometry and provenance, not image
quality or generalization. Online augmentation remains disabled by the trainer.
"""

from __future__ import annotations

from collections import Counter
import copy
import math


def yolo_labels(boxes, width, height):
    lines = []
    for box in boxes:
        x1, y1, x2, y2 = box["xyxy_px"]
        lines.append(
            f"{box['class_id']} {(x1 + x2) / 2 / width:.9f} {(y1 + y2) / 2 / height:.9f} "
            f"{(x2 - x1) / width:.9f} {(y2 - y1) / height:.9f}"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def validate_partial_augmentation(root, receipt, parents, digest):
    from PIL import Image

    aug = receipt.get("augmentation", {})
    policy = aug.get("policy", {})
    variants = policy.get("variants_per_image")
    if (
        policy.get("schema_version") != "annotation-workbench-augmentation/1"
        or type(variants) is not int or not 1 <= variants <= 8
        or aug.get("counts_as_new_sources") is not False
        or aug.get("validation_and_test") != "originals_only_unchanged"
        or not aug.get("records")
    ):
        raise ValueError("Offline augmentation needs its explicit policy and training-only records")
    expected, seen, counts = {}, set(), Counter()
    train = {iid: r for iid, r in parents.items() if r["annotation"]["split"] == "train"}
    for d in aug["records"]:
        parent = train.get(d.get("source_image_id"))
        if parent is None:
            raise ValueError("Augmentation parent is not a frozen training image")
        a, s = parent["annotation"], parent["source"]
        index = d.get("variant_index")
        key = (parent["id"], index)
        if type(index) is not int or not 0 <= index < variants or key in seen:
            raise ValueError("Duplicate or invalid augmentation variant")
        seen.add(key)
        identity = dict(source_id=parent["id"], source_sha256=s["sha256"],
                        annotation_sha256=digest(a), policy=policy, index=index,
                        supervision="outside_ignore")
        iid = "awaug-" + digest(identity)[:24]
        fields = dict(
            id=iid, source_revision=parent["revision"], source_path=s["path"],
            source_sha256=s["sha256"], source_group=s["source_group"],
            camera_id=s["camera_id"], baseline_exposure=s["baseline_exposure"], license=s["license"],
            annotation_sha256=digest(a), role=a["role"], split="train",
            truth_status="project_annotations", independent_ground_truth=False,
            derivation="deterministic_training_augmentation", counts_as_new_source=False,
            policy_sha256=digest(policy), supervision="outside_ignore",
            output_size=[s["width"], s["height"]],
            image_file=f"{a['role']}/images/train/{iid}.png",
            label_file=f"{a['role']}/labels/train/{iid}.txt",
            operation_order=["brightness", "contrast", "gamma_power", "gaussian_blur",
                             "resize_and_pad", "jpeg_444_then_png"],
        )
        if iid in parents or any(d.get(k) != v for k, v in fields.items()):
            raise ValueError("Augmentation changed its parent, role, split or derivation identity")
        matrix = d.get("source_to_output_matrix")
        if (
            not isinstance(matrix, list) or len(matrix) != 3
            or any(not isinstance(row, list) or len(row) != 3 for row in matrix)
            or any(type(x) not in {int, float} or not math.isfinite(x) for row in matrix for x in row)
        ):
            raise ValueError("Invalid augmentation geometry")
        sx, tx, sy, ty = matrix[0][0], matrix[0][2], matrix[1][1], matrix[1][2]
        width, height = s["width"], s["height"]
        nw, nh = round(sx * width), round(sy * height)
        if (
            matrix != [[sx, 0, tx], [0, sy, ty], [0, 0, 1]]
            or not (0 < sx <= 1 and 0 < sy <= 1 and tx >= 0 and ty >= 0)
            or sx != nw / width or sy != nh / height
            or tx != round(tx) or ty != round(ty)
            or tx + nw > width or ty + nh > height
            or abs(sx - sy) > max(1 / width, 1 / height)
        ):
            raise ValueError("Offline augmentation must preserve every source pixel without cropping")
        transformed = {}
        for field in ["boxes", "ignore_regions"]:
            transformed[field] = copy.deepcopy(a[field])
            for box in transformed[field]:
                x1, y1, x2, y2 = box["xyxy_px"]
                box["xyxy_px"] = [x1 * sx + tx, y1 * sy + ty, x2 * sx + tx, y2 * sy + ty]
            if d.get(field) != transformed[field]:
                raise ValueError("Known boxes and ignore regions must retain every instance with the same transform")
        ann_sha = digest(transformed)
        if d.get("derived_annotation_sha256") != ann_sha:
            raise ValueError("Derived annotation identity differs from transformed regions")
        for kind in ["image", "label"]:
            if receipt["files"].get(d[f"{kind}_file"]) != d.get(f"{kind}_sha256"):
                raise ValueError("Missing pinned augmentation file")
        with Image.open(root / d["image_file"]) as image:
            if image.size != (width, height) or image.format != "PNG":
                raise ValueError("Derived image dimensions or encoding differ from its geometry")
        if (root / d["label_file"]).read_text() != yolo_labels(d["boxes"], width, height):
            raise ValueError("Derived YOLO labels differ from the frozen transformed instances")
        expected[d["image_file"]] = dict(image_id=iid, source_size=[width, height],
                                        annotation_sha256=ann_sha,
                                        xyxy_px=[r["xyxy_px"] for r in d["ignore_regions"]])
        counts[a["role"]] += 1
    if seen != {(iid, i) for iid in train for i in range(variants)} or dict(counts) != aug.get("derived_roles"):
        raise ValueError("Augmentation membership or original/derived counts differ from its policy")
    # In v2, also verify original label text, not just its file hash.
    for row in parents.values():
        a, s = row["annotation"], row["source"]
        label = root / a["role"] / "labels" / a["split"] / f"{row['id']}.txt"
        if label.read_text() != yolo_labels(a["boxes"], s["width"], s["height"]):
            raise ValueError("Original labels differ from frozen annotations")
    return expected
