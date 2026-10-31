"""Small, explicit real-glyph adapters for difficult display fonts.

These are source-pinned glyph classifiers, not stored readings. All slots use
one source frame and one shared translation. Missing evidence remains unknown;
template matching cannot establish decimal positions or business facts.
"""

import base64
import hashlib
import json
import re

import numpy as np

from .temporal_display import _identity, _reference
from .temporal_registration import normalized_box

REVISION = "vc-native-glyph-prototypes/1"
FEATURE_SHAPE = (3, 24, 16)


def feature(image, box):
    import cv2

    x, y, xx, yy = box
    crop = image[y:yy, x:xx].astype(float)
    features = []
    spans = []
    for c in range(3):
        signal = crop[:, :, c]
        low, high = np.percentile(signal, [10, 95])
        spans.append(float(high - low))
        normalized = np.clip((signal - low) / max(high - low, 8), 0, 1)
        normalized = cv2.resize(normalized, (16, 24), interpolation=cv2.INTER_LINEAR)
        normalized -= normalized.mean()
        normalized /= max(np.linalg.norm(normalized), 1e-8)
        features.append(normalized)
    return np.stack(features).ravel() / np.sqrt(3), spans


def calibrate(examples, *, revision):
    """Examples contain reviewed single-glyph labels and exact source refs.

    Caller owns label review and train/test separation. This function does not
    claim labels are independent truth merely because provenance is present.
    """
    if not isinstance(revision, str) or not 1 <= len(revision) <= 128:
        raise ValueError("Explicit glyph adapter revision required")
    if not isinstance(examples, list) or not 2 <= len(examples) <= 12:
        raise ValueError("Two to twelve reviewed real-glyph examples required")
    prototypes = []
    for example in examples:
        label = example.get("label")
        if not isinstance(label, str) or len(label) != 1 or label not in "0123456789":
            raise ValueError("Single digit labels only; never whole readings")
        if example.get("label_basis") != "reviewed_original_glyph":
            raise ValueError("Explicit original-glyph label review required")
        source, _ = _reference(example["source_ref"])
        image = example["image"]
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Native uint8 BGR glyph input required")
        box = normalized_box(example["box"], image.shape)
        if box[3] - box[1] < 12 or image.shape[0] * image.shape[1] > 262144:
            raise ValueError("Insufficient native glyph resolution or oversized ROI")
        f, spans = feature(image, box)
        if max(spans) < 20:
            raise ValueError("Calibration glyph lacks native contrast")
        x, y, xx, yy = box
        prototypes.append(
            dict(
                label=label,
                source_ref=source,
                native_box=list(box),
                native_bgr_sha256=hashlib.sha256(
                    image[y:yy, x:xx].tobytes()
                ).hexdigest(),
                feature_b64=base64.b64encode(f.astype("<f4").tobytes()).decode("ascii"),
                native_channel_spans=spans,
                label_basis=example["label_basis"],
            )
        )
    if len({p["label"] for p in prototypes}) < 2:
        raise ValueError("At least two distinct calibrated classes required")
    return dict(
        method=REVISION,
        revision=revision,
        prototypes=prototypes,
        minimum_similarity=0.85,
        minimum_class_margin=0.12,
        maximum_shared_shift_pixels=2,
        score_is_accuracy=False,
        accepts_whole_values=False,
        decimal_calibration=False,
    )


def recognize(image, layout, adapter):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Native uint8 BGR numeric ROI required")
    if image.shape[0] * image.shape[1] > 262144:
        raise ValueError("Glyph ROI exceeds its pixel budget")
    if (
        adapter.get("method") != REVISION
        or len(json.dumps(adapter).encode()) > 96 * 1024
    ):
        raise ValueError("Bounded versioned glyph adapter required")
    minimum, margin = (
        adapter.get("minimum_similarity"),
        adapter.get("minimum_class_margin"),
    )
    if (
        minimum != 0.85
        or margin != 0.12
        or adapter.get("maximum_shared_shift_pixels") != 2
    ):
        raise ValueError("Glyph abstention gates cannot be weakened")
    prototypes = adapter.get("prototypes")
    if not isinstance(prototypes, list) or not 2 <= len(prototypes) <= 12:
        raise ValueError("Bounded single-glyph prototypes required")
    vectors, labels = [], []
    for p in prototypes:
        if (
            p.get("label") not in tuple("0123456789")
            or p.get("label_basis") != "reviewed_original_glyph"
        ):
            raise ValueError("Only reviewed individual digit labels are supported")
        _reference(p["source_ref"])
        raw = base64.b64decode(p["feature_b64"], validate=True)
        f = np.frombuffer(raw, dtype="<f4")
        if (
            f.size != np.prod(FEATURE_SHAPE)
            or not np.isfinite(f).all()
            or not 0.95 <= np.linalg.norm(f) <= 1.05
        ):
            raise ValueError("Invalid or degenerate glyph feature")
        labels.append(p["label"])
        vectors.append(f)
    classes = sorted(set(labels))
    if len(classes) < 2:
        raise ValueError("At least two distinct calibrated classes required")
    digits = layout.get("digits", [])
    if not 1 <= len(digits) <= 12:
        raise ValueError("Explicit per-field digit-slot geometry required")
    boxes = [
        normalized_box(d.get("box") if isinstance(d, dict) else d, image.shape)
        for d in digits
    ]
    if any(a[2] > b[0] for a, b in zip(boxes, boxes[1:], strict=False)):
        raise ValueError("Digit slots must be ordered and cannot overlap")
    if any(b[3] - b[1] < 12 for b in boxes):
        raise ValueError("Native glyph height below twelve pixels")
    matrix = np.stack(vectors)
    best = None
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            shifted = [[x + dx, y + dy, xx + dx, yy + dy] for x, y, xx, yy in boxes]
            if any(
                x < 0 or y < 0 or xx > image.shape[1] or yy > image.shape[0]
                for x, y, xx, yy in shifted
            ):
                continue
            observations = []
            for box in shifted:
                f, spans = feature(image, box)
                scores = matrix @ f
                rank = sorted(
                    [
                        (
                            c,
                            max(
                                float(scores[i])
                                for i, label in enumerate(labels)
                                if label == c
                            ),
                        )
                        for c in classes
                    ],
                    key=lambda r: r[1],
                    reverse=True,
                )
                score, distance = rank[0][1], rank[0][1] - rank[1][1]
                known = score >= minimum and distance >= margin and max(spans) >= 20
                observations.append(
                    dict(
                        glyph=rank[0][0] if known else None,
                        best_candidate=rank[0][0],
                        similarity=score,
                        class_margin=distance,
                        native_box=box,
                        native_channel_spans=spans,
                        calibrated_probability=False,
                    )
                )
            quality = (
                sum(d["glyph"] is not None for d in observations),
                min(d["similarity"] for d in observations),
                sum(d["similarity"] for d in observations),
            )
            if best is None or quality > best[0]:
                best = quality, observations, [dx, dy]
    if best is None:
        raise ValueError("All native glyph search windows are clipped")
    _, observations, shift = best
    pattern = "".join(d["glyph"] or "?" for d in observations)
    return dict(
        method=REVISION,
        adapter_revision=adapter["revision"],
        adapter_sha256=hashlib.sha256(
            json.dumps(adapter, sort_keys=True).encode()
        ).hexdigest(),
        glyphs=observations,
        glyph_pattern=pattern,
        complete="?" not in pattern,
        shared_source_shift=shift,
        calibrated_classes=classes,
        decimal_status="unknown",
        value_text=None,
        confirmed=False,
        template_labels_are_not_independent_acceptance_truth=True,
    )


def reconcile(glyph_observation, ocr_observation):
    """Offer a review suggestion from the same original frame and field.

    Unknown stroke positions stay unknown even when OCR proposes a character.
    A suggestion is never a numeric measurement or independent visual proof.
    """
    from .video_numeric import number

    a, _ = _identity(glyph_observation["identity"])
    b, _ = _identity(ocr_observation["identity"])
    source_a, time_a = _reference(glyph_observation["source_ref"])
    source_b, time_b = _reference(ocr_observation["source_ref"])
    common_hashes = [
        k
        for k in ("source_frame_pixels_sha256", "original_frame_rgb_sha256")
        if source_a.get(k)
        and source_a.get(k) == source_b.get(k)
        and isinstance(source_a[k], str)
        and re.fullmatch("[0-9a-f]{64}", source_a[k])
    ]
    if (
        a != b
        or time_a != time_b
        or source_a["source_sha256"] != source_b["source_sha256"]
        or not common_hashes
    ):
        raise ValueError(
            "Glyph/OCR suggestions require the exact same original frame and field"
        )
    decoded = glyph_observation["result"]
    glyphs = decoded["glyphs"]
    text = number(ocr_observation.get("raw_text"))
    result = dict(
        glyph_pattern=decoded["glyph_pattern"],
        raw_ocr_text=ocr_observation.get("raw_text"),
        suggestion=None,
        numeric_candidate=None,
        confirmed=False,
        source_ref=source_a,
        unverified_positions=[i for i, g in enumerate(glyphs) if g["glyph"] is None],
        method="same_source_glyph_OCR_review/1",
    )
    if text is None or len(text.replace(".", "")) != len(glyphs):
        return dict(result, reason="OCR_does_not_cover_registered_slots")
    digits = text.replace(".", "")
    proposed = "".join(g["glyph"] or digits[i] for i, g in enumerate(glyphs))
    # Preserve any model-reported decimal at exactly its original position.
    if "." in text:
        i = text.index(".")
        proposed = proposed[:i] + "." + proposed[i:]
    conflicts = [
        dict(slot=i, glyph=g["glyph"], OCR=digits[i])
        for i, g in enumerate(glyphs)
        if g["glyph"] is not None and g["glyph"] != digits[i]
    ]
    return dict(
        result,
        suggestion=proposed,
        conflicts=conflicts,
        reason="review_suggestion_unknown_strokes_and_decimal_not_confirmed",
    )
