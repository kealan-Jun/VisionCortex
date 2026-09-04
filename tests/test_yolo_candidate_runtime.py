from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from labvision_evidence.yolo_candidate_runtime import (
    _exact_batches,
    export_and_benchmark_yolo_candidate,
)


def test_candidate_runtime_builds_only_complete_static_batches():
    assert _exact_batches(["a", "b", "c", "d", "e"], 2) == [
        ["a", "b"],
        ["c", "d"],
    ]
    with pytest.raises(ValueError, match="positive"):
        _exact_batches(["a"], 0)


def test_candidate_runtime_rejects_mismatched_integrity_audit_before_export(
    tmp_path,
):
    dataset = tmp_path / "dataset"
    (dataset / "images" / "val").mkdir(parents=True)
    (dataset / "labels" / "val").mkdir(parents=True)
    (dataset / "images" / "val" / "image.jpg").write_bytes(b"image")
    (dataset / "labels" / "val" / "image.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    (dataset / "dataset.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(dataset),
                "train": "images/val",
                "val": "images/val",
                "test": "images/val",
                "names": {0: "hand"},
            }
        ),
        encoding="utf-8",
    )
    (dataset / "dataset-receipt.json").write_text(
        json.dumps(
            {
                "schema_version": "visioncortex-mapped-public-yolo-union/1",
                "status": "completed",
                "truth_status": "public_human_annotations",
                "classes": ["hand"],
                "source_copy_bytes": 0,
                "nas_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    model = tmp_path / "model.pt"
    model.write_bytes(b"model")
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "schema_version": "visioncortex-yolo-dataset-integrity-audit/1",
                "passed": True,
                "dataset_receipt_sha256": "0" * 64,
                "cross_split_content_hash_count": 0,
                "source_copy_bytes": 0,
                "nas_accessed": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="passing matching"):
        export_and_benchmark_yolo_candidate(
            model,
            dataset,
            audit,
            tmp_path / "runtime",
            export_batch=1,
            benchmark_image_limit=1,
        )


def test_candidate_runtime_exports_static_engine_and_chunks_exact_batches(
    monkeypatch, tmp_path
):
    dataset = tmp_path / "dataset"
    (dataset / "images" / "val").mkdir(parents=True)
    (dataset / "labels" / "val").mkdir(parents=True)
    for index in range(4):
        (dataset / "images" / "val" / f"image-{index}.jpg").write_bytes(
            f"image-{index}".encode()
        )
        (dataset / "labels" / "val" / f"image-{index}.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )
    (dataset / "dataset.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(dataset),
                "train": "images/val",
                "val": "images/val",
                "test": "images/val",
                "names": {0: "hand"},
            }
        ),
        encoding="utf-8",
    )
    receipt = dataset / "dataset-receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": "visioncortex-mapped-public-yolo-union/1",
                "status": "completed",
                "truth_status": "public_human_annotations",
                "classes": ["hand"],
                "source_copy_bytes": 0,
                "nas_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "schema_version": "visioncortex-yolo-dataset-integrity-audit/1",
                "passed": True,
                "dataset_receipt_sha256": hashlib.sha256(
                    receipt.read_bytes()
                ).hexdigest(),
                "cross_split_content_hash_count": 0,
                "source_copy_bytes": 0,
                "nas_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    model = tmp_path / "model.pt"
    model.write_bytes(b"model")
    export_calls = []
    prediction_batch_sizes = []

    class FakeYOLO:
        def __init__(self, model_path, task=None):
            self.model_path = model_path
            self.names = {0: "hand"}

        def export(self, **kwargs):
            export_calls.append(kwargs)
            engine = Path(self.model_path).with_suffix(".engine")
            engine.write_bytes(b"engine")
            return str(engine)

        def predict(self, *, source, **kwargs):
            prediction_batch_sizes.append(len(source))
            return [SimpleNamespace() for _ in source]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))

    report = export_and_benchmark_yolo_candidate(
        model,
        dataset,
        audit,
        tmp_path / "runtime",
        export_batch=2,
        benchmark_image_limit=4,
    )

    assert export_calls[0]["dynamic"] is False
    assert export_calls[0]["batch"] == 2
    assert report["benchmarks"][0]["image_count"] == 4
    assert report["export"]["benchmark_batch_policy"] == (
        "exact_static_export_batch_only"
    )
    assert set(prediction_batch_sizes) == {2}
