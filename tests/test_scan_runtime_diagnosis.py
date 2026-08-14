import json

from labvision_evidence.archive import ArchiveLayout
from labvision_evidence.pipeline import EvidencePipeline


def test_scan_runtime_classifies_decode_starvation(tmp_path):
    layout = ArchiveLayout(tmp_path / "archive")
    layout.create()
    work = tmp_path / "scan"
    work.mkdir()
    (work / "runtime_fine_first_person.json").write_text(
        json.dumps(
            {
                "role_total_seconds": 100.0,
                "queue_wait_seconds": 60.0,
                "inference_seconds": 20.0,
                "tracking_and_ledger_seconds": 5.0,
                "inference_call_count": 100,
                "inference_frame_count": 200,
                "final_effective_batch_size": 8,
            }
        ),
        encoding="utf-8",
    )
    (work / "scheduler_fine.json").write_text(
        json.dumps({"mode": "sequential_role_residency"}),
        encoding="utf-8",
    )

    EvidencePipeline._archive_scan_runtime(layout, work, "fine")

    report = json.loads(
        (layout.json_config / "scan_runtime_fine.json").read_text(encoding="utf-8")
    )
    diagnosis = report["bottleneck_diagnosis"]
    assert diagnosis["classification"] == "decode_or_source_starved"
    assert diagnosis["actual_batch_size_mean"] == 2.0
    assert diagnosis["effective_batch_capacity_mean"] == 8.0
    assert diagnosis["batch_fill_ratio"] == 0.25
