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
    assert diagnosis["inference_frames_per_second"] == 10.0
    assert diagnosis["inference_milliseconds_per_call"] == 200.0


def test_scan_runtime_aggregates_progressive_passes(tmp_path):
    layout = ArchiveLayout(tmp_path / "archive")
    layout.create()
    work = tmp_path / "scan"
    for name, role, frames in (
        ("pass-00-primary", "first_person", 80),
        ("pass-01-supplemental", "third_person", 40),
    ):
        pass_dir = work / name
        pass_dir.mkdir(parents=True)
        (pass_dir / f"runtime_fine_{role}.json").write_text(
            json.dumps(
                {
                    "role": role,
                    "role_total_seconds": 10.0,
                    "queue_wait_seconds": 2.0,
                    "inference_seconds": 6.0,
                    "tracking_and_ledger_seconds": 1.0,
                    "inference_call_count": 10,
                    "inference_frame_count": frames,
                    "final_effective_batch_size": 8,
                }
            ),
            encoding="utf-8",
        )
        (pass_dir / "scheduler_fine.json").write_text(
            json.dumps({"mode": "sequential_role_residency", "active_view_ids": [role]}),
            encoding="utf-8",
        )

    progressive = {"enabled": True, "passes": [{"pass_index": 0}, {"pass_index": 1}]}
    EvidencePipeline._archive_scan_runtime(
        layout, work, "fine", progressive_report=progressive
    )

    report = json.loads(
        (layout.json_config / "scan_runtime_fine.json").read_text(encoding="utf-8")
    )
    assert len(report["role_reports"]) == 2
    assert {item["scan_pass"] for item in report["role_reports"]} == {
        "pass-00-primary",
        "pass-01-supplemental",
    }
    assert report["scheduler"]["mode"] == "progressive_cross_view"
    assert len(report["scheduler"]["passes"]) == 2
    assert report["progressive_cross_view"] == progressive
    assert report["bottleneck_diagnosis"]["inference_frame_count"] == 120
    assert report["bottleneck_diagnosis"]["inference_frames_per_second"] == 10.0
    assert report["bottleneck_diagnosis"]["inference_milliseconds_per_call"] == 600.0


def test_fine_runtime_does_not_absorb_nested_fine_scout_reports(tmp_path):
    layout = ArchiveLayout(tmp_path / "archive")
    layout.create()
    work = tmp_path / "scan"
    formal = work / "pass-00-primary"
    scout = work / "scout"
    formal.mkdir(parents=True)
    scout.mkdir(parents=True)
    (formal / "runtime_fine_first_person.json").write_text(
        json.dumps(
            {
                "role": "first_person",
                "inference_call_count": 2,
                "inference_frame_count": 16,
                "inference_seconds": 1.0,
            }
        ),
        encoding="utf-8",
    )
    (scout / "runtime_fine_scout_third_person.json").write_text(
        json.dumps(
            {
                "role": "third_person",
                "inference_call_count": 100,
                "inference_frame_count": 800,
                "inference_seconds": 50.0,
            }
        ),
        encoding="utf-8",
    )

    EvidencePipeline._archive_scan_runtime(layout, work, "fine")
    formal_report = json.loads(
        (layout.json_config / "scan_runtime_fine.json").read_text(encoding="utf-8")
    )
    assert len(formal_report["role_reports"]) == 1
    assert formal_report["bottleneck_diagnosis"]["inference_frame_count"] == 16

    EvidencePipeline._archive_scan_runtime(layout, scout, "fine_scout")
    scout_report = json.loads(
        (layout.json_config / "scan_runtime_fine_scout.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(scout_report["role_reports"]) == 1
    assert scout_report["bottleneck_diagnosis"]["inference_frame_count"] == 800
