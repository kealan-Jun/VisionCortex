import json

from labvision_evidence.detection import _read_checkpoint, _write_checkpoint
from labvision_evidence.storage import IncrementalArchivePublisher


def test_detection_checkpoint_requires_matching_nonempty_ledger(tmp_path):
    checkpoint = tmp_path / "view.checkpoint.json"
    ledger = tmp_path / "view.detections.jsonl"
    ledger.write_text('{"frame":1}\n', encoding="utf-8")
    _write_checkpoint(checkpoint, {0, 1}, ledger)

    assert _read_checkpoint(checkpoint, ledger) == {0, 1}
    ledger.write_text("", encoding="utf-8")
    assert _read_checkpoint(checkpoint, ledger) == set()


def test_incremental_publisher_reuses_identical_destination(tmp_path):
    local = tmp_path / "local"
    nas = tmp_path / "nas"
    source = local / "JSON-Config-Files" / "result.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps({"passed": True}), encoding="utf-8")
    publisher = IncrementalArchivePublisher(local, nas)

    destination = publisher.publish_file(source)
    first_mtime = destination.stat().st_mtime_ns
    destination = publisher.publish_file(source)

    assert destination.read_bytes() == source.read_bytes()
    assert destination.stat().st_mtime_ns == first_mtime
