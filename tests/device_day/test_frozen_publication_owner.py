"""A separate immutable producer owns receipt keys; rendering never runs models."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import subprocess

import pytest

from visioncortex.device_day_publication_worker import FrozenReceiptKeyOwner


def owner_case(tmp_path):
    source = tmp_path / "Producer"
    package = source / "src/visioncortex"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "config.py").write_text("def load_config(path): return {}\n")
    (package / "ai_settings.py").write_text("def apply_active(config): return config\n")
    (package / "device_day.py").write_text("""import hashlib,json
class DeviceDayRunner:
    def __init__(self, config, backend):
        assert type(backend) is object
    def _key(self, stage, recording, inputs, *, _binding=True):
        value = [stage,recording,inputs,_binding]
        if stage == 'report': value.append(self._context(None,recording))
        return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    def process(self,*args,**kwargs): raise AssertionError("No execution endpoint")
""")
    for args in (
        ["init", "-q"],
        ["add", "src"],
        [
            "-c",
            "user.name=Owned Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "Owned immutable fixture",
        ],
    ):
        subprocess.run(["git", *args], cwd=source, check=True, capture_output=True)
    config = tmp_path / "Owner.json"
    config.write_text("{}")
    setting = {
        "schema_version": "visioncortex-frozen-receipt-key-owner/1",
        "source_root": str(source),
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip(),
        "config_path": str(config),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    }
    return setting, source, config


def test_real_key_only_ipc_preserves_exact_inputs_and_serializes_concurrent_readers(
    tmp_path,
):
    setting, source, config = owner_case(tmp_path)
    owner = FrozenReceiptKeyOwner(setting)
    requests = [
        ("vision", {"recording_id": str(i)}, {"source": "original", "n": i})
        for i in range(4)
    ]
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            actual = list(pool.map(lambda values: owner.key(*values), requests))
        expected = [
            hashlib.sha256(
                json.dumps([*request, True], sort_keys=True).encode()
            ).hexdigest()
            for request in requests
        ]
        assert actual == expected
        with pytest.raises(ValueError, match="rejected"):
            owner.key("execute_model", {}, {})
    finally:
        owner.close()
    assert owner.process.poll() is not None


@pytest.mark.parametrize("changed", ["source", "config"])
def test_frozen_owner_drift_fails_closed_before_a_child_is_started(tmp_path, changed):
    setting, source, config = owner_case(tmp_path)
    if changed == "source":
        (source / "src/visioncortex/device_day.py").write_text("changed")
    else:
        config.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="changed"):
        FrozenReceiptKeyOwner(setting)


def test_report_context_is_an_explicit_input_without_another_nas_reader(tmp_path):
    setting, source, config = owner_case(tmp_path)
    owner = FrozenReceiptKeyOwner(setting)
    values = [
        "report",
        {"recording_id": "r"},
        {"understanding": "actual saved result"},
        True,
    ]
    context = {"events": [], "field_event_status": "no_archive_for_camera_day"}
    try:
        actual = owner.key(*values[:3], context=context)
        assert (
            actual
            == hashlib.sha256(
                json.dumps([*values, context], sort_keys=True).encode()
            ).hexdigest()
        )
    finally:
        owner.close()
