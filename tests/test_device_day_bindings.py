"""Completed-provider bindings preserve provenance without granting execution."""
from copy import deepcopy
from pathlib import Path

import pytest

from visioncortex.device_day_bindings import (
    SETTING, CompletedBindingInvalid, CompletedExecutionBindings,
    make_entry, make_hold, settings_snapshot, write_manifest,
)
from visioncortex.device_day_contract import atomic_json, digest, file_hash, read_json


@pytest.fixture
def example(tmp_path):
    config = {
        "mllm": {"provider": "aliyun", "model": "old-model", "enabled": True,
                 "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                 "api_protocol": "chat_completions", "quality_mode": "quality",
                 "max_output_tokens": 8192, "api_key_env": "SAFE_TEST_ENV"},
        "speech_recognition": {"model": "fun-asr-flash-2026-06-15", "adapter_sha256": "a" * 64},
    }
    recording = {"recording_id": "record-1", "source_signature": "capture-1"}
    receipt = {"recording_id": "record-1", "stage": "understanding", "status": "completed",
               "key": "b" * 64, "artifacts": [], "provider": "aliyun"}
    entry = make_entry("understanding", recording, receipt, receipt["key"], config,
                       queue_key="c" * 64, queue_revision="d" * 64)
    ref = write_manifest(tmp_path / "bindings.json", [entry])
    return config, recording, receipt, entry, CompletedExecutionBindings({SETTING: ref}), ref


def test_original_recipe_and_receipt_remain_separate_from_new_provider(example):
    old, recording, receipt, entry, bindings, _ = example
    current = deepcopy(old)
    current["mllm"].update(provider="nianfeng", model="gpt-6-sol")
    snapshot = deepcopy(current)
    selected = bindings.candidate("understanding", recording)
    historical = bindings.config_for(current, selected)
    assert historical["mllm"]["provider"] == "aliyun"
    assert historical["mllm"]["model"] == "old-model"
    assert "api_key_env" not in historical["mllm"]
    assert bindings.match_key(entry, receipt["key"])
    assert bindings.match_key(entry, entry["queue_key"])
    assert not bindings.match_key(entry, digest(["current-or-changed-inputs"]))
    assert bindings.accepts(receipt, receipt["key"], queue_revision="d" * 64)
    assert not bindings.accepts(receipt, receipt["key"], queue_revision="e" * 64)
    assert current == snapshot


def test_bound_completion_can_never_fall_through_to_model_execution(example):
    _, _, receipt, _, bindings, _ = example
    for invalid in (None, {}, receipt | {"provider": "nianfeng"}, receipt | {"status": "failed"},
                    receipt | {"artifacts": [{"path": "tampered"}]}):
        with pytest.raises(CompletedBindingInvalid, match="historical key"):
            bindings.require_valid(invalid, receipt["key"])
    # Artifact verification reports None even when the JSON itself is intact.
    assert bindings.require_valid(receipt, receipt["key"]) is receipt
    assert bindings.require_valid(None, "f" * 64) is None


def test_source_or_record_change_does_not_select_history(example):
    _, recording, _, _, bindings, _ = example
    assert bindings.candidate("understanding", recording)
    assert bindings.candidate("understanding", recording | {"source_signature": "changed"}) is None
    assert bindings.candidate("understanding", recording | {"recording_id": "other"}) is None
    assert bindings.candidate("stt", recording) is None


def test_manifest_must_match_its_exact_digest(example):
    *_, reference = example
    path = Path(reference["path"])
    value = read_json(path)
    value["entries"][0]["settings"]["mllm"]["model"] = "different"
    atomic_json(path, value)
    with pytest.raises(ValueError, match="checksum"):
        CompletedExecutionBindings({SETTING: reference})


def test_duplicate_sources_or_keys_are_rejected(example, tmp_path):
    _, _, _, entry, _, _ = example
    with pytest.raises(ValueError, match="Duplicate"):
        write_manifest(tmp_path / "duplicate.json", [entry, deepcopy(entry)])
    with pytest.raises(ValueError, match="Duplicate"):
        write_manifest(tmp_path / "collision.json", [entry, entry | {"recording_id": "other"}])


def test_uncompleted_receipts_and_unreviewed_aliases_cannot_be_pinned(example):
    config, recording, receipt, _, _, _ = example
    for invalid in (receipt | {"status": "failed"}, receipt | {"status": "running"},
                    receipt | {"recording_id": "other"}, receipt | {"stage": "vision"}):
        with pytest.raises(ValueError, match="exact completed"):
            make_entry("understanding", recording, invalid, receipt["key"], config)
    with pytest.raises(ValueError, match="original execution recipe"):
        make_entry("understanding", recording, receipt, "a" * 64, config)
    # Existing audited historical checkpoint support preserves BOTH identities.
    entry = make_entry("understanding", recording, receipt, "a" * 64, config,
                       original_accepts=lambda actual, key: actual == receipt and key == "a" * 64)
    assert entry["expected_key"] == "a" * 64 and entry["receipt_key"] == receipt["key"]


def test_asr_snapshot_preserves_implicit_binding_key_without_credentials(example, tmp_path):
    config, recording, receipt, _, _, _ = example
    original_speech = deepcopy(config["speech_recognition"])
    receipt = receipt | {"stage": "stt"}
    entry = make_entry("stt", recording, receipt, receipt["key"], config)
    ref = write_manifest(tmp_path / "stt.json", [entry])
    bindings = CompletedExecutionBindings({SETTING: ref})
    current = deepcopy(config)
    current["speech_recognition"]["connection"] = {
        "provider": "aliyun", "credential_ref": "approved-untracked-reference"}
    selected = bindings.config_for(current, bindings.candidate("stt", recording))
    assert selected["speech_recognition"] == original_speech
    assert current["speech_recognition"]["connection"]
    assert bindings.require_valid(receipt, receipt["key"]) == receipt


def test_secrets_and_malformed_manifests_are_rejected(example, tmp_path):
    config, recording, receipt, entry, _, _ = example
    config["speech_recognition"]["connection"] = {"api_key": "fixture-is-not-a-real-key"}
    with pytest.raises(ValueError, match="credentials"):
        settings_snapshot(config, "stt")
    for invalid in (entry | {"queue_key": None}, entry | {"source_signature": ""},
                    entry | {"settings": {"mllm": {"api_key": "forbidden"}}}):
        with pytest.raises(ValueError):
            write_manifest(tmp_path / "invalid.json", [invalid])
    path = tmp_path / "wrong-schema.json"
    atomic_json(path, {"schema_version": "wrong", "entries": []})
    with pytest.raises(ValueError, match="Unknown"):
        CompletedExecutionBindings({SETTING: {"path": str(path), "sha256": file_hash(path)}})


def test_report_preserves_original_understanding_without_provider_relabelling(example):
    config, recording, receipt, _, _, _ = example
    receipt = receipt | {"stage": "report", "understanding_key": "old-understanding"}
    entry = make_entry("report", recording, receipt, receipt["key"], config)
    assert entry["settings"] == {}
    assert entry["receipt_digest"] == digest(receipt)


@pytest.mark.parametrize("protocol", ["ark_responses", "chat_completions"])
@pytest.mark.parametrize("provider", ["aliyun", "nianfeng", "zhipu"])
def test_snapshot_preserves_every_request_policy_field(example, protocol, provider):
    from visioncortex.scene_requests import request_policy
    config, *_ = example
    config = deepcopy(config)
    config["mllm"].update(
        provider=provider, api_protocol=protocol, quality_mode="balanced",
        enabled=False, max_images_per_group=17, temperature=.123,
        max_output_tokens=4321, request_image_max_edge=1024,
        request_image_jpeg_quality=71, compact_scene_metadata=True,
        # Operational fields do not belong to historical generation identity.
        timeout_seconds=444, connection_verification={"unrelated": "receipt"},
    )
    selected = CompletedExecutionBindings.config_for(
        {"mllm": {"provider": "new-provider"}},
        {"settings": settings_snapshot(config, "understanding")})
    assert request_policy(selected) == request_policy(config)
    for key in ("model", "provider", "base_url", "enabled", "max_images_per_group", "temperature"):
        assert selected["mllm"].get(key) == config["mllm"].get(key)
    assert "timeout_seconds" not in selected["mllm"]
    assert "connection_verification" not in selected["mllm"]


def test_hold_preserves_only_the_original_completed_source_without_accepting_results(example, tmp_path):
    _, recording, receipt, _, _, _ = example
    hold = make_hold("understanding", recording, "a" * 64, status="completed",
                     observed_key="b" * 64, observed_queue_key="c" * 64)
    reference = write_manifest(tmp_path / "hold.json", [], holds=[hold])
    bindings = CompletedExecutionBindings({SETTING: reference})
    assert bindings.held("understanding", recording) == hold
    assert bindings.held("understanding", recording, key="b" * 64) == hold
    assert bindings.held("understanding", recording, key="c" * 64) == hold
    # The key includes full source/vision/STT/context: new comments or protocol release the hold.
    assert bindings.held("understanding", recording, key="d" * 64) is None
    assert bindings.held("understanding", recording | {"source_signature": "new-input"}) is None
    assert bindings.held("understanding", recording | {"recording_id": "new-record"}) is None
    assert bindings.held("stt", recording) is None
    assert bindings.candidate("understanding", recording) is None
    assert not bindings.requires(receipt["key"])
    assert not bindings.accepts(receipt, receipt["key"])


@pytest.mark.parametrize("status", ["queued", "failed", "running", "waiting_for_prerequisite"])
def test_unfinished_jobs_cannot_enter_completion_hold(example, status):
    _, recording, *_ = example
    with pytest.raises(ValueError, match="pre-existing"):
        make_hold("understanding", recording, "a" * 64, status=status,
                  observed_key="b" * 64, observed_queue_key="c" * 64)


def test_completion_hold_cannot_override_a_verified_binding_or_duplicate(example, tmp_path):
    _, recording, _, entry, _, _ = example
    hold = make_hold("understanding", recording, "a" * 64, status="completed",
                     observed_key="b" * 64, observed_queue_key="c" * 64)
    with pytest.raises(ValueError, match="conflicting"):
        write_manifest(tmp_path / "conflict.json", [entry], holds=[hold])
    with pytest.raises(ValueError, match="conflicting"):
        write_manifest(tmp_path / "duplicate-hold.json", [], holds=[hold, hold])
