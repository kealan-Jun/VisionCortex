from pathlib import Path

from visioncortex.pipeline import build_cache_identity
from visioncortex.schemas import RunManifest, ViewInput, ViewRole


def _manifest(tmp_path: Path) -> RunManifest:
    views = []
    for view_id, role in (("fp", ViewRole.FIRST_PERSON), ("tp", ViewRole.THIRD_PERSON)):
        video = tmp_path / f"{view_id}.mp4"
        clock = tmp_path / f"{view_id}.csv"
        video.write_bytes(view_id.encode("ascii"))
        clock.write_text("frame_index,timestamp_ms\n0,0\n", encoding="utf-8")
        views.append(ViewInput(view_id=view_id, role=role, video=video, timestamps_csv=clock))
    return RunManifest(experiment_id="benchmark", views=views)


def test_cache_identity_changes_with_detection_configuration(default_config, tmp_path):
    first_weight = tmp_path / "first.pt"
    third_weight = tmp_path / "third.pt"
    first_weight.write_bytes(b"first")
    third_weight.write_bytes(b"third")
    default_config["models"]["first_person"] = str(first_weight)
    default_config["models"]["third_person"] = str(third_weight)
    manifest = _manifest(tmp_path)

    original = build_cache_identity(default_config, manifest)
    default_config["performance"]["detection_fps"] += 1
    modified = build_cache_identity(default_config, manifest)

    assert original["cache_key"] != modified["cache_key"]
    assert original["models"]["first_person"]["sha256"]
    assert len(original["inputs"]) == 2


def test_cache_identity_ignores_run_specific_storage_paths(default_config, tmp_path):
    first_weight = tmp_path / "first.pt"
    third_weight = tmp_path / "third.pt"
    first_weight.write_bytes(b"first")
    third_weight.write_bytes(b"third")
    default_config["models"]["first_person"] = str(first_weight)
    default_config["models"]["third_person"] = str(third_weight)
    manifest = _manifest(tmp_path)

    original = build_cache_identity(default_config, manifest)
    default_config["project"]["output_root"] = str(tmp_path / "runs" / "new-run")
    default_config["project"]["preprocessing_acceptance_only"] = True
    default_config["storage"]["active_archive_path"] = str(tmp_path / "staging" / "new-run")
    default_config["storage"]["archive_root"] = str(tmp_path / "other-nas")
    modified = build_cache_identity(default_config, manifest)

    assert original["cache_key"] == modified["cache_key"]


def test_cache_identity_isolated_by_namespace_but_shared_by_cold_hot_mode(
    default_config, tmp_path
):
    first_weight = tmp_path / "first.pt"
    third_weight = tmp_path / "third.pt"
    first_weight.write_bytes(b"first")
    third_weight.write_bytes(b"third")
    default_config["models"]["first_person"] = str(first_weight)
    default_config["models"]["third_person"] = str(third_weight)
    manifest = _manifest(tmp_path)

    default_config["project"]["cache_namespace"] = "audit-a"
    default_config["project"]["cache_mode"] = "cold"
    cold = build_cache_identity(default_config, manifest)
    default_config["project"]["cache_mode"] = "reuse"
    hot = build_cache_identity(default_config, manifest)
    default_config["project"]["cache_namespace"] = "audit-b"
    isolated = build_cache_identity(default_config, manifest)

    assert cold["cache_key"] == hot["cache_key"]
    assert cold["execution_cache_policy"]["mode"] == "cold"
    assert hot["execution_cache_policy"]["mode"] == "reuse"
    assert isolated["cache_key"] != hot["cache_key"]


def test_downstream_settings_do_not_invalidate_cv(default_config, tmp_path):
    manifest = _manifest(tmp_path)
    original = build_cache_identity(default_config, manifest)
    default_config['mllm']['model'] = 'another-understanding-model'
    default_config['mllm']['timeout_seconds'] = 321
    default_config['speech_recognition'] = {'enabled': True, 'language': 'en'}
    default_config['daily_report'] = {'title': 'new format'}
    manifest.views[0].audio_offset_ms = 2500
    changed = build_cache_identity(default_config, manifest)
    assert changed['cache_key'] == original['cache_key']
    assert changed['downstream_dependencies'] != original['downstream_dependencies']
    manifest.views[0].view_id = 'different-camera'
    assert build_cache_identity(default_config, manifest)['cache_key'] != original['cache_key']
