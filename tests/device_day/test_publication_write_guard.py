"""Owned local directories simulate mount loss; no NAS is disconnected."""
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from visioncortex import device_day_write_guard as guard_module, scene_requests
from visioncortex.device_day_contract import atomic_json, read_json
from visioncortex.multimodal_usage import UsageLedger
from test_paid_request_safety import FakeAnalyzer, invoke, setup_window


def mocked_storage(tmp_path, monkeypatch):
    volume = tmp_path/'PinnedVolume'
    volume.mkdir()
    state = {'mounted': True, 'marker_verified': 0}
    snapshot = {'mount_id': 'owned-test-mount', 'device': volume.stat().st_dev,
                'filesystem': 'cifs', 'source': '//synthetic.invalid/owned-test-share'}
    monkeypatch.setattr(guard_module, 'mount_snapshot', lambda: {volume: snapshot} if state['mounted'] else {})
    def marker(config):
        state['marker_verified'] += 1
        if not state['mounted']:
            raise ValueError('owned test mount is absent')
    monkeypatch.setattr('visioncortex.device_day_recovery_worker.verify_storage', marker)
    config = {'runtime': {'local_only': False}, 'storage': {
        'archive_root': str(volume/'Archive'), 'local_runtime_root': str(tmp_path/'Runtime'),
        'local_cache_root': str(tmp_path/'Cache')},
        'device_day': {'publication_storage_checks': [{'mount_path': str(volume),
            'filesystem': snapshot['filesystem'], 'source': snapshot['source']}]}}
    return config, volume, state


def test_absent_mount_never_creates_a_formal_output_parent(tmp_path, monkeypatch):
    config, volume, state = mocked_storage(tmp_path, monkeypatch)
    state['mounted'] = False
    target = volume/'NeverCreated'/'Result.json'
    with pytest.raises(guard_module.NASUnavailable), guard_module.storage_guard(config):
        atomic_json(target, {'unsafe': True})
    assert not target.parent.exists()


def test_production_without_explicit_mount_identity_fails_closed(tmp_path):
    target = tmp_path/'FakeArchive'/'Result.json'
    config = {'runtime': {'local_only': False}, 'device_day': {}, 'storage': {'archive_root': str(target.parent)}}
    with pytest.raises(guard_module.NASUnavailable), guard_module.storage_guard(config):
        atomic_json(target, {})
    assert not target.parent.exists()


def test_write_uses_verified_directory_fds_and_safe_replacement(tmp_path, monkeypatch):
    config, volume, state = mocked_storage(tmp_path, monkeypatch)
    target = volume/'Archive'/'Day'/'Index.json'
    with guard_module.storage_guard(config):
        atomic_json(target, {'version': 1})
        atomic_json(target, {'version': 2})
    assert read_json(target) == {'version': 2}
    assert not list(target.parent.glob('*.partial'))
    assert state['marker_verified'] > 2


def test_declared_storage_aliases_work_and_internal_artifact_symlinks_stay_rejected(tmp_path, monkeypatch):
    config, volume, _ = mocked_storage(tmp_path, monkeypatch)
    alias = tmp_path/'TrustedConfiguredAlias'
    alias.symlink_to(volume, target_is_directory=True)
    config['storage'].update(archive_root=str(alias/'Archive'), local_cache_root=str(alias/'Backend'))
    with guard_module.storage_guard(config):
        atomic_json(alias/'Archive'/'Day'/'Index.json', {'preserved_alias': True})
        atomic_json(alias/'Backend'/'Stage.json', {'actual': 'receipt'})
        outside = tmp_path/'Outside'
        outside.mkdir()
        (volume/'Archive'/'Escape').symlink_to(outside, target_is_directory=True)
        with pytest.raises(guard_module.NASUnavailable):
            atomic_json(alias/'Archive'/'Escape'/'Index.json', {})
    assert read_json(volume/'Archive'/'Day'/'Index.json')['preserved_alias']
    assert read_json(volume/'Backend'/'Stage.json')['actual'] == 'receipt'
    assert not (outside/'Index.json').exists()


def test_volume_marker_is_not_reread_for_each_directory_component(tmp_path, monkeypatch):
    config, volume, state = mocked_storage(tmp_path, monkeypatch)
    with guard_module.storage_guard(config):
        atomic_json(volume/'Archive'/'Deep'/'Day'/'Task'/'Frames'/'Sample'/'Result.json', {'safe': True})
    assert state['marker_verified'] == 2, 'scope admission and final atomic publication verify the marker'


def test_check_then_unmount_cannot_create_a_local_fallback_parent(tmp_path, monkeypatch):
    config, volume, state = mocked_storage(tmp_path, monkeypatch)
    pinned = tmp_path/'PreviousVolume'
    original = os.mkdir
    def detached(path, *args, **kwargs):
        if path == 'NewOutput' and kwargs.get('dir_fd') is not None and state['mounted']:
            # This occurs after validation, immediately before mkdirat. The
            # original directory fd stays pinned while the pathname changes.
            os.rename(volume, pinned)
            original(volume)
            state['mounted'] = False
        return original(path, *args, **kwargs)
    monkeypatch.setattr(guard_module.os, 'mkdir', detached)
    with pytest.raises(guard_module.NASUnavailable), guard_module.storage_guard(config):
        atomic_json(volume/'NewOutput'/'Result.json', {'received': True})
    assert not (volume/'NewOutput').exists()
    assert (pinned/'NewOutput').is_dir()
    assert not (pinned/'NewOutput'/'Result.json').exists()


def test_unverified_destination_and_symlink_parents_are_rejected(tmp_path, monkeypatch):
    config, volume, _ = mocked_storage(tmp_path, monkeypatch)
    outside = tmp_path/'UnverifiedArchive'
    with guard_module.storage_guard(config):
        with pytest.raises(guard_module.NASUnavailable):
            atomic_json(outside/'Result.json', {})
        outside.mkdir()
        (volume/'Escape').symlink_to(outside, target_is_directory=True)
        with pytest.raises(guard_module.NASUnavailable):
            atomic_json(volume/'Escape'/'Result.json', {})
    assert not (outside/'Result.json').exists()


def test_paid_response_survives_flight_mount_loss_and_replays_without_payment(default_config, tmp_path, monkeypatch):
    private, volume, state = mocked_storage(tmp_path, monkeypatch)
    config, metadata, _, _ = setup_window(default_config, tmp_path)
    config['runtime'].update(private['runtime'])
    config['storage'].update(private['storage'])
    config['device_day'].update(private['device_day'])
    window = config, metadata, volume/'Requests'/'Input.json', volume/'NewResponseDirectory'/'Result.json'
    class ReceivedThenDetached(FakeAnalyzer):
        def _call(self, *args, **kwargs):
            response = super()._call(*args, **kwargs)
            state['mounted'] = False
            return response
    first = ReceivedThenDetached()
    with pytest.raises(guard_module.NASUnavailable):
        invoke(window, first)
    assert first.calls == 1
    assert not window[3].parent.exists()
    assert UsageLedger(config).recover(scene_requests.request_key(
        config, 'saved-test-prompt', metadata)) is not None
    state['mounted'] = True
    restarted = FakeAnalyzer()
    assert invoke(window, restarted)[2]
    assert restarted.calls == 0
    assert read_json(window[3])['model_result']['request_id'] == 'synthetic-request'
    assert UsageLedger(config).summary(datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat())['calls'] == 1
