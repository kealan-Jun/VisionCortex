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


def test_lazy_source_read_failure_is_not_reported_as_nas_write_rejection(tmp_path, monkeypatch):
    config, volume, _ = mocked_storage(tmp_path, monkeypatch)
    target = volume/'Archive'/'Result.json'
    missing = tmp_path/'MissingSource.json'
    def blocks():
        yield b'{'
        with missing.open('rb') as handle:
            yield handle.read()
    with guard_module.storage_guard(config) as guard:
        with pytest.raises(FileNotFoundError) as caught:
            guard.atomic(target, blocks())
    assert caught.value.filename == str(missing)
    assert not target.exists()
    assert not list(target.parent.glob('*.partial'))


def test_destination_failure_preserves_errno_and_atomic_operation(tmp_path, monkeypatch):
    import errno
    config, volume, _ = mocked_storage(tmp_path, monkeypatch)
    target = volume/'Archive'/'Result.json'
    def denied(*args, **kwargs):
        raise PermissionError(errno.EACCES, 'owned test replacement denied')
    monkeypatch.setattr(guard_module.os, 'replace', denied)
    with guard_module.storage_guard(config):
        with pytest.raises(guard_module.NASUnavailable) as caught:
            atomic_json(target, {'safe': True})
    assert caught.value.errno == errno.EACCES
    assert caught.value.operation == 'replace_destination'
    assert caught.value.filename == str(target)
    assert not target.exists()
    assert not list(target.parent.glob('*.partial'))


def test_transient_cifs_rename_denial_retries_the_same_synced_file(tmp_path, monkeypatch):
    import errno
    config, volume, state = mocked_storage(tmp_path, monkeypatch)
    target = volume/'Archive'/'Result.json'
    original = guard_module.os.replace
    attempts = []
    def temporarily_denied(source, destination, **kwargs):
        attempts.append((source, destination, kwargs['src_dir_fd']))
        if len(attempts) < 3:
            raise PermissionError(errno.EACCES, 'owned transient sharing lease')
        return original(source, destination, **kwargs)
    monkeypatch.setattr(guard_module.os, 'replace', temporarily_denied)
    monkeypatch.setattr(guard_module.time, 'sleep', lambda _: None)
    with guard_module.storage_guard(config):
        atomic_json(target, {'safe': True})
    assert len(attempts) == 3 and len(set(attempts)) == 1
    assert read_json(target) == {'safe': True}
    assert state['marker_verified'] >= 4


def test_retry_revalidates_mount_before_trying_the_rename_again(tmp_path, monkeypatch):
    import errno
    config, volume, state = mocked_storage(tmp_path, monkeypatch)
    attempts = []
    def detached(*args, **kwargs):
        attempts.append(True)
        state['mounted'] = False
        raise PermissionError(errno.EACCES, 'owned transient denial')
    monkeypatch.setattr(guard_module.os, 'replace', detached)
    monkeypatch.setattr(guard_module.time, 'sleep', lambda _: None)
    with guard_module.storage_guard(config):
        with pytest.raises(guard_module.NASUnavailable):
            atomic_json(volume/'Archive'/'Result.json', {})
    assert len(attempts) == 1


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


def source_storage(tmp_path, monkeypatch):
    config, volume, state = mocked_storage(tmp_path, monkeypatch)
    capture = tmp_path/'CaptureTree'
    capture.mkdir()
    alias = tmp_path/'ConfiguredCaptureAlias'
    alias.symlink_to(capture, target_is_directory=True)
    config['collection_ingest'] = {'source_root': str(alias)}
    source = capture/'lab_video'
    source.mkdir()
    original = source/'rgb.mp4'
    original.write_bytes(b'owned synthetic input')
    snapshot = guard_module.mount_snapshot()[volume]
    mounts = {volume: snapshot, source: snapshot | {'mount_id': 'owned-source-mount'}}
    monkeypatch.setattr(guard_module, 'mount_snapshot', lambda: dict(mounts))
    return config, volume, state, alias, original, mounts


def test_source_reader_accepts_configured_alias_without_granting_writes(tmp_path, monkeypatch):
    config, volume, _, alias, original, _ = source_storage(tmp_path, monkeypatch)
    before = original.read_bytes()
    with guard_module.storage_guard(config) as guard:
        mount, resolved = guard.source_mount_for(alias/'lab_video'/'rgb.mp4')
        assert resolved == original and mount.snapshot['filesystem'] == 'cifs'
        assert guard.source_mount_for(original)[0] is mount
        assert len(guard.source_mounts) == 1
        with pytest.raises(guard_module.NASUnavailable):
            atomic_json(original.parent/'Forbidden.json', {})
        with pytest.raises(guard_module.NASUnavailable):
            atomic_json(alias/'lab_video'/'AlsoForbidden.json', {})
        atomic_json(volume/'Archive'/'Result.json', {'safe': True})
        descriptor = mount.descriptor
    with pytest.raises(OSError):
        os.fstat(descriptor)
    assert original.read_bytes() == before
    assert not list(original.parent.glob('*.json'))
    assert read_json(volume/'Archive'/'Result.json') == {'safe': True}


@pytest.mark.parametrize('change', ['missing', 'wrong_share', 'local_fallback', 'nested_mount'])
def test_capture_reader_rejects_unverified_source_storage(tmp_path, monkeypatch, change):
    config, _, _, alias, original, mounts = source_storage(tmp_path, monkeypatch)
    snapshot = mounts[original.parent]
    if change == 'missing':
        mounts.pop(original.parent)
    elif change == 'wrong_share':
        mounts[original.parent] = snapshot | {'source': '//synthetic.invalid/other-share'}
    elif change == 'local_fallback':
        mounts[original.parent] = snapshot | {'filesystem': 'ext4'}
    else:
        mounts[original] = snapshot | {'mount_id': 'unapproved-nested-mount', 'source': 'other'}
    with guard_module.storage_guard(config) as guard:
        with pytest.raises(guard_module.NASUnavailable):
            guard.source_mount_for(alias/'lab_video'/'rgb.mp4')


def test_capture_reader_rejects_alias_escape_and_changed_mount(tmp_path, monkeypatch):
    config, volume, _, alias, original, mounts = source_storage(tmp_path, monkeypatch)
    escape = original.parent/'Outside'
    escape.symlink_to(volume, target_is_directory=True)
    with guard_module.storage_guard(config) as guard:
        with pytest.raises(guard_module.NASUnavailable, match='escaped'):
            guard.source_mount_for(alias/'lab_video'/'Outside'/'Fake.mp4')
        guard.source_mount_for(original)
        mounts[original.parent] = mounts[original.parent] | {'mount_id': 'replaced-source-mount'}
        with pytest.raises(guard_module.NASUnavailable, match='disappeared or changed'):
            guard.validate(volume=False)


def test_capture_reader_rejects_retargeted_configured_root(tmp_path, monkeypatch):
    config, volume, _, alias, original, _ = source_storage(tmp_path, monkeypatch)
    with guard_module.storage_guard(config) as guard:
        guard.source_mount_for(original)
        alias.unlink()
        alias.symlink_to(volume, target_is_directory=True)
        with pytest.raises(guard_module.NASUnavailable, match='capture storage alias changed'):
            guard.validate(volume=False)


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
