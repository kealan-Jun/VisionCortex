"""Proof renewal cannot rotate credentials, change providers or erase history."""
from copy import deepcopy

import pytest

from test_web_ai_settings import local_settings, selection, verified  # noqa: F401
from visioncortex import ai_settings
from visioncortex.provider_credentials import read_private, read_revision, settings_directory, write_private


def saved(monkeypatch, config):
    monkeypatch.setattr(ai_settings, 'verify_connection', lambda connection, *_: verified(connection))
    ai_settings.verify_and_activate(selection(), 'synthetic-owned-key', config)
    reference = ai_settings.read_index()['active_ref']
    original = read_revision(reference)
    original['verification']['adapter_sha256'] = 'old-adapter'
    write_private(settings_directory() / 'credentials' / f'{reference}.json', original)
    ai_settings.verify_and_activate(selection('zhipu'), 'synthetic-other-provider-key', config)
    return reference, original, ai_settings.read_index()


def test_refresh_keeps_identity_selection_and_private_history(local_settings, monkeypatch):  # noqa: F811
    reference, original, index = saved(monkeypatch, local_settings)
    result = ai_settings.reverify_saved_connection(reference)
    assert result['refreshed'] and not result['credential_identity_changed'] and not result['active_provider_changed']
    current = read_revision(reference)
    assert {k: v for k, v in current.items() if k != 'verification'} == {
        k: v for k, v in original.items() if k != 'verification'}
    assert ai_settings.read_index() == index
    history = list((settings_directory() / 'verification-history').glob('*.json'))
    assert len(history) == 1 and read_private(history[0]) == original['verification']
    assert 'api_key' not in result and 'synthetic-owned-key' not in str(result)


def test_failed_probe_preserves_original_record_and_does_not_store_new_proof(local_settings, monkeypatch):  # noqa: F811
    reference, original, index = saved(monkeypatch, local_settings)
    monkeypatch.setattr(ai_settings, 'verify_connection', lambda *_: {'status': 'failed'})
    assert not ai_settings.reverify_saved_connection(reference)['refreshed']
    assert read_revision(reference) == original and ai_settings.read_index() == index
    assert not (settings_directory() / 'verification-history').exists()


def test_concurrent_credential_change_is_never_overwritten(local_settings, monkeypatch):  # noqa: F811
    reference, original, index = saved(monkeypatch, local_settings)
    changed = deepcopy(original)
    changed['connection']['model'] = 'concurrently-selected-model'
    def race(connection, *_):
        write_private(settings_directory() / 'credentials' / f'{reference}.json', changed)
        return verified(connection)
    monkeypatch.setattr(ai_settings, 'verify_connection', race)
    with pytest.raises(ValueError, match='changed during verification'):
        ai_settings.reverify_saved_connection(reference)
    assert read_revision(reference) == changed and ai_settings.read_index() == index
