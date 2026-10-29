from ._support import research_helpers
import hashlib
import importlib.util
import json
import time

import pytest


PATH = research_helpers() / 'onestreamer_cache_admission.py'
SPEC = importlib.util.spec_from_file_location('one_cache_admission', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture
def owned_cache(tmp_path):
    root = tmp_path/'root'
    media = root/'runtime/memory-long60-20261027/media'
    media.mkdir(parents=True)
    output = tmp_path/'out'
    output.mkdir()
    raw = b'bounded owned cached source bytes'
    identity = hashlib.sha256(raw).hexdigest()
    (media/(identity+'.mp4')).write_bytes(raw)
    lease = media.parent/'CacheLease.json'
    lease.write_text(json.dumps(dict(created_at_unix=time.time()-100,
        expires_at_unix=time.time()+100, max_bytes=8*1024**3)))
    cases = [dict(sources=[dict(sha256=identity, bytes=len(raw))])]
    return root, output, media, lease, cases


def test_new_trial_cleans_verified_old_entries_before_renewal(owned_cache):
    root, out, media, lease, cases = owned_cache
    previous = lease.read_bytes()
    receipt = MODULE.renew_empty_cache(root, out, cases, reader_scan=lambda _: ([], []))
    assert (out/'PreviousCacheLease.json').read_bytes() == previous
    assert not list(media.iterdir())
    assert receipt['removed_files'] == 1
    assert receipt['empty_media_cache_verified'] is True
    value = json.loads(lease.read_text())
    assert value['expires_at_unix']-value['created_at_unix'] == 86400
    assert value['max_bytes'] == 8*1024**3
    assert value['previous_lease_sha256'] == hashlib.sha256(previous).hexdigest()


def test_unknown_entry_preserves_all_cache_bytes(owned_cache):
    root, out, media, lease, cases = owned_cache
    old = lease.read_bytes()
    (media/'foreign.mp4').write_bytes(b'not this evaluation cache')
    before = {p.name: p.read_bytes() for p in media.iterdir()}
    with pytest.raises(RuntimeError, match='Unowned'):
        MODULE.renew_empty_cache(root, out, cases, reader_scan=lambda _: ([], []))
    assert lease.read_bytes() == old
    assert {p.name: p.read_bytes() for p in media.iterdir()} == before


def test_observed_reader_prevents_cleanup_and_new_lease(owned_cache):
    root, out, media, lease, cases = owned_cache
    old = lease.read_bytes()
    with pytest.raises(RuntimeError, match='still has readers'):
        MODULE.renew_empty_cache(root, out, cases,
            reader_scan=lambda _: ([dict(pid=42, cached_file='owned.mp4')], []))
    assert len(list(media.iterdir())) == 1
    assert lease.read_bytes() == old
    assert not (out/'CacheAdmissionCleanup.json').exists()


def test_expired_old_entries_cannot_be_hidden_by_new_lease(owned_cache):
    root, out, media, lease, cases = owned_cache
    value = json.loads(lease.read_text())
    value['expires_at_unix'] = time.time()-1
    lease.write_text(json.dumps(value))
    old = lease.read_bytes()
    with pytest.raises(RuntimeError, match='original 24-hour deadline'):
        MODULE.renew_empty_cache(root, out, cases, reader_scan=lambda _: ([], []))
    assert lease.read_bytes() == old
    assert len(list(media.iterdir())) == 1


def test_cache_source_tamper_is_preserved_for_inspection(owned_cache):
    root, out, media, lease, cases = owned_cache
    next(media.iterdir()).write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='Changed cache'):
        MODULE.renew_empty_cache(root, out, cases, reader_scan=lambda _: ([], []))
    assert next(media.iterdir()).read_bytes() == b'changed'
