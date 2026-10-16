"""A sealed reference wins over cached stat metadata, never over file bytes."""
import hashlib

import pytest

from visioncortex import device_day_inputs as inputs


@pytest.mark.parametrize('matches_seal', [True, False])
def test_conflicting_cached_hash_forces_stable_bytes_to_be_read(tmp_path, monkeypatch, matches_seal):
    path = tmp_path/'captured-video.bin'
    old, sealed, different = b'old bytes', b'new bytes', b'bad bytes'
    path.write_bytes(old)
    original_identity = inputs.identity(path)
    # Simulate a NAS publisher preserving stat metadata while finalizing bytes.
    monkeypatch.setattr(inputs, 'identity', lambda p: original_identity)
    assert inputs.hash_content(path) == hashlib.sha256(old).hexdigest()
    path.write_bytes(sealed if matches_seal else different)
    reference = {'size_bytes': len(sealed), 'sha256': hashlib.sha256(sealed).hexdigest()}
    assert inputs.verify_content(path, reference) is matches_seal
    assert inputs.hash_content(path) == hashlib.sha256(path.read_bytes()).hexdigest()
    assert reference['sha256'] == hashlib.sha256(sealed).hexdigest()
