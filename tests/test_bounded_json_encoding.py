"""Exact historical digest bytes with bounded C-encoder calls, no runtime data."""
import hashlib
import json
import math
import random

import pytest

from visioncortex import device_day_contract as contract


def encoder(**kwargs):
    return json.JSONEncoder(ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                           allow_nan=False, **kwargs)


def assert_same(value, selected=None):
    selected = selected or encoder()
    expected = ''.join(selected.iterencode(value)).encode()
    actual = b''.join(contract._json_blocks(value, selected, block_bytes=1031))
    assert actual == expected
    if selected.sort_keys:
        assert contract.digest(value) == hashlib.sha256(expected).hexdigest()


def test_many_nested_records_preserve_exact_digest_bytes():
    rng = random.Random(4162026)
    strings = ['中文🧪', '"}]\\\n,', '', '\u2028', '\x00\t', 'MetaVideo/rgb.mp4']
    value = {'inputs': [{'frame': index, 'source': strings[index % len(strings)],
                        'box': [rng.uniform(-1e30, 1e30), rng.random(), -0.0, 5e-324],
                        'extra': [None, False, True, 2**100, {}, [], (1, 2)]}
                       for index in range(3000)]}
    assert_same(value)


def test_large_mapping_keeps_sorted_keys_and_key_coercion():
    assert_same({index: {'key': index / 8} for index in range(900, -1, -1)})
    selected = json.JSONEncoder(ensure_ascii=False, sort_keys=False, separators=(',', ':'), allow_nan=False)
    value = {str(i): i for i in range(600)} | {None: 'none', False: 'false', 1.25: 'float'}
    assert_same(value, selected)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_invalid_numbers_still_fail_after_large_prefix(value):
    with pytest.raises(ValueError):
        contract.digest({'prefix': list(range(800)), 'tail': value})


def test_cycles_and_unsupported_keys_still_fail_in_large_containers():
    value = [list(range(700))]
    value.append(value)
    with pytest.raises(ValueError, match='Circular'):
        contract.digest(value)
    value = {str(i): i for i in range(600)}
    value[object()] = 'unsupported'
    with pytest.raises(TypeError):
        contract.digest(value)
    selected = json.JSONEncoder(separators=(',', ':'), skipkeys=True)
    assert_same(value, selected)


def test_c_encoder_never_receives_entire_large_day(monkeypatch):
    original = json.JSONEncoder.encode
    calls = []
    def bounded(self, value):
        result = original(self, value)
        calls.append(len(result.encode()))
        assert len(result.encode()) <= 16384
        return result
    monkeypatch.setattr(json.JSONEncoder, 'encode', bounded)
    value = {'audit': [{'frame': i, 'labels': ['移液'] * 12, 'score': .1}
                       for i in range(8000)]}
    assert_same(value)
    assert len(calls) > 100


def test_pretty_and_custom_encoders_keep_existing_iterencode_path(monkeypatch):
    def unexpected(*_args, **_kwargs):
        pytest.fail('Pretty/custom output must not enter compact subtree estimation')
    monkeypatch.setattr(contract, '_json_subtree_fits', unexpected)
    pretty = json.JSONEncoder(ensure_ascii=False, indent=2, allow_nan=False)
    value = {'audit': [{'frame': i, 'unicode': '实验'} for i in range(900)]}
    assert b''.join(contract._json_blocks(value, pretty)) == pretty.encode(value).encode()
    custom = json.JSONEncoder(separators=(',', ':'), default=lambda _: {'number': math.pi})
    assert b''.join(contract._json_blocks(object(), custom)) == custom.encode(object()).encode()


def test_container_subclasses_preserve_custom_iteration():
    class OddList(list):
        def __iter__(self):
            return iter([1, 2, 3])
        def __getitem__(self, _):
            raise AssertionError('Do not slice a user-defined sequence')
    assert_same({'ordinary': list(range(600)), 'custom': OddList(range(600))})
