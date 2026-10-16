import pytest

from visioncortex.receipt_projection import configured_limit


def test_site_limit_keeps_default_and_explicit_bounded_opt_in(monkeypatch):
    name = 'VISIONCORTEX_RECEIPT_PROJECTION_MAX_BYTES'
    monkeypatch.delenv(name, raising=False)
    assert configured_limit(name, 64, 256) == 64
    monkeypatch.setenv(name, '256')
    assert configured_limit(name, 64, 256) == 256


@pytest.mark.parametrize('value', ['', 'unlimited', '-1', '0', '257', '1.5'])
def test_unbounded_or_invalid_site_limit_fails_closed(monkeypatch, value):
    name = 'VISIONCORTEX_RECEIPT_PROJECTION_MAX_BYTES'
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        configured_limit(name, 64, 256)
