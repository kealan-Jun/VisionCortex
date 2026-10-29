"""World-clock support for publication; reported timestamps remain evidence."""
from __future__ import annotations


def source_clock_supported(basis, source_kind):
    if (not isinstance(basis, dict) or not isinstance(basis.get('basis'), str) or not basis['basis']
            or type(basis.get('uncertainty_us')) is not int or basis['uncertainty_us'] != 0
            or basis.get('clock_sync_verified') is False or basis.get('clock_sync_valid') is False):
        return False
    synthetic = source_kind == 'fixture' and basis['basis'] in {'synthetic_fixture_clock', 'fixture_receipt'}
    return synthetic or basis.get('clock_sync_verified') is True or basis.get('clock_sync_valid') is True


def source_clock_sql(media='m', inbox='i'):
    """Fixed internal aliases only; mirrors source_clock_supported JSON types."""
    if media not in {'m', 'em'} or inbox not in {'i', 'ei'}:
        raise ValueError('Only internal clock query aliases are allowed')
    b = f'{media}.time_basis'
    return f'''(jsonb_typeof({b}->'basis')='string' AND {b}->>'basis'<>''
        AND jsonb_typeof({b}->'uncertainty_us')='number' AND {b}->>'uncertainty_us'='0'
        AND COALESCE({b}->'clock_sync_verified','null'::jsonb)<>'false'::jsonb
        AND COALESCE({b}->'clock_sync_valid','null'::jsonb)<>'false'::jsonb
        AND ({b}->'clock_sync_verified'='true'::jsonb OR {b}->'clock_sync_valid'='true'::jsonb
             OR ({inbox}.payload->>'source_kind'='fixture'
                 AND {b}->>'basis' IN ('synthetic_fixture_clock','fixture_receipt'))))'''
