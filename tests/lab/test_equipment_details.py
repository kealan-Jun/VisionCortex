"""Scoped read-only QR details, using only generated fixture identities."""
from contextlib import contextmanager
from dataclasses import replace
import json
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest

from visioncortex.lab import cli
from visioncortex.lab.api import create_app
from visioncortex.lab.domain import Core
from visioncortex.lab.equipment import equipment_details


def at(hour):
    return f'2026-10-29T{hour:02d}:00:00+08:00'


@pytest.fixture
def reserved(core):
    device = str(uuid4())
    core.reserve_equipment_catalogue('generated-catalogue', dict(
        schema_version='vc-lab-equipment-catalogue/1', source={'sha256': 'a'*64},
        equipment=[dict(device_id=device, qr_payload=device, device_code='EQ-000001',
                        name='Generated fixture instrument', source_location='Fixture room', ordinal=1)]))
    return device


@pytest.fixture
def retired(core, reserved):
    core.activate_equipment('fixture-activation', reserved, verified_at=at(10), actor='synthetic fixture',
        evidence={'fixture_only': True}, metadata={'serial_number': 'SYNTHETIC-SERIAL', 'current_location': 'Fixture bench'})
    core.retire_equipment('fixture-retirement', reserved, at=at(12), actor='synthetic fixture', reason='generated lifecycle')
    return reserved


@pytest.fixture
def client(core, settings):
    with TestClient(create_app(core, settings)) as result:
        yield result


def auth(settings):
    return {'Authorization': 'Bearer '+settings.token}


def snapshot(core):
    tables = ['equipment_identity', 'entity', 'tag_assignment', 'command_receipt', 'outbox',
              'operation', 'quantity_ledger_entry', 'hj_operator_scan_event', 'hj_experiment_record']
    with core.db.transaction(readonly=True) as conn:
        return {table: conn.execute('SELECT to_jsonb(t) AS row FROM '+table+
                    ' t WHERE customer_id=%s ORDER BY to_jsonb(t)::text', (core.customer,)).fetchall() for table in tables}


def test_reserved_QR_returns_registered_details_without_physical_activation(client, settings, reserved):
    response = client.get('/internal/equipment/'+reserved, headers=auth(settings))
    assert response.status_code == 200
    doc = response.json()
    assert doc['permanent_uuid'] == doc['qr']['payload'] == reserved
    assert doc['device_code'] == 'EQ-000001' and doc['name'] == 'Generated fixture instrument'
    assert doc['registered_metadata']['source_location'] == 'Fixture room'
    assert 'serial_number' not in doc['registered_metadata']
    assert doc['current_state'] == 'reserved' and doc['state_as_of'] == 'not_activated'
    assert doc['instrument_entity_id'] is doc['label_entity_id'] is None
    assert doc['label_assignment_as_of']['target_id'] is None
    assert doc['qr']['payload_format'] == 'bare_uuid' and not doc['qr']['reprint_required']
    assert not doc['physical_presence_or_operation_proven'] and not doc['activation_performed']


@pytest.mark.parametrize('hour,state,bound', [(9, 'not_activated', False), (10, 'active', True),
                                            (11, 'active', True), (12, 'retired', False)])
def test_as_of_uses_effective_half_open_lifecycle_not_current_retired_state(client, settings, retired, hour, state, bound):
    response = client.get('/internal/equipment/'+retired, headers=auth(settings), params={'as_of': at(hour)})
    assert response.status_code == 200
    doc = response.json()
    assert doc['current_state'] == 'retired' and doc['state_as_of'] == state
    assert doc['as_of_mode'] == 'explicit'
    assert doc['instrument_entity_id'] == retired and doc['label_entity_id'] != retired
    assert doc['label_assignment_as_of']['target_id'] == (retired if bound else None)
    assert doc['label_assignment_as_of']['association_status'] == ('explicit' if bound else 'needs_review')
    assert doc['registered_metadata']['serial_number'] == 'SYNTHETIC-SERIAL'
    assert doc['metadata_time_basis'] == 'current_registered_snapshot_not_historical_as_of'
    assert not doc['historical_metadata_available'] and doc['reserved_at'] is None
    assert 'not_recorded' in doc['reservation_time_status']


def test_existing_authentication_protects_device_details(client, settings, reserved):
    path = '/internal/equipment/'+reserved
    assert client.get(path).status_code == 401
    assert client.get(path, headers={'Authorization': 'Bearer wrong-fixture-token'}).status_code == 401
    assert client.get(path, headers=auth(settings)).status_code == 200


@pytest.mark.parametrize('foreign', [True, False])
def test_unknown_or_other_customer_UUID_does_not_leak_details(core, client, settings, foreign):
    if foreign:
        other = Core(core.db, Core.customer_instance(core.db, 'other-equipment-'+str(uuid4()), scope='fixture')['id'])
        value = other.issue_equipment('foreign', metadata={'name': 'FOREIGN-PRIVATE'}, actor='fixture')['device_id']
    else:
        value = uuid4()
    response = client.get('/internal/equipment/'+str(value), headers=auth(settings), params={'customer_id': 'ignored'})
    assert response.status_code == 404 and 'FOREIGN-PRIVATE' not in response.text


@pytest.mark.parametrize('value,when', [('not-a-uuid', None), ('EQ-000001', None), (None, '2026-10-29T10:00:00')])
def test_details_reject_malformed_UUID_or_naive_historical_time(client, settings, reserved, value, when):
    response = client.get('/internal/equipment/'+(value or reserved), headers=auth(settings),
                          params={'as_of': when} if when else {})
    assert response.status_code == 422


def test_lookup_is_one_readonly_snapshot_and_leaves_every_business_row_identical(core, client, settings, retired, monkeypatch):
    before = snapshot(core)
    original, calls = core.db.transaction, []
    with monkeypatch.context() as patch:
        @contextmanager
        def transaction(*, readonly=False):
            calls.append(readonly)
            with original(readonly=readonly) as conn:
                yield conn
        patch.setattr(core.db, 'transaction', transaction)
        for hour in [9, 11, 12]:
            response = client.get('/internal/equipment/'+retired, headers=auth(settings), params={'as_of': at(hour)})
            assert response.status_code == 200 and response.json()['read_only']
        assert calls == [True]*3
    assert snapshot(core) == before


def test_CLI_queries_the_same_core_scope_without_creating_commands(core, settings, retired, monkeypatch, capsys):
    with core.db.transaction(readonly=True) as conn:
        code = conn.execute('SELECT code FROM customer_instance WHERE id=%s', (core.customer,)).fetchone()['code']
    monkeypatch.setattr(cli.Settings, 'from_env', classmethod(lambda cls: replace(settings, customer_code=code)))
    before = snapshot(core)
    cli.main(['equipment-details', retired, '--as-of', at(11)])
    actual = json.loads(capsys.readouterr().out)
    assert actual == equipment_details(core, retired, as_of=at(11))
    assert actual['customer_id'] == str(core.customer) and actual['state_as_of'] == 'active'
    assert snapshot(core) == before
