"""Synthetic recorder metadata validates identity policy, never real video quality."""
import json

import pytest

from test_device_day import capture, device_config  # noqa: F401
from visioncortex.input_availability import configured_record
from visioncortex.nas_recordings import scan_recordings


def recording(config, *, vendor='orbbec', serial='physical-camera-1', configured=None):
    folder = capture(config, camera='renamed_cam02')
    path = folder / 'meta.json'
    metadata = json.loads(path.read_text())
    metadata['camera_key'] = 'renamed_cam02'
    metadata['camera_announce_raw'] = json.dumps({'device': {
        'vendor': vendor, 'serial_number': serial, 'configured_serial_number': configured}})
    path.write_text(json.dumps(metadata))
    config['collection_ingest'].update(settle_seconds=0, camera_hardware_bindings=[{
        'vendor': 'orbbec', 'serial_number': 'physical-camera-1', 'camera_key': 'a_cam01'}])
    return path


def test_renamed_camera_keeps_registered_physical_view_without_changing_identity(device_config):  # noqa: F811
    metadata = recording(device_config)
    before = metadata.read_bytes()
    record, = scan_recordings(device_config)['recordings']
    assert record['configured_role'] == 'first_person'
    assert record['camera_key'] == 'renamed_cam02'
    assert record['camera_binding_status'] == 'configured_hardware_serial'
    assert record['camera_role_binding']['configured_camera_key'] == 'a_cam01'
    assert len(record['camera_role_binding']['metadata_sha256']) == 64
    assert configured_record(device_config, record) == record
    assert metadata.read_bytes() == before
    assert 'renamed_cam02' not in device_config['collection_ingest']['camera_role_map']


@pytest.mark.parametrize('vendor,serial,configured', [
    ('other-vendor', 'physical-camera-1', None),
    ('orbbec', 'another-physical-camera', None),
    ('orbbec', None, 'physical-camera-1'),
    ('orbbec', 'physical-camera-1', 'conflicting-configured-serial'),
])
def test_names_and_configured_serial_cannot_substitute_for_physical_identity(
        device_config, vendor, serial, configured):  # noqa: F811
    recording(device_config, vendor=vendor, serial=serial, configured=configured)
    record, = scan_recordings(device_config)['recordings']
    assert not record.get('configured_role')


def test_ambiguous_physical_registration_does_not_inherit_either_role(device_config):  # noqa: F811
    recording(device_config)
    device_config['collection_ingest']['camera_hardware_bindings'].append({
        'vendor': 'orbbec', 'serial_number': 'physical-camera-1', 'camera_key': 'b_cam02'})
    record, = scan_recordings(device_config)['recordings']
    assert record['camera_binding_status'] == 'hardware_identity_conflict'
    assert not record.get('configured_role')


def test_no_hardware_policy_keeps_existing_metadata_only_call_without_source_access():
    config = {'collection_ingest': {'camera_role_map': {'a_cam01': 'first_person'}}}
    assert configured_record(config, {'camera_key': 'a_cam01'})['configured_role'] == 'first_person'
    assert configured_record(config, {'camera_key': 'unknown'}) == {'camera_key': 'unknown'}
