"""Receiver compatibility must preserve source identity, bytes and job count."""
from dataclasses import replace
import hashlib
import json
import os
import time

import pytest
import qrcode

from visioncortex.lab.components import enable_components
from visioncortex.lab.domain import Conflict
from visioncortex.lab.ingest import Ingest
from visioncortex.lab.input_layout import photo_location
from visioncortex.lab.processing import Worker
from visioncortex.lab.projections import Projections
from visioncortex.lab.retention import Retention


def photo_paths(camera):
    name = '20261023_100000_1.png'
    return [
        f'{camera}/2026-10-23/10-00-00/{name}',
        f'2026-10-23_{camera}/MetaData/10-00-00/{name}',
        f'2026-10-23_{camera}/MetaData/Photos/10-00-00/{name}',
        f'voice_photos/{camera}/2026-10-23/10-00-00/{name}',
    ]


@pytest.mark.parametrize('relative', [
    '../camera/2026-10-23/10-00-00/20261023_100000_1.png',
    '/camera/2026-10-23/10-00-00/20261023_100000_1.png',
    'camera//2026-10-23/10-00-00/20261023_100000_1.png',
    '2026-10-23_camera/Comment/OCR/10-00-00/20261023_100000_1.png',
    '2026-10-24_camera/MetaData/Photos/10-00-00/20261023_100000_1.png',
    '2026-10-23_camera/MetaData/Photos/10-00-01/20261023_100000_1.png',
    'camera/2026-10-23/10-00-00/20261023_100000_1.png\\suffix',
    f'2026-10-23_{"c"*129}/MetaData/Photos/10-00-00/20261023_100000_1.png',
])
def test_path_aliases_do_not_admit_invalid_times_devices_or_result_paths(relative):
    with pytest.raises(ValueError):
        photo_location(relative)


@pytest.mark.parametrize('camera', ['lab.cam_'+('x'*105), 'voice_photos'])
def test_four_photo_layouts_reuse_one_capture_and_publish_without_rewriting_sources(core, settings, camera):
    settings = replace(settings, photo_batch_mode='directory_triplet', photo_cameras=(camera,))
    paths = photo_paths(camera)
    assert {photo_location(p).identity for p in paths} == {paths[0]}
    enable_components()
    from field_recognition.capture.photo_sources import photo_location as original_parser
    from field_recognition.runtime import source_io
    # Binding in the short-lived reader cannot alter the original parent module.
    assert source_io.photo_location is original_parser
    with pytest.raises(ValueError):
        original_parser(paths[2])
    first = settings.incoming/paths[0]
    first.parent.mkdir(parents=True)
    qrcode.make('input-layout-fixture').save(first)
    raw = first.read_bytes()
    snapshots = {}
    ingest = Ingest(core, settings)
    for relative in paths:
        path = settings.incoming/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        os.utime(path, (time.time()-2, time.time()-2))
        snapshots[path] = (path.read_bytes(), path.stat().st_mtime_ns)
        result = ingest.discover_once()
        assert result['processed'] == 1 and not result['errors'], result
    assert ingest.discover_once()['skipped'] == 4
    with core.db.transaction(readonly=True) as conn:
        for table, count in [('capture', 1), ('media_asset', 1), ('inbox', 1), ('capture_session', 1), ('processing_job', 2)]:
            assert conn.execute(f'SELECT count(*) AS n FROM {table} WHERE customer_id=%s', (core.customer,)).fetchone()['n'] == count
        locations = conn.execute("SELECT relative_path FROM media_location WHERE customer_id=%s AND storage='incoming'", (core.customer,)).fetchall()
        assert {row['relative_path'] for row in locations} == set(paths)
        initial = conn.execute('SELECT payload FROM inbox WHERE customer_id=%s', (core.customer,)).fetchone()['payload']
        assert initial['capture_id'] == paths[0] and initial['relative_path'] == paths[0]
        assert initial['camera_id'] == camera and initial['group_expected_photos'] == 3
    worker = Worker(core, settings)
    assert {worker.run_once()['status'], worker.run_once()['status']} == {'completed', 'blocked_model'}
    for _ in range(4):
        assert Retention(core, settings).tick().get('status') != 'needs_attention'
    evidence = list(settings.incoming.glob(f'2026-10-23_{camera}/Comment/OCR/VoicePhotoReadings/*/Evidence.json'))
    assert len(evidence) == 1
    document = json.loads(evidence[0].read_text())
    assert len(document['sources']) == 1 and document['sources'][0]['path'] == paths[0]
    assert document['measurement_files'] == []
    assert not list(settings.incoming.rglob('Result*.json'))
    index = Projections(core)
    while index.tick()['indexed']:
        pass
    assert len(index.search('input-layout-fixture')['items']) == 1
    for path, snapshot in snapshots.items():
        assert (path.read_bytes(), path.stat().st_mtime_ns) == snapshot
        assert not (path.parent/'PhotoReceipt.json').exists()
    assert source_io.photo_location is original_parser


def test_path_aliases_cannot_hide_changed_capture_metadata_or_bytes(core, settings, photo):
    # Hash is intentionally absent in this producer receipt; the receiver's
    # verified digest must still protect the original media identity.
    receipt = {k: v for k, v in photo.items() if k != 'sha256'}
    ingest = Ingest(core, settings)
    captured = ingest.accept(receipt)
    alias = settings.incoming/'alias.png'
    original = settings.incoming/photo['relative_path']
    alias.write_bytes(original.read_bytes())
    moved = dict(receipt, relative_path='alias.png')
    assert ingest.accept(moved) == dict(capture_id=captured['capture_id'], duplicate=True)
    with pytest.raises(Conflict, match='content/metadata'):
        ingest.accept(dict(moved, captured_at='2026-10-23T11:00:00+08:00'))
    with pytest.raises(Conflict, match='new capture identity'):
        ingest.accept(dict(moved, capture_id='different-identity'))
    alias.write_bytes(b'changed alias fixture')
    with pytest.raises(Conflict, match='different media bytes'):
        ingest.accept(moved)
    original.write_bytes(b'changed original fixture')
    with pytest.raises(Conflict, match='different media bytes'):
        ingest.accept(receipt)
    with core.db.transaction(readonly=True) as conn:
        assert conn.execute('SELECT count(*) AS n FROM capture WHERE customer_id=%s', (core.customer,)).fetchone()['n'] == 1
        assert conn.execute('SELECT count(*) AS n FROM processing_job WHERE customer_id=%s', (core.customer,)).fetchone()['n'] == 2


@pytest.mark.parametrize('mode', ['receipt', 'directory_triplet', 'capture_receipt'])
def test_ancestor_upload_marker_blocks_intake_and_completion_wakes_retry(core, settings, mode):
    relative = photo_paths('camera-one')[2]
    path = settings.incoming/relative
    path.parent.mkdir(parents=True)
    qrcode.make('upload-marker-fixture').save(path)
    os.utime(path, (time.time()-2, time.time()-2))
    if mode == 'directory_triplet':
        settings = replace(settings, photo_batch_mode=mode, photo_cameras=('camera-one',))
    elif mode == 'receipt':
        value = dict(schema_version='field-photo-receipt/1', camera_id='camera-one', source_kind='fixture',
            measurement={'expected_photos': 1, 'burst_id': 'upload-marker-fixture'},
            photos=[dict(filename=path.name, capture_id='upload-marker-fixture', captured_at='2026-10-23T10:00:00+08:00', sha256=hashlib.sha256(path.read_bytes()).hexdigest())])
        (path.parent/'PhotoReceipt.json').write_text(json.dumps(value))
    else:
        value = dict(capture_id='upload-marker-fixture', source_instance='camera-one', source_revision='1', kind='photo',
            relative_path=relative, captured_at='2026-10-23T10:00:00+08:00', time_basis={'basis': 'fixture'}, source_kind='fixture')
        (path.parent/'CaptureReceipt.json').write_text(json.dumps(value))
    marker = settings.incoming/'2026-10-23_camera-one/.uploading'
    marker.write_text('explicit fixture upload in progress')
    ingest = Ingest(core, settings)
    blocked = ingest.discover_once()
    assert blocked['processed'] == 0 and len(blocked['errors']) == 1
    assert not ingest.discover_once()['errors']  # unchanged blocked input backs off
    with core.db.transaction(readonly=True) as conn:
        assert conn.execute('SELECT count(*) AS n FROM processing_job WHERE customer_id=%s', (core.customer,)).fetchone()['n'] == 0
    marker.unlink()
    # No 30-second wait: only the ancestor state has changed.
    accepted = ingest.discover_once()
    assert accepted['processed'] == 1 and not accepted['errors'], accepted
    assert ingest.discover_once()['skipped'] == 1
