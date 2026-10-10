import threading

import anyio
from fastapi import FastAPI
from fastapi.testclient import TestClient

from visioncortex.device_day_contract import atomic_json
from visioncortex.device_day_service import DeviceDayService, install_routes
from visioncortex.web_player import MediaFileResponse, render_player


def test_player_does_not_replace_video_bytes_or_allow_unowned_paths(tmp_path):
    root = tmp_path / 'archive'
    name = '2026-09-14_camera_cam01'
    atomic_json(root / name / 'ProcessedClips/Index.json', {'archive': name})
    source = root / name / 'MetaVideo/10-00.mp4'
    source.parent.mkdir()
    source.write_bytes(b'synthetic-structure-test-not-real-media')
    cfg = {'storage': {'archive_root': str(root), 'local_runtime_root': str(tmp_path / 'runtime')}}
    app = FastAPI()
    install_routes(app, lambda: cfg, DeviceDayService(lambda: cfg, threading.Lock()))
    client = TestClient(app)
    page = client.get(f'/device-days/{name}/watch/MetaVideo/10-00.mp4')
    assert page.status_code == 200
    assert 'text/html' in page.headers['content-type']
    assert f'/api/device-days/{name}/files/MetaVideo/10-00.mp4' in page.text
    assert 'drawImage(video' in page.text and 'requestVideoFrameCallback' in page.text
    media = client.get(f'/api/device-days/{name}/files/MetaVideo/10-00.mp4', headers={'Range': 'bytes=0-8'})
    assert media.status_code == 206 and media.content == b'synthetic'
    direct = client.get(f'/api/device-days/{name}/files/MetaVideo/10-00.mp4',
                        headers={'Sec-Fetch-Dest': 'document', 'Sec-Fetch-Mode': 'navigate'}, follow_redirects=False)
    assert direct.status_code == 307
    assert direct.headers['location'] == f'/device-days/{name}/watch/MetaVideo/10-00.mp4'
    embedded = client.get(f'/api/device-days/{name}/files/MetaVideo/10-00.mp4',
                          headers={'Sec-Fetch-Dest': 'video', 'Sec-Fetch-Mode': 'cors', 'Range': 'bytes=0-8'})
    assert embedded.status_code == 206 and embedded.content == b'synthetic'
    assert source.read_bytes() == b'synthetic-structure-test-not-real-media'
    assert client.get(f'/device-days/{name}/watch/ProcessedClips/Index.json').status_code == 404
    assert client.get(f'/device-days/{name}/watch/MetaVideo/missing.mp4').status_code == 404
    outside = tmp_path / 'outside.mp4'
    outside.write_bytes(b'private')
    (source.parent / 'Link.mp4').symlink_to(outside)
    assert client.get(f'/device-days/{name}/watch/MetaVideo/Link.mp4').status_code == 404


def test_player_escapes_source_and_keeps_identity():
    page = render_player('2026-09-14_camera_cam01', 'ProcessedClips/Clips/One/evil"<script>.mp4')
    assert 'evil"<script>' not in page
    assert 'evil%22%3Cscript%3E.mp4' in page
    assert '@@' not in page
    assert '2026-09-14_camera_cam01' in page


def test_closed_player_stops_transfer_before_reading_the_entire_file(tmp_path):
    source = tmp_path / 'video.mp4'
    source.write_bytes(b'x' * (5 * MediaFileResponse.chunk_size))

    async def check():
        closed = anyio.Event()
        bodies = []

        async def receive():
            await closed.wait()
            return {'type': 'http.disconnect'}

        async def send(message):
            if message['type'] == 'http.response.body':
                bodies.append(len(message.get('body', b'')))
                closed.set()
                await anyio.sleep(0)

        await MediaFileResponse(source)({'type': 'http', 'method': 'GET', 'headers': []}, receive, send)
        assert bodies and sum(bodies) < source.stat().st_size

    anyio.run(check)
