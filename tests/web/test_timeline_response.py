"""Large timeline conversion must not occupy the ASGI progress loop."""
import threading

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from visioncortex import device_day_timeline


def test_timeline_encodes_on_worker_and_preserves_json(tmp_path, monkeypatch):
    day = '2026-10-10'
    payload = {'date': day, 'entries': [{'frame_id': '帧1', 'start_us': 1,
                                      'values': [None, False, 1.25]}],
               'cross_view_links': [], 'index_age_seconds': 3.5,
               'evidence_status': 'PARTIAL_EVIDENCE'}
    monkeypatch.setattr(device_day_timeline, 'query_timeline', lambda *_: payload)
    threads = {}
    original = JSONResponse.render

    def render(response, content):
        if isinstance(content, dict) and content.get('date') == day:
            threads['encoding'] = threading.get_ident()
        return original(response, content)

    monkeypatch.setattr(JSONResponse, 'render', render)
    app = FastAPI()

    @app.middleware('http')
    async def observe_loop(request, call_next):
        threads['asgi'] = threading.get_ident()
        return await call_next(request)

    device_day_timeline.install_routes(
        app, lambda: {'storage': {'local_runtime_root': str(tmp_path)}})
    with TestClient(app) as client:
        response = client.get('/api/day-timeline/' + day)
    assert response.status_code == 200
    assert response.headers['content-type'] == 'application/json'
    assert response.json() == payload
    assert threads['encoding'] != threads['asgi']
