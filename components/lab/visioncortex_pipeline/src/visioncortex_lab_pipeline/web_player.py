"""Readable player for archived videos; source bytes and indexes stay unchanged."""
import html
from pathlib import Path
from urllib.parse import quote

import anyio
from fastapi.responses import FileResponse

from .report_brand import brand_logo_data_url


class MediaFileResponse(FileResponse):
    """Stop reading NAS media on disconnect and avoid tiny network reads."""
    chunk_size = 1024 * 1024

    async def __call__(self, scope, receive, send):
        class ClientDisconnected(Exception):
            pass

        closed = False
        async with anyio.create_task_group() as tasks:
            async def disconnected():
                nonlocal closed
                while True:
                    if (await receive())['type'] == 'http.disconnect':
                        closed = True
                        return

            async def send_if_connected(message):
                if closed:
                    # Unwind the async file context normally so its awaited
                    # close is not itself cancelled by a task cancellation.
                    raise ClientDisconnected
                await send(message)

            tasks.start_soon(disconnected)
            try:
                await super().__call__(scope, receive, send_if_connected)
            except ClientDisconnected:
                pass
            finally:
                tasks.cancel_scope.cancel()


def render_player(name: str, relative: str) -> str:
    source = f'/api/device-days/{quote(name, safe="")}/files/{quote(relative, safe="/")}'
    web = Path(__file__).with_name('web')
    values = {
        'Title': name,
        'Path': relative,
        'Source': source,
        'Overview': '/archive-overview#Day' + name[:10],
        'Logo': brand_logo_data_url(),
    }
    page = (web / 'clip-player.html').read_text(encoding='utf-8')
    for key, value in values.items():
        page = page.replace('@@' + key + '@@', html.escape(value, quote=True))
    return page.replace('@@Script@@', (web / 'clip-player.js').read_text(encoding='utf-8'))
