"""Owned resident OCR process using only the explicit customer environment."""
import json
import os
import signal
import sys
import threading
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def main():
    from visioncortex.lab.cli import runtime
    from visioncortex.lab.config import Settings
    from visioncortex.lab.live import ReceiverService, load_sources
    from visioncortex.lab.processing import ModelRegistry
    settings=Settings.from_env()
    settings.prepare()
    core=runtime(settings)
    stop=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):
        signal.signal(sig,lambda *_:stop.set())
    status=settings.runtime_root/'receiver-status.json'
    service=None
    try:
        sources=load_sources(core,settings.live_sources_manifest)
        service=ReceiverService(core,settings,sources,ModelRegistry(settings).verified('ocr'))
        service.start()
        while not stop.is_set():
            # Bounded SSD status only; no frame buffers or endpoint secrets are logged.
            snapshot=service.snapshot()
            raw=json.dumps(snapshot,ensure_ascii=False,allow_nan=False).encode()
            if len(raw)>1024**2:
                raise ValueError('Receiver status exceeded its bound')
            temporary=status.with_suffix('.partial')
            temporary.write_bytes(raw)
            os.replace(temporary,status)
            stop.wait(1)
    except Exception as exc:
        status.write_text(json.dumps(dict(status='needs_attention',error_code=type(exc).__name__)))
        raise
    finally:
        if service:
            service.close()


if __name__=='__main__':
    main()
