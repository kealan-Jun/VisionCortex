"""Authenticated loopback-only ASR/OCR inference with pinned local checkpoints."""
import argparse
import base64
import hashlib
import hmac
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tempfile
import threading
import time

_REMOTE_FS = {'cifs', 'smb3', 'nfs', 'nfs4', 'fuse.sshfs'}


def local_path(value, *, directory=False, owned=False):
    """Require an existing canonical local path before reading private state."""
    path = Path(value)
    if (not path.is_absolute() or path in {Path('/mnt'), Path('/media')}
            or str(path).startswith(('/mnt/', '/media/', '//'))
            or path.is_symlink() or path.resolve() != path):
        raise ValueError('Explicit canonical local path required')
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError('Existing local directory/file required')
    if owned and path.stat().st_uid != os.getuid():
        raise ValueError('Runtime/private configuration owner differs from current user')
    result = subprocess.run(['findmnt', '-J', '-T', str(path), '-o', 'FSTYPE'],
                            capture_output=True, text=True, check=True, timeout=10)
    mounts = json.loads(result.stdout).get('filesystems', [])
    if not mounts or any(m.get('fstype') in _REMOTE_FS for m in mounts):
        raise ValueError('Runtime/checkpoint paths must use local storage')
    return path


def runtime_root(value):
    return local_path(value, directory=True, owned=True)


def load_configuration(path):
    path = local_path(path, owned=True)
    if path.stat().st_size > 1024**2:
        raise ValueError('Private service configuration exceeds size bound')
    config = json.loads(path.read_text())
    if config.get('schema_version') != 'vc-lab-model-services/1':
        raise ValueError('Explicit versioned private service configuration required')
    return config


def read_token(config):
    # Provisioning a credential is a separate operation; this code never creates one.
    path = local_path(config['token_file'], owned=True)
    if stat.S_IMODE(path.stat().st_mode) & 0o077 or path.stat().st_size > 4096:
        raise ValueError('Private service credential permissions/size are unsafe')
    token = path.read_text().strip()
    if not 32 <= len(token) <= 4096 or any(ord(c) < 33 or ord(c) > 126 for c in token):
        raise ValueError('Existing private service token is required')
    return token


def private_directory(root, relative):
    path = root / relative
    if not path.resolve().is_relative_to(root) or path.resolve() != path:
        raise ValueError('Runtime directory escaped its owned root')
    path.mkdir(parents=True, exist_ok=True)
    return local_path(path, directory=True, owned=True)


def checkpoint(root, name, config, verified=None):
    """Reuse existing weights in place; verify the complete private manifest."""
    entry = config.get('checkpoints', {}).get(name)
    if not isinstance(entry, dict) or 'manifest' not in entry:
        raise ValueError('Explicit checkpoint manifest required: ' + name)
    manifest_path = local_path(entry['manifest'], owned=True)
    if manifest_path.stat().st_size > 8 * 1024**2:
        raise ValueError('Checkpoint manifest exceeds bound')
    raw = manifest_path.read_bytes()
    if entry.get('manifest_sha256') is not None and hashlib.sha256(raw).hexdigest() != entry['manifest_sha256']:
        raise ValueError('Checkpoint manifest fingerprint mismatch')
    manifest = json.loads(raw)
    source = local_path(manifest['model_root'], directory=True)
    files = manifest.get('files')
    if not isinstance(manifest.get('revision'), str) or not manifest['revision'] or not isinstance(files, dict) or not 1 <= len(files) <= 100000:
        raise ValueError('Checkpoint revision and complete bounded file inventory required')
    actual = set()
    for item in source.rglob('*'):
        if item.is_symlink():
            raise ValueError('Checkpoint symlinks are not accepted')
        if item.is_file():
            actual.add(item.relative_to(source).as_posix())
    if actual != set(files):
        raise ValueError('Checkpoint inventory differs from complete local files')
    for relative, expected in files.items():
        logical = PurePosixPath(relative)
        if (not isinstance(relative, str) or not relative or logical.is_absolute()
                or logical.as_posix() != relative or any(part in {'.', '..'} for part in logical.parts)
                or '\\' in relative or '\x00' in relative or not isinstance(expected, dict)
                or type(expected.get('bytes')) is not int or expected['bytes'] < 0
                or not re.fullmatch('[0-9a-f]{64}', expected.get('sha256', ''))):
            raise ValueError('Invalid checkpoint path or SHA/byte identity')
        path = source / relative
        if path.resolve() != path or not path.resolve().is_relative_to(source) or not path.is_file():
            raise ValueError('Invalid checkpoint path')
        before = path.stat()
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(4 * 1024**2), b''):
                digest.update(block)
        after = path.stat()
        def identity(value):
            return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
        if (identity(before) != identity(after) or after.st_size != expected['bytes']
                or digest.hexdigest() != expected['sha256']):
            raise ValueError('Local checkpoint failed integrity verification')
    if verified is not None:
        verified[name] = dict(manifest_sha256=hashlib.sha256(raw).hexdigest(),
                              files_sha256=hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest())
    return source, manifest['revision']


def service_settings(config, name):
    value = config.get('services', {}).get(name, {})
    if not isinstance(value, dict):
        raise ValueError('Invalid private service settings')
    return value


def inference_identity(kind, config, verified):
    """Hash effective model settings and the manifests actually verified at load."""
    settings = service_settings(config, kind)
    if kind == 'ocr':
        det = settings.get('detection_model_name', 'PP-OCRv5_mobile_det')
        rec = settings.get('recognition_model_name', 'PP-OCRv5_mobile_rec')
        selected = dict(detection_model_name=det, recognition_model_name=rec,
                        device=settings.get('device', 'gpu:0'))
        names = [settings.get('detection_checkpoint', det), settings.get('recognition_checkpoint', rec)]
    elif kind == 'asr':
        selected = dict(dtype='bfloat16', device_map='cuda:0', attention='sdpa',
                        max_inference_batch_size=1, max_new_tokens=1024)
        names = [settings.get('checkpoint', 'Qwen3-ASR-1.7B'),
                 settings.get('aligner_checkpoint', 'Qwen3-ForcedAligner-0.6B')]
    else:
        raise ValueError('Unknown inference kind')
    selected['cuda_visible_devices'] = settings.get('cuda_visible_devices', '0')
    identities = [verified[name] for name in names]
    document = dict(kind=kind, effective_settings=selected, checkpoints=identities)
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def make_asr(root, config, verified=None):
    import torch
    from qwen_asr import Qwen3ASRModel
    settings = service_settings(config, 'asr')
    source, revision = checkpoint(root, settings.get('checkpoint', 'Qwen3-ASR-1.7B'), config, verified)
    aligner, aligner_revision = checkpoint(root, settings.get('aligner_checkpoint', 'Qwen3-ForcedAligner-0.6B'), config, verified)
    model = Qwen3ASRModel.from_pretrained(str(source), dtype=torch.bfloat16, device_map='cuda:0',
        attn_implementation='sdpa', max_inference_batch_size=1, max_new_tokens=1024,
        forced_aligner=str(aligner), forced_aligner_kwargs={'dtype':torch.bfloat16, 'device_map':'cuda:0'})
    def infer(raw):
        import soundfile as sf
        try:
            samples, rate = sf.read(io.BytesIO(raw), dtype='float32')
        except (sf.LibsndfileError, ValueError) as exc:
            raise ValueError('Unsupported or invalid recording') from exc
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        duration = len(samples)/rate
        if not 0 < duration <= 180:
            raise ValueError('ASR requests must contain at most 180 seconds of audio')
        torch.cuda.reset_peak_memory_stats()
        reply = model.transcribe(audio=(samples, rate), language='Chinese', return_time_stamps=True)[0]
        words = [dict(text=w.text, start_seconds=float(w.start_time), end_seconds=float(w.end_time)) for w in (reply.time_stamps or [])]
        if any(not 0 <= w['start_seconds'] <= w['end_seconds'] <= duration+.5 for w in words):
            raise ValueError('ASR alignment escaped the source audio interval')
        segments = [dict(text=reply.text, start_seconds=words[0]['start_seconds'], end_seconds=words[-1]['end_seconds'])] if words else []
        return dict(text=reply.text, language=reply.language, duration_seconds=duration, words=words, segments=segments,
            time_basis='local_forced_alignment', alignment_revision=aligner_revision,
            peak_torch_gpu_bytes=torch.cuda.max_memory_allocated(), accuracy='NOT_PROVEN')
    return infer, revision


def make_ocr(root, config, verified=None):
    import numpy as np
    from PIL import Image
    from paddleocr import PaddleOCR
    settings = service_settings(config, 'ocr')
    det_name = settings.get('detection_model_name', 'PP-OCRv5_mobile_det')
    rec_name = settings.get('recognition_model_name', 'PP-OCRv5_mobile_rec')
    det, det_revision = checkpoint(root, settings.get('detection_checkpoint', det_name), config, verified)
    rec, rec_revision = checkpoint(root, settings.get('recognition_checkpoint', rec_name), config, verified)
    model = PaddleOCR(device=settings.get('device', 'gpu:0'), cpu_threads=2, enable_mkldnn=False,
        text_detection_model_name=det_name, text_recognition_model_name=rec_name,
        text_detection_model_dir=str(det), text_recognition_model_dir=str(rec),
        use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)
    def infer(raw):
        try:
            with Image.open(io.BytesIO(raw)) as image:
                if image.width*image.height > 16_000_000:
                    raise ValueError('OCR image exceeds pixel bound')
                pixels = np.asarray(image.convert('RGB'))[:, :, ::-1]
        except (OSError, Image.DecompressionBombError) as exc:
            raise ValueError('Unsupported or invalid image') from exc
        results = []
        for result in model.predict(pixels):
            item = {}
            for key in ['rec_texts', 'rec_scores', 'rec_polys', 'rec_boxes']:
                value = result.get(key, [])
                if key == 'rec_texts':
                    item[key] = [str(v) for v in value]
                elif key == 'rec_scores':
                    item[key] = [float(v) for v in value]
                else:
                    item[key] = np.asarray(value).tolist()
            results.append(item)
        return dict(results=results, recognition_revision=rec_revision, accuracy='NOT_PROVEN', physical_action_confirmed=False)
    return infer, det_revision+':'+rec_revision


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--kind', choices=['asr', 'ocr'], required=True)
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args(argv)
    root = runtime_root(args.runtime_root)
    config = load_configuration(args.config)
    token = read_token(config)
    if not 1 <= args.port <= 65535:
        parser.error('port must be between 1 and 65535')
    private_directory(root, 'runtime/cache/huggingface')
    private_directory(root, 'runtime/cache/paddlex')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
        HF_HOME=str(root/'runtime/cache/huggingface'), PADDLE_PDX_CACHE_HOME=str(root/'runtime/cache/paddlex'),
        PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK='True', FLAGS_allocator_strategy='auto_growth',
        OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    from starlette.concurrency import run_in_threadpool
    import uvicorn
    verified = {}
    infer, revision = make_asr(root, config, verified) if args.kind == 'asr' else make_ocr(root, config, verified)
    execution_identity = inference_identity(args.kind, config, verified)
    runtime_revision = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    admission = threading.Lock()
    receipts = private_directory(root, f'runtime/receipts/{args.kind}')
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    @app.middleware('http')
    async def bounds(request: Request, call_next):
        if request.url.path != '/health':
            if not hmac.compare_digest(request.headers.get('authorization', ''), 'Bearer '+token):
                return JSONResponse({'detail':'Unauthorized'}, status_code=401)
            try:
                declared_size = int(request.headers.get('content-length', '0'))
            except ValueError:
                return JSONResponse({'detail':'Invalid request length'}, status_code=400)
            if declared_size < 0 or declared_size > 32*1024**2:
                return JSONResponse({'detail':'Request exceeds size bound'}, status_code=413)
        return await call_next(request)
    @app.get('/health')
    def health():
        return dict(status='ready', kind=args.kind, model_revision=revision, execution_identity=execution_identity, local_only=True)
    @app.post('/v1/infer')
    async def execute(request: Request):
        if not admission.acquire(blocking=False):
            raise HTTPException(429, 'Inference slot is busy; retry later')
        try:
            # Bound chunked requests while receiving them, before buffering.
            encoded = bytearray()
            async for block in request.stream():
                if len(encoded)+len(block) > 32*1024**2:
                    raise HTTPException(413, 'Request exceeds size bound')
                encoded.extend(block)
            payload = json.loads(encoded)
            if not isinstance(payload, dict) or not isinstance(payload.get('media_base64'), str):
                raise ValueError('media_base64 must be a string')
            raw = base64.b64decode(payload['media_base64'], validate=True)
            if not 0 < len(raw) <= (24 if args.kind == 'asr' else 12)*1024**2:
                raise ValueError('Media exceeds size bound')
            image_hash = hashlib.sha256(raw).hexdigest()
            key = hashlib.sha256((revision+':'+runtime_revision+':'+execution_identity+':'+image_hash).encode()).hexdigest()
            path = receipts/(key+'.json')
            if path.is_file():
                return dict(json.loads(path.read_text()), cache_hit=True)
            started = time.monotonic()
            inferred = await run_in_threadpool(infer, raw)
            result = dict(inferred, source_sha256=image_hash, model_revision=revision, runtime_revision=runtime_revision,
                execution_identity=execution_identity,
                elapsed_seconds=time.monotonic()-started, actual_model_invocation=True, cache_hit=False,
                source_kind=payload.get('source_kind', 'explicit_request'))
            output = json.dumps(result, ensure_ascii=False, allow_nan=False)
            if len(output.encode()) > 2*1024**2:
                raise ValueError('Inference result exceeds receipt bound')
            used = sum(p.stat().st_size for p in receipts.glob('*.json'))
            if used+len(output.encode()) > 128*1024**2:
                raise HTTPException(503, 'Receipt capacity reached; archive results before admitting more')
            # Requests are content-addressed and serialized; a crash may leave
            # a partial file but cannot publish an incomplete readable receipt.
            with tempfile.NamedTemporaryFile('w', dir=receipts, prefix='.', delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(output+'\n')
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
            return result
        except (ValueError, KeyError) as exc:
            raise HTTPException(422, str(exc)[:200]) from exc
        finally:
            admission.release()
    uvicorn.run(app, host='127.0.0.1', port=args.port, access_log=False)


if __name__ == '__main__':
    main()
