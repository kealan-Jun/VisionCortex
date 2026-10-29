"""Bounded localhost transports and media preparation for the timeline runner."""
from __future__ import annotations

import base64
from contextlib import contextmanager
from decimal import Decimal
import fcntl
from fractions import Fraction
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from PIL import Image

from .domain import Conflict, fingerprint
from .media import Media, hash_file
from .subprocesses import run_bounded
from .timeline import seconds


class ProviderUnavailable(RuntimeError):
    pass


def execution_pin(model):
    """Optional serving-configuration pin; old manifests remain readable."""
    if 'execution_identity' not in model:
        return None
    value = model['execution_identity']
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ProviderUnavailable('Local execution identity requires an exact SHA256')
    return value


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ProviderUnavailable('Local inference redirects are not allowed')


class LocalProviders:
    def __init__(self, settings):
        self.settings = settings

    def load(self):
        path = self.settings.timeline_providers_path
        if path is None:
            return None
        if path.is_symlink() or path.stat().st_size > 128 * 1024:
            raise ProviderUnavailable('Invalid local provider manifest')
        doc = json.loads(path.read_text())
        if doc.get('schema_version') != 'vc-lab-local-providers/1' or doc.get('customer_code') != self.settings.customer_code:
            raise ProviderUnavailable('Explicit customer local provider manifest required')
        if doc.get('enabled') is not True:
            return None
        models = doc.get('models', {})
        if not models or set(models) - {'transcription', 'ocr', 'video_understanding'}:
            raise ProviderUnavailable('Unsupported local provider kind')
        for kind, model in models.items():
            pin = execution_pin(model)
            if pin is not None and kind == 'video_understanding':
                raise ProviderUnavailable('Media-infer execution identity does not attest a VLM server')
            if model.get('panel_profiles'):
                spec=model['panel_profiles']
                if kind!='ocr' or not isinstance(spec,dict) or len(spec.get('sha256',''))!=64 or not spec.get('path'):
                    raise ProviderUnavailable('OCR panel profiles need an explicit pinned local manifest')
                profile_path=Path(spec['path'])
                if profile_path.is_symlink() or not profile_path.resolve().is_relative_to((self.settings.runtime_root/'models').resolve()):
                    raise ProviderUnavailable('Active OCR templates must use this project local model root')
            endpoint = urlsplit(model.get('base_url', ''))
            if endpoint.scheme != 'http' or endpoint.hostname not in {'127.0.0.1', '::1'} or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment or endpoint.path not in {'', '/'} or not endpoint.port:
                raise ProviderUnavailable('Only explicit loopback HTTP endpoints without credentials are allowed')
            if not model.get('model_revision') or not model.get('token_env', '').startswith('VC_LAB_'):
                raise ProviderUnavailable('Pinned model revision and private token environment reference required')
            if kind != 'video_understanding' and (len(model.get('runtime_revision', '')) != 64 or model.get('protocol') != 'vc-lab-media-infer/1'):
                raise ProviderUnavailable('ASR/OCR require a pinned serving script revision')
            if kind == 'video_understanding' and (model.get('protocol') != 'local-vllm-chat/1' or not model.get('served_model') or model.get('video_policy') != 'all_frames'):
                raise ProviderUnavailable('VLM needs an explicit serving ID and all-frame policy')
            if kind == 'video_understanding' and model.get('decoder_configuration_verified') is not True:
                raise ProviderUnavailable('All-frame server decoding must be explicitly verified before enabling')
            complete=kind=='video_understanding' and model.get('video_input_mode')=='complete_retained_clip'
            if complete:
                geometry=model.get('vision_token_geometry',{})
                if model.get('spatial_policy')!='preserve_source_pixels' or model.get('processor_resize_verified_disabled') is not True:
                    raise ProviderUnavailable('Complete-clip providers must explicitly preserve source pixels')
                if geometry.get('schema_version')!='qwen-vl-patch-grid/1' or any(type(geometry.get(n)) is not int or not 1<=geometry[n]<=64 for n in ['patch_size','merge_size','temporal_patch_size']):
                    raise ProviderUnavailable('Audited checkpoint vision-token geometry required')
                if type(model.get('max_model_len')) is not int or not 4096<=model['max_model_len']<=1048576:
                    raise ProviderUnavailable('Explicit complete-clip context capacity required')
            for name, default, lo, hi in [('timeout_seconds', 300, 1, 1800), ('max_payload_bytes', 16*1024**2, 1024, (64 if complete else 24)*1024**2), ('video_max_edge', 768, 128, 4096), ('max_tokens', 2048, 128, 8192)]:
                value = model.get(name, default)
                if type(value) is not int or not lo <= value <= hi:
                    raise ProviderUnavailable('Invalid provider resource bound')
        if doc.get('data_scope') not in {'fixture', 'development', 'customer'}:
            raise ProviderUnavailable('Explicit provider data scope required')
        return doc


class LocalHTTP:
    def __init__(self, model):
        self.model = model
        self.execution_identity = execution_pin(model)
        token = os.environ.get(model['token_env'])
        if not token:
            raise ProviderUnavailable('Local service token is unavailable')
        self._token = token
        self.opener = build_opener(ProxyHandler({}), NoRedirect())

    def call(self, path, body=None):
        raw = json.dumps(body, ensure_ascii=False, allow_nan=False).encode() if body is not None else None
        if raw is not None and len(raw) > 96 * 1024**2:
            raise ValueError('Local request exceeds memory bound')
        request = Request(self.model['base_url'].rstrip('/') + path, data=raw,
                          headers={'Authorization': 'Bearer ' + self._token, 'Content-Type': 'application/json'})
        started = time.monotonic()
        with self.opener.open(request, timeout=self.model.get('timeout_seconds', 300)) as response:
            data = response.read(2 * 1024**2 + 1)
            if len(data) > 2 * 1024**2:
                raise ValueError('Local response exceeds memory bound')
        result = json.loads(data)
        if not isinstance(result, dict):
            raise ValueError('Structured local response required')
        return result, dict(protocol=self.model['protocol'], model_revision=self.model['model_revision'],
                            request_sha256=hashlib.sha256(raw or b'').hexdigest(),
                            response_sha256=hashlib.sha256(data).hexdigest(), elapsed_seconds=time.monotonic()-started,
                            accuracy='NOT_PROVEN')

    def preflight(self, kind):
        if kind == 'video_understanding':
            result, _ = self.call('/v1/models')
            if self.model['served_model'] not in {x.get('id') for x in result.get('data', [])}:
                raise ProviderUnavailable('VLM serving identity differs from the manifest')
        else:
            result, _ = self.call('/health')
            if result.get('status') != 'ready' or result.get('model_revision') != self.model['model_revision'] or result.get('kind') != ('asr' if kind == 'transcription' else 'ocr'):
                raise ProviderUnavailable('Local service health/revision mismatch')
            if self.execution_identity is not None and result.get('execution_identity') != self.execution_identity:
                raise ProviderUnavailable('Local service execution settings differ from the manifest')

    def infer(self, raw, source_kind):
        digest = hashlib.sha256(raw).hexdigest()
        result, receipt = self.call('/v1/infer', dict(media_base64=base64.b64encode(raw).decode(), source_kind=source_kind))
        if result.get('source_sha256') != digest or result.get('model_revision') != self.model['model_revision'] or result.get('runtime_revision') != self.model['runtime_revision'] or result.get('actual_model_invocation') is not True or type(result.get('cache_hit')) is not bool:
            raise Conflict('Local service execution receipt differs from the requested bytes/model/runtime')
        if self.execution_identity is not None and result.get('execution_identity') != self.execution_identity:
            raise Conflict('Local service execution settings changed or differ from the manifest')
        receipt.update(source_sha256=digest, runtime_revision=result['runtime_revision'],
                       execution_identity=result.get('execution_identity'),
                       serving_configuration_pinned=self.execution_identity is not None,
                       execution='cached_model_response' if result['cache_hit'] else 'model_response',
                       model_weights_verification='serving_script_checkpoint_manifest')
        return result, receipt


class PreparedMedia:
    """One serialized, bounded SSD working set. Raw originals are never rewritten."""
    def __init__(self, core, settings):
        self.settings, self.media = settings, Media(core, settings)

    @contextmanager
    def workspace(self):
        root = self.settings.runtime_root / 'timeline-work'
        root.mkdir(exist_ok=True)
        if root.is_symlink():
            raise ValueError('Working directory cannot be a symbolic link')
        with (root / '.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for path in root.iterdir():
                if path.is_symlink():
                    raise ValueError('Unexpected link in owned working directory')
                if path.is_dir() and path.name.startswith('job-') and time.time() - path.stat().st_mtime > 86400:
                    shutil.rmtree(path)
            used = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
            if used + 64*1024**2 > 256 * 1024**2 or shutil.disk_usage(root).free < 80 * 1024**3:
                raise Conflict('Bounded timeline workspace/free-space guard reached')
            with tempfile.TemporaryDirectory(prefix='job-', dir=root) as folder:
                yield Path(folder)

    def prepare(self, ref, kind, model, folder, *, box=None):
        prepared_started=time.monotonic()
        located = self.media.locate(ref['media_id'])
        if located['status'] != 'available' or located['version'] != ref['media_version']:
            raise Conflict('Model source media unavailable or superseded')
        path = located['path']
        before = hash_file(path, self.settings.max_native_bytes)
        if before[0] != ref['sha256']:
            raise Conflict('Model source differs from the immutable request')
        source_verified=time.monotonic()
        complete=kind=='video_understanding' and ref.get('input_unit')=='complete_retained_clip'
        bound = min(model.get('max_payload_bytes', 16*1024**2), (64 if complete else 12 if kind == 'ocr' else 24)*1024**2)
        if complete:
            from .native_video import prepare_complete
            raw,transform=prepare_complete(path,ref,model,folder,bound)
            mime='video/mp4'
        elif model.get('video_input_mode')=='complete_retained_clip' and kind=='video_understanding':
            raise ProviderUnavailable('Native provider cannot silently fall back to a sliced/resized input')
        elif kind == 'ocr':
            with Image.open(path) as im:
                if im.width * im.height > 16_000_000:
                    raise ValueError('OCR source exceeds pixel bound')
                im = im.convert('RGB')
                if box is not None:
                    x1, y1, x2, y2 = [float(x) for x in box]
                    im = im.crop((math.floor(x1*im.width), math.floor(y1*im.height), math.ceil(x2*im.width), math.ceil(y2*im.height)))
                stream = io.BytesIO()
                im.save(stream, format='PNG')
                raw, mime = stream.getvalue(), 'image/png'
            transform = dict(kind='explicit_field_crop' if box else 'full_frame', box=box)
        else:
            start, end = seconds(ref['start_seconds']), seconds(ref['end_seconds'])
            if not start < end or end - start > 180:
                raise ValueError('Provider window must be positive and at most 180 seconds')
            output = folder / ('input-' + fingerprint(dict(ref=ref, kind=kind)) + ('.mp4' if kind == 'video_understanding' else '.wav'))
            # Accurate input seeking decodes the keyframe preroll, rather than
            # repeatedly decoding from media zero for each later window.
            command = ['ffmpeg', '-v', 'error', '-nostdin', '-y', '-ss', str(start), '-accurate_seek', '-i', str(path), '-t', str(end-start)]
            if kind == 'transcription':
                from .audio_source import decimal_time, probe_audio
                coverage = probe_audio(path, start, end)
                sample_range = coverage['actual_sample_range']
                # Preserve decoded native PTS through the bounded seek, then
                # trim exact source support before rebasing the model input.
                command = ['ffmpeg', '-v', 'error', '-nostdin', '-y', '-copyts',
                           '-ss', decimal_time(max(0, start-2)), '-seek_timestamp', '1', '-accurate_seek', '-i', str(path)]
                command += ['-map', '0:a:0', '-vn', '-af',
                            f"atrim=start={decimal_time(sample_range['start_seconds'])}:end={decimal_time(sample_range['end_seconds'])},asetpts=PTS-STARTPTS",
                            '-t', decimal_time(Fraction(coverage['prepared_duration_seconds'])),
                            '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le']
                mime, transform = 'audio/wav', dict(kind='audio_decode', rate=16000, channels=1)
            else:
                edge = model.get('video_max_edge', 768)
                command += ['-map', '0:v:0', '-an', '-vf', f"scale=w='min(iw,{edge})':h='min(ih,{edge})':force_original_aspect_ratio=decrease:force_divisible_by=2,setpts=PTS-STARTPTS", '-vsync', '0', '-c:v', 'libx264', '-crf', '18', '-preset', 'fast']
                mime, transform = 'video/mp4', dict(kind='video_decode', temporal_sampling=False, max_edge=edge, spatial_resize_and_lossy_encoding=True,
                                                    server_decode_policy=model['video_policy'], server_decoder_verification='deployment_configuration')
            command += ['-fs', str(bound), str(output)]
            try:
                run_bounded(command, '', env={'PATH': os.environ.get('PATH', '/usr/bin:/bin')}, timeout=180, max_bytes=128*1024)
                if output.stat().st_size >= bound - 1024:
                    raise ValueError('Derived media reached its bound; refusing a truncated input')
                if kind == 'transcription':
                    from .audio_source import verify_prepared_wave
                    transform.update(verify_prepared_wave(output, coverage))
                else:
                    probe = json.loads(run_bounded(['ffprobe', '-v', 'error', '-show_entries', 'format=duration:stream=avg_frame_rate,nb_frames', '-of', 'json', str(output)], '',
                                                   env={'PATH': os.environ.get('PATH', '/usr/bin:/bin')}, timeout=30, max_bytes=32*1024))
                    decoded_duration = seconds(probe['format']['duration'])
                    frame_rate = probe['streams'][0].get('avg_frame_rate', '0/1')
                    fps = float(Fraction(frame_rate)) if frame_rate != '0/0' else 0
                    tolerance = Decimal(str(min(1, 1.1/fps) if fps > 0 else .1))
                    if abs(decoded_duration - (end - start)) > tolerance:
                        raise ValueError('Decoded media does not cover the requested interval')
                    transform.update(decoded_duration_seconds=str(decoded_duration), decoded_frame_count=probe['streams'][0].get('nb_frames'),
                                     duration_tolerance_seconds=str(tolerance))
                raw = output.read_bytes()
            finally:
                output.unlink(missing_ok=True)
        prepared_finished=time.monotonic()
        if not 0 < len(raw) < bound or before != hash_file(path, self.settings.max_native_bytes):
            raise Conflict('Derived input exceeds bounds or its source changed')
        return raw, mime, dict(source=ref, derived_sha256=hashlib.sha256(raw).hexdigest(), derived_bytes=len(raw), transform=transform,
            timing=dict(source_locate_and_verify_seconds=source_verified-prepared_started,
                prepare_seconds=prepared_finished-source_verified,post_source_verify_seconds=time.monotonic()-prepared_finished,
                total_seconds=time.monotonic()-prepared_started))
