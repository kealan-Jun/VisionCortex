"""Official HF JoyAI video serving with native frames and auditable prompt input.

Imports do not load torch, weights, start a listener or provision credentials.
The caller owns local checkpoint verification and supplies the existing token.
"""
import base64
import hashlib
import hmac
import io
import inspect
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time
import traceback


class CapacityError(ValueError):
    pass


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def validate_model_inputs(data, model):
    """Preserve processor tensors; reject unexplained runtime API mismatches."""
    parameters = inspect.signature(model.forward).parameters
    unsupported = sorted(set(data) - set(parameters))
    if unsupported:
        raise ValueError('Processor/model input keys require explicit compatibility: ' + ','.join(unsupported))
    return data


def settings(raw):
    if (raw.get('engine') != 'joyai-hf' or raw.get('device') != 'cuda:0'
            or raw.get('do_resize') is not False or raw.get('do_sample_frames') is not False
            or raw.get('attention') != 'flash_attention_2' or raw.get('cache_implementation') != 'offloaded'
            or type(raw.get('served_model_name')) is not str or not raw['served_model_name']):
        raise ValueError('Explicit JoyAI native-frame CUDA/FA2/offloaded configuration required')
    bounds = dict(max_frames=(2, 240), max_duration_seconds=(1, 10),
        max_media_bytes=(1024, 64*1024**2), max_request_bytes=(2048, 96*1024**2),
        max_decoded_bytes=(1024, 2*1024**3), max_frame_pixels=(1024, 16_000_000),
        max_model_len=(4096, 262144), max_new_tokens=(1, 512), request_seconds=(10, 900),
        gpu_memory_limit_bytes=(18*1024**3, 42*1024**3), gpu_reserve_bytes=(4*1024**3, 8*1024**3))
    result = {key: raw[key] for key in ('engine', 'device', 'do_resize', 'do_sample_frames',
        'attention', 'cache_implementation', 'served_model_name')}
    for key, (low, high) in bounds.items():
        value = raw.get(key)
        if type(value) is not int or not low <= value <= high:
            raise ValueError('Explicit bounded JoyAI setting required: ' + key)
        result[key] = value
    return result


def runtime_identity(raw):
    return dict(settings(raw), helper_source_sha256=sha(Path(__file__).read_bytes()),
        processor='official_AutoProcessor_no_resize_no_sampling_verified_grid',
        model_class='Qwen3VLForConditionalGeneration', dtype='bfloat16',
        source_pixel_policy='all_decoded_RGB_frames_right_bottom_zero_padding_only')


def parse_request(body, config):
    if (not isinstance(body, dict) or body.get('model') != config['served_model_name']
            or body.get('temperature', 0) != 0 or body.get('n', 1) != 1
            or body.get('stream', False) is not False
            or body.get('response_format') != {'type': 'json_object'}):
        raise ValueError('One deterministic nonstreaming JSON video completion required')
    count = body.get('max_tokens')
    if type(count) is not int or not 1 <= count <= config['max_new_tokens']:
        raise CapacityError('Requested output tokens exceed explicit bound')
    messages = body.get('messages')
    if (not isinstance(messages, list) or len(messages) != 1
            or not isinstance(messages[0], dict) or messages[0].get('role') != 'user'):
        raise ValueError('One explicit video/context user message required')
    parts = messages[0].get('content')
    if not isinstance(parts, list) or not 2 <= len(parts) <= 16:
        raise ValueError('Bounded ordered video/text parts required')
    prepared, texts, videos = [], [], []
    for part in parts:
        if not isinstance(part, dict):
            raise ValueError('Structured video/text parts required')
        if part.get('type') == 'text' and set(part) == {'type', 'text'}:
            text = part['text']
            if not isinstance(text, str) or len(text.encode()) > 64*1024:
                raise CapacityError('Prompt text exceeds bound')
            prepared.append(dict(type='text', text=text))
            texts.append(text)
        elif part.get('type') == 'video_url' and set(part) == {'type', 'video_url'}:
            url = part['video_url'].get('url') if isinstance(part['video_url'], dict) else None
            prefix = 'data:video/mp4;base64,'
            if not isinstance(url, str) or not url.startswith(prefix):
                raise ValueError('Only inline original MP4 bytes are accepted; no remote fetch')
            encoded = url[len(prefix):]
            if len(encoded) > math.ceil(config['max_media_bytes']/3)*4:
                raise CapacityError('Encoded original video exceeds bound')
            raw = base64.b64decode(encoded, validate=True)
            if not 0 < len(raw) <= config['max_media_bytes']:
                raise CapacityError('Original video exceeds bound')
            videos.append(raw)
            prepared.append(dict(type='video'))
        else:
            raise ValueError('Unsupported content part; nothing may be silently dropped')
    if len(videos) != 1 or not texts:
        raise ValueError('One complete video plus its full context text required')
    template = body.get('chat_template_kwargs', {})
    if not isinstance(template, dict) or set(template) - {'enable_thinking'} or template.get('enable_thinking', False) is not False:
        raise ValueError('Only explicitly disabled thinking is supported')
    return [dict(role='user', content=prepared)], videos[0], texts, count


def decode_video(raw, config):
    import av
    import numpy as np
    frames, proof, total = [], [], 0
    with av.open(io.BytesIO(raw), mode='r') as container:
        if len(container.streams.video) != 1:
            raise ValueError('Exactly one video stream required')
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 0)
        if not 0 < fps <= 120:
            raise ValueError('Original video has no supported explicit frame rate')
        origin = None
        for index, frame in enumerate(container.decode(video=0)):
            if index >= config['max_frames'] or frame.width*frame.height > config['max_frame_pixels']:
                raise CapacityError('Original decoded frame/pixel budget exceeded; no sampling fallback')
            if frame.pts is None or frame.time_base is None:
                raise ValueError('Original PTS/time base required for every frame')
            at = float(frame.pts*frame.time_base)
            origin = at if origin is None else origin
            if abs((at-origin)-index/fps) > .0001:
                raise ValueError('Variable original PTS cannot be silently replaced by nominal FPS')
            pixels = frame.to_ndarray(format='rgb24')
            pad_h, pad_w = (-frame.height) % 32, (-frame.width) % 32
            padded = np.pad(pixels, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=0)
            total += padded.nbytes
            if total*2 > config['max_decoded_bytes']:
                raise CapacityError('Decoded RGB RAM budget exceeded')
            if frames and padded.shape != frames[0].shape:
                raise ValueError('Original dimensions changed inside the clip')
            if not np.array_equal(padded[:frame.height, :frame.width], pixels):
                raise ValueError('Spatial padding changed source RGB pixels')
            frames.append(padded)
            proof.append(dict(index=index, pts=frame.pts, time_base=str(frame.time_base),
                relative_seconds=at-origin, width=frame.width, height=frame.height,
                RGB_sha256=sha(pixels.tobytes()), padding_right_bottom=[pad_w, pad_h]))
    if not frames or len(frames) % 2:
        raise ValueError('Nonempty even native frame count required; no implicit temporal duplication')
    if len(frames)/fps > config['max_duration_seconds']+.000001:
        raise CapacityError('Complete video duration exceeds configured bound; no recut fallback')
    return np.stack(frames), fps, proof


def make_backend(source, raw_settings):
    import torch
    import transformers
    import av
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from transformers.video_utils import VideoMetadata
    config = settings(raw_settings)
    free, total = torch.cuda.mem_get_info()
    cap = min(config['gpu_memory_limit_bytes'], free-config['gpu_reserve_bytes'])
    if cap < 18*1024**3:
        raise CapacityError('Insufficient free GPU memory with the explicit reserve')
    torch.cuda.set_per_process_memory_fraction(cap/total)
    started = time.monotonic()
    processor = AutoProcessor.from_pretrained(str(source), local_files_only=True)
    vp = processor.video_processor
    if (vp.patch_size, vp.merge_size, vp.temporal_patch_size) != (16, 2, 2):
        raise ValueError('JoyAI processor geometry differs from the audited native path')
    model = Qwen3VLForConditionalGeneration.from_pretrained(str(source), local_files_only=True,
        dtype=torch.bfloat16, device_map={'': 'cuda:0'}, attn_implementation='flash_attention_2').eval()
    torch.cuda.synchronize()
    loaded = dict(model_load_seconds=time.monotonic()-started, gpu_allocator_limit_bytes=cap,
        initial_free_bytes=free, gpu_total_bytes=total, torch_version=torch.__version__,
        transformers_version=transformers.__version__, av_version=av.__version__,
        model_class=type(model).__name__, processor_class=type(processor).__name__)

    def infer(body, receipt):
        messages, raw, texts, maximum = parse_request(body, config)
        receipt.update(stage='decode', source_sha256=sha(raw), encoded_source_bytes=len(raw),
            input_texts=texts, input_texts_sha256=[sha(t.encode()) for t in texts],
            request_sha256=sha(canonical(body).encode()), loaded=loaded)
        tick = time.monotonic()
        pixels, fps, frames = decode_video(raw, config)
        receipt.update(decode_seconds=time.monotonic()-tick, source_frames=frames,
            decoded_frame_count=len(frames), original_pixels_preserved=True,
            temporal_sampling=False, spatial_resize=False, stage='processor')
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
        if not all(t in text for t in texts):
            raise ValueError('Official template dropped or changed supplied context text')
        receipt.update(rendered_prompt=text, rendered_prompt_sha256=sha(text.encode()),
                       full_body_text_preserved_in_template=True)
        tick = time.monotonic()
        data = processor(text=[text], videos=[pixels], videos_kwargs=dict(
            video_metadata=[VideoMetadata(total_num_frames=len(pixels), fps=fps, duration=len(pixels)/fps,
                frames_indices=list(range(len(pixels))))], do_sample_frames=False, do_resize=False), return_tensors='pt')
        data = validate_model_inputs(data, model)
        receipt['processor_model_input_keys'] = sorted(data)
        grid = data['video_grid_thw'][0].tolist()
        tokens = int(data['input_ids'].shape[-1])
        if grid[0]*2 != len(frames) or [grid[2]*16, grid[1]*16] != [pixels.shape[2], pixels.shape[1]]:
            raise ValueError('Official processor resized or sampled the original video')
        if tokens+maximum > min(config['max_model_len'], model.config.text_config.max_position_embeddings):
            raise CapacityError('Complete native video and full text exceed the model context')
        receipt.update(processor_seconds=time.monotonic()-tick, processor_grid=grid,
            input_token_count=tokens, input_ids_sha256=sha(data['input_ids'].numpy().tobytes()),
            processor_frame_count_preserved=True, processor_size_preserved=True, stage='generation')
        del pixels
        if torch.cuda.mem_get_info()[0] < config['gpu_reserve_bytes']:
            raise CapacityError('GPU free reserve exhausted before generation')
        torch.cuda.reset_peak_memory_stats()
        tick = time.monotonic()
        data = {k: (v.to(device='cuda:0', dtype=torch.bfloat16) if k.startswith('pixel_values')
                    else v.to('cuda:0')) for k, v in data.items()}
        hooks = []
        receipt.update(model_forward_entries=0, vision_forward_entries=0)

        def before_model(module, args, kwargs):
            receipt['model_forward_entries'] += 1
            if receipt['model_forward_entries'] == 1:
                ids = kwargs.get('input_ids')
                if ids is not None:
                    receipt['model_consumed_input_ids_sha256'] = sha(ids.detach().cpu().numpy().tobytes())

        def before_vision(module, args, kwargs):
            receipt['vision_forward_entries'] += 1
            receipt['model_consumed_video_grid'] = kwargs['grid_thw'].tolist() if 'grid_thw' in kwargs else None

        hooks.append(model.register_forward_pre_hook(before_model, with_kwargs=True))
        hooks.append(model.model.visual.register_forward_pre_hook(before_vision, with_kwargs=True))
        try:
            with torch.inference_mode():
                result = model.generate(**data, max_new_tokens=maximum, do_sample=False, use_cache=True,
                    cache_implementation='offloaded', logits_to_keep=1)
            torch.cuda.synchronize()
            new = result[:, tokens:]
            answer = processor.batch_decode(new, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            receipt.update(generation_seconds=time.monotonic()-tick, output_tokens=int(new.shape[-1]),
                model_consumed_full_prompt=receipt.get('model_consumed_input_ids_sha256') == receipt['input_ids_sha256'],
                output_text=answer, stage='completed', accuracy='NOT_PROVEN')
            return answer, int(new.shape[-1]), 'length' if new.shape[-1] >= maximum else 'stop'
        except Exception:
            receipt['generation_failure_traceback'] = traceback.format_exc(limit=24)[-16000:]
            raise
        finally:
            for hook in hooks:
                hook.remove()
            receipt.update(torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                           torch_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    return infer, loaded


def application(infer, config, token, revision, execution_identity, receipts, loaded):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    from starlette.concurrency import run_in_threadpool
    import asyncio
    limits = settings(config)
    lock = threading.Lock()
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    def save(path, value):
        raw = (canonical(dict(value))+'\n').encode()
        if len(raw) > 2*1024**2 or sum(p.stat().st_size for p in receipts.glob('*.json'))+len(raw) > 128*1024**2:
            raise CapacityError('Bounded VLM receipt storage exhausted')
        with tempfile.NamedTemporaryFile('wb', dir=receipts, prefix='.', delete=False) as f:
            temporary = Path(f.name)
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        temporary.replace(path)

    @app.middleware('http')
    async def authenticate(request: Request, next_handler):
        if request.url.path != '/health' and not hmac.compare_digest(request.headers.get('authorization', ''), 'Bearer '+token):
            return JSONResponse({'detail': 'Unauthorized'}, status_code=401)
        return await next_handler(request)

    @app.get('/health')
    def health():
        return dict(status='ready', kind='vlm', engine='joyai-hf', model_revision=revision,
                    execution_identity=execution_identity, local_only=True, loaded=loaded)

    @app.get('/v1/models')
    def models():
        return dict(object='list', data=[dict(id=limits['served_model_name'], object='model',
            model_revision=revision, execution_identity=execution_identity)])

    @app.post('/v1/chat/completions')
    async def execute(request: Request):
        if not lock.acquire(blocking=False):
            raise HTTPException(429, 'Owned video inference slot is busy')
        timer = None
        path, receipt = None, None
        try:
            encoded = bytearray()
            async with asyncio.timeout(30):
                async for block in request.stream():
                    if len(encoded)+len(block) > limits['max_request_bytes']:
                        raise HTTPException(413, 'Bounded video request exceeded')
                    encoded.extend(block)
            body = json.loads(encoded)
            parse_request(body, limits)  # Reject bad input before GPU invocation.
            key = sha((revision+execution_identity).encode()+encoded)
            path = receipts/(key+'.json')
            if path.is_file():
                saved = json.loads(path.read_text())
                if saved.get('status') == 'completed':
                    return dict(saved['response'], vc_lab_cache_hit=True)
                raise HTTPException(409, 'Previous bounded attempt failed; explicit new request/revision required')
            receipt = dict(schema_version='vc-lab-joyai-native-execution/1', status='running',
                model_revision=revision, execution_identity=execution_identity,
                raw_http_body_sha256=sha(encoded), pid=os.getpid(), started_unix=time.time(),
                actual_model_invocation=False)
            save(path, receipt)

            def deadline():
                receipt.update(status='deadline_exceeded', deadline_seconds=limits['request_seconds'])
                try:
                    save(path, receipt)
                finally:
                    os._exit(124)  # Only this owned service; no lingering CUDA worker.
            timer = threading.Timer(limits['request_seconds'], deadline)
            timer.daemon = True
            timer.start()
            started = time.monotonic()
            answer, count, finish = await run_in_threadpool(infer, body, receipt)
            response = dict(id='joyai-'+key[:24], object='chat.completion', model=limits['served_model_name'],
                choices=[dict(index=0, message=dict(role='assistant', content=answer), finish_reason=finish)],
                usage=dict(prompt_tokens=receipt.get('input_token_count'), completion_tokens=count),
                vc_lab_execution=dict(model_revision=revision, execution_identity=execution_identity,
                    request_sha256=receipt['raw_http_body_sha256'], receipt_key=key,
                    model_consumed_full_prompt=receipt.get('model_consumed_full_prompt', False)))
            receipt.update(status='completed', elapsed_seconds=time.monotonic()-started,
                actual_model_invocation=receipt.get('model_forward_entries', 0) > 0, response=response)
            save(path, receipt)
            return response
        except HTTPException:
            raise
        except (ValueError, KeyError, TimeoutError) as exc:
            if receipt is not None:
                receipt.update(status='failed', error_code=type(exc).__name__, error=str(exc)[:500],
                    actual_model_invocation=receipt.get('model_forward_entries', 0) > 0)
                save(path, receipt)
            raise HTTPException(422, str(exc)[:200]) from exc
        except Exception as exc:
            if receipt is not None:
                receipt.update(status='failed', error_code=type(exc).__name__, error=str(exc)[:500],
                    actual_model_invocation=receipt.get('model_forward_entries', 0) > 0)
                save(path, receipt)
            raise HTTPException(503, 'Owned model inference failed; preserved receipt') from exc
        finally:
            if timer:
                timer.cancel()
            lock.release()
    return app
