"""Source-bound playback maps; clock gaps and unknown offsets stay unmapped."""
from __future__ import annotations

import math
import json
import hashlib
import os
import subprocess
from pathlib import Path

from . import speech_worker
from .alignment import read_timestamp_csv
from .input_seal import verify_input_seal
from .schemas import AlignmentTransform, RunManifest, VideoInfo
from .storage import _bounded_source_fingerprint

CONTROL = 'JSON-Config-Files/speech_timeline.json'


def monotonic(points: list[list[float]]) -> bool:
    return len(points) >= 2 and all(math.isfinite(value) for row in points for value in row) and all(
        b[0] > a[0] and b[1] > a[1] for a, b in zip(points, points[1:], strict=False))


def audio_map(source: dict, chunk: dict, part, info, transform, virtual_start: float) -> dict:
    start, end = chunk['start_seconds'], chunk['end_seconds']
    if source.get('alignment_basis') == 'recorder_shared_clock':
        clock = part.timestamps_csv
        if not clock or speech_worker.sha256(clock) != source.get('video_clock_sha256'):
            raise ValueError('录音时钟与已验证的转写不一致')
        request = source['_request']
        epoch = request['source']['start_global_us'] / 1000
        points = read_timestamp_csv(clock, info.fps)
        if speech_worker.sha256(clock) != source['video_clock_sha256']:
            raise ValueError('录音时钟在读取期间变化')
        anchors = [[(point.source_ms - epoch) / 1000 - start,
                    transform.to_global(virtual_start + point.local_ms)]
                   for point in points if point.source_ms is not None and point.clock_sync_valid is not False]
        # Preserve bracketing anchors; playback itself stays within chunk bounds.
        anchors = [row for i, row in enumerate(anchors)
                   if (i == 0 or anchors[i-1][0] <= end-start) and (i == len(anchors)-1 or anchors[i+1][0] >= 0)]
        gap = 2.0
    elif source.get('audio_offset_ms') is not None:
        anchors = [[t-start, transform.to_global(virtual_start + source['audio_offset_ms'] + t*1000)] for t in (start, end)]
        gap = None
    else:
        return {'anchors': [], 'state': 'offset_unknown', 'max_gap_seconds': None}
    if not monotonic(anchors):
        return {'anchors': [], 'state': 'clock_unavailable', 'max_gap_seconds': gap}
    return {'anchors': anchors, 'state': 'mapped', 'max_gap_seconds': gap}



def make_preview(root: Path, source: dict, duration: float) -> dict:
    """A derived browser proxy: resized video only, with unchanged time scale."""
    identity = {"source":source, "duration_seconds":duration, "width":960, "height":540,
                "fps":15, "crf":28, "codec":"libx264", "schema":"visioncortex-speech-preview/1"}
    key = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24]
    directory = root / "Key-Materials/Experiment-Audio/Video-Previews" / key
    target, receipt_path = directory / "preview.mp4", directory / "receipt.json"
    if receipt_path.is_file():
        receipt = speech_worker.read_json(receipt_path)
        if receipt.get("identity") == identity and receipt.get("preview") == speech_worker.file_record(target):
            return {"path":str(target.relative_to(root)), **receipt["preview"], "cache_reused":True}
    directory.mkdir(parents=True,exist_ok=True)
    temporary = target.with_name("preview.partial.mp4")
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-threads", "2",
               "-i", source["path"], "-map", "0:v:0", "-an", "-vf",
               "scale=960:540:force_original_aspect_ratio=decrease:force_divisible_by=2,fps=15",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-threads", "2",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary)]
    process = subprocess.run(command,capture_output=True,timeout=max(120,duration*2),check=False)
    if process.returncode or not temporary.is_file():
        raise ValueError("浏览器同步预览生成失败")
    probe = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","json",str(temporary)],
                           capture_output=True,text=True,timeout=30,check=False)
    if probe.returncode or abs(float(json.loads(probe.stdout)["format"]["duration"])-duration) > .15:
        raise ValueError("预览视频与原视频时长不一致")
    original = Path(source["path"])
    observed = original.stat()
    if original.resolve() != Path(source["resolved_path"]) or observed.st_size != source["size_bytes"] or observed.st_mtime_ns != source["mtime_ns"] or _bounded_source_fingerprint(original)["source_fingerprint"] != source["source_fingerprint"]:
        raise ValueError("生成预览期间原视频身份变化")
    os.replace(temporary,target)
    record = speech_worker.file_record(target)
    speech_worker.atomic_json(receipt_path,{"identity":identity,"preview":record,"source_media_decodes":1,
                                           "physical_action_confirmation":False,"timing_scale":1})
    return {"path":str(target.relative_to(root)), **record, "cache_reused":False}

def build(root: Path) -> dict:
    json_root = root / 'JSON-Config-Files'
    speech_path = json_root / 'speech.json'
    index = speech_worker.read_json(speech_path, 16*1024*1024)
    seal_path = json_root / 'Input-Manifests/input_seal.json'
    seal = speech_worker.read_json(seal_path, 16*1024*1024)
    if not verify_input_seal(seal):
        raise ValueError('原始输入封存回执无效')
    manifest = RunManifest.model_validate(seal['manifest'])
    probe_path, alignment_path = json_root / 'video_probe.json', json_root / 'time_alignment.json'
    infos = {key: VideoInfo.model_validate(value) for key, value in speech_worker.read_json(probe_path).items()}
    if alignment_path.is_symlink() or alignment_path.stat().st_size > 2*1024*1024:
        raise ValueError('对齐控制文件无效')
    transforms = {value['view_id']: AlignmentTransform.model_validate(value) for value in json.loads(alignment_path.read_text(encoding='utf-8-sig'))}
    if index.get('alignment_file_sha256') != speech_worker.sha256(alignment_path):
        raise ValueError('录音对齐版本已变化')
    videos, audios = [], []
    for view in manifest.views:
        info, transform = infos[view.view_id], transforms[view.view_id]
        for ordinal, part in enumerate(view.segments or [view]):
            physical = info.segments[ordinal] if info.segments else info
            virtual = physical.virtual_start_ms if info.segments else 0
            if physical.path != part.video or physical.duration_ms <= 0 or physical.fps <= 0:
                raise ValueError('视频探测记录与封存输入不一致')
            # A failed transform is never extrapolated into a playback claim.
            valid = transform.state != 'failed' and all(s.state != 'failed' for s in transform.segment_transforms if s.segment_index == ordinal)
            source_record = next((r for r in seal['sources'] if r.get('kind') == 'video' and r.get('view_id') == view.view_id and r['path'] == str(part.video)), None)
            if valid and source_record:
                path = part.video
                observed = path.stat()
                if observed.st_size != source_record['size_bytes']:
                    raise ValueError('原视频大小已变化')
                identity = _bounded_source_fingerprint(path)
                if source_record.get('source_fingerprint'):
                    if identity['source_fingerprint'] != source_record['source_fingerprint'] or observed.st_mtime_ns != source_record.get('mtime_ns'):
                        raise ValueError('原视频身份已变化')
                elif source_record.get('content_hash_algorithm') == 'sha256':
                    if speech_worker.sha256(path) != source_record['content_hash']:
                        raise ValueError('上传视频哈希已变化')
                else:
                    # Unknown legacy identity contracts cannot authorize a new
                    # original-source stream; retained derived clips still work.
                    continue
                anchors = [[0, transform.to_global(virtual)], [physical.duration_ms/1000, transform.to_global(virtual+physical.duration_ms)]]
                if monotonic(anchors):
                    videos.append({'view_id':view.view_id, 'role':view.role.value, 'segment_ordinal':ordinal,
                                   'duration_seconds':physical.duration_ms/1000, 'anchors':anchors,
                                   'uncertainty_ms':transform.uncertainty_ms, 'timing_evidence':'PARTIAL_EVIDENCE',
                                   '_source':{'path':str(path), 'resolved_path':str(path.resolve()),
                                              'size_bytes':observed.st_size, 'mtime_ns':observed.st_mtime_ns, **identity}})
            for source in index['sources']:
                if source['view_id'] != view.view_id or source['segment_ordinal'] != ordinal:
                    continue
                for chunk in source.get('chunks', []):
                    request_spec = chunk['files']['request.json']
                    request_path = root / request_spec['path']
                    checked_artifact(root, request_spec)
                    request = speech_worker.read_json(request_path)
                    mapping = audio_map({**source,'_request':request}, chunk, part, physical, transform, virtual) if valid else {'anchors':[], 'state':'clock_unavailable'}
                    audios.append({'chunk_id':chunk['id'], **mapping})
    for video in videos:
        video['preview'] = make_preview(root, video['_source'], video['duration_seconds'])
    result = {'schema_version':'visioncortex-speech-timeline/1', 'videos':videos, 'audio':audios,
              'physical_action_confirmation':False, 'actual_av_sync':'PARTIAL_EVIDENCE',
              'bindings':{str(p.relative_to(root)):speech_worker.sha256(p) for p in (speech_path, seal_path, probe_path, alignment_path)}}
    speech_worker.atomic_json(root / CONTROL, result)
    return result


def checked_artifact(root: Path, spec: dict) -> Path:
    path = root / spec['path']
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('时间轴文件路径越界')
    if speech_worker.file_record(path) != {key:spec[key] for key in ('size','sha256')}:
        raise ValueError('时间轴依赖文件哈希不匹配')
    return path


def load(root: Path) -> dict | None:
    path = root / CONTROL
    if not path.is_file():
        return None
    result = speech_worker.read_json(path, 16*1024*1024)
    required = {'JSON-Config-Files/speech.json', 'JSON-Config-Files/Input-Manifests/input_seal.json',
                'JSON-Config-Files/video_probe.json', 'JSON-Config-Files/time_alignment.json'}
    if result.get('schema_version') != 'visioncortex-speech-timeline/1' or set(result.get('bindings', {})) != required:
        raise ValueError('时间轴版本或依赖无效')
    for relative, digest in result['bindings'].items():
        if speech_worker.sha256(root / relative) != digest:
            raise ValueError('时间轴引用的录音或对齐版本已变化')
    return {**result, 'sha256':speech_worker.sha256(path)}


def video_path(root: Path, timeline_hash: str, view_id: str, ordinal: int) -> Path:
    timeline = load(root)
    if not timeline or timeline['sha256'] != timeline_hash:
        raise ValueError('时间轴已更新，请刷新页面')
    video = next((v for v in timeline['videos'] if v['view_id'] == view_id and v['segment_ordinal'] == ordinal), None)
    if not video:
        raise ValueError('本实验不存在该视频来源')
    source = video['_source']
    path = Path(source['path'])
    seal = speech_worker.read_json(root / 'JSON-Config-Files/Input-Manifests/input_seal.json', 16*1024*1024)
    if not verify_input_seal(seal) or not any(r.get('kind') == 'video' and r.get('view_id') == view_id and r['path'] == str(path) for r in seal['sources']):
        raise ValueError('视频不在本实验封存输入内')
    observed = path.stat()
    if (path.resolve() != Path(source['resolved_path']) or observed.st_size != source['size_bytes']
            or observed.st_mtime_ns != source['mtime_ns']
            or _bounded_source_fingerprint(path)['source_fingerprint'] != source['source_fingerprint']):
        raise ValueError('原视频身份已变化')
    if not video.get('preview'):
        raise ValueError('请刷新时间轴以生成可跳转的浏览器预览')
    return checked_artifact(root, video['preview'])
