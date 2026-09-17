"""Incremental, local/NAS-readable overview of published evidence and live queues.

An index is the authority for published clips; queue completion is only an
execution metric. Neither is human ground truth or a current release receipt.
"""
from datetime import datetime
import html
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

from .device_day_contract import atomic_bytes, atomic_json, read_json, validate_archive_name
from .device_day_progress import snapshot
from .report_brand import brand_logo_data_url

STAGE_NAMES = {'retention': '原片归档', 'vision': 'YOLO 预处理', 'stt': '录音识别',
               'understanding': '多模态理解', 'report': '日报更新'}
WAIT_LABELS = {'upstream_failed': '上游失败待恢复', 'upstream_pending': '上游未完成',
               'provider_blocked': '云端账户不可用', 'night_window': '等待夜间窗口',
               'lease_recovery': '租约过期待恢复', 'pending_validation': '待校验调度'}


def union_seconds(intervals):
    end, total = 0, 0
    for start, stop in sorted(intervals):
        if stop > start:
            total += max(0, stop - max(start, end))
            end = max(end, stop)
    return total / 1e6


def summarize(name, index, config, *, report_exists=False, understanding_exists=False):
    """Count unique published records/materials at their actual grain."""
    validate_archive_name(name)
    if index.get('archive') != name:
        raise ValueError('Archive/index identity mismatch')
    camera, day = name[11:], name[:10]
    records = {r['recording_id']: r for r in index.get('recordings', [])}
    segments = {s['segment_id']: s for s in index.get('segments', [])}
    if any(s['recording_id'] not in records for s in segments.values()):
        raise ValueError('Segment has no source recording')
    active = [s for s in segments.values() if s.get('activity') == 'active']
    sources = {s['retained']['path']: s for r in records.values() for s in r.get('sources', [])}
    understood = {s['segment_id'] for s in index.get('understandings', [])} & segments.keys()
    keys = {f['path'] for s in active for f in s.get('key_frames', [])}
    covered = {s['recording_id'] for s in segments.values()}
    role = config.get('collection_ingest', {}).get('camera_role_map', {}).get(camera, 'unknown')
    return {'archive': name, 'date': day, 'camera': camera, 'role': role, 'updated_at': index.get('updated_at'),
            'index_path': name + '/ProcessedClips/Index.json', 'indexed_recordings': len(records),
            'preprocessed_recordings': len(covered), 'active_count': len(active),
            'inactive_count': sum(s.get('activity') == 'inactive' for s in segments.values()),
            'keyframes': len(keys), 'understood_segments': len(understood), 'segment_count': len(segments),
            'video_files': sum(s['kind'] == 'video' for s in sources.values()),
            'audio_files': sum(s['kind'] == 'audio_audio' for s in sources.values()),
            'csv_files': sum(p.lower().endswith('.csv') for p in sources),
            'source_bytes': sum(s['retained'].get('size_bytes', 0) for s in sources.values()),
            'stt_transcribed_recordings': sum(r.get('transcription', {}).get('status') == 'completed'
                                             and r.get('transcription', {}).get('outcome') == 'transcribed'
                                             for r in records.values()),
            'report_exists': report_exists, 'understanding_exists': understanding_exists,
            'active': [{k: s.get(k) for k in ('segment_id', 'recording_id', 'start_us', 'end_us',
                'start_ms', 'end_ms', 'video', 'source_ref', 'json_path', 'key_frames')} for s in active]}


def assemble(progress, archives, errors):
    dates = sorted(set(progress['days']) | {a['date'] for a in archives}, reverse=True)
    days = []
    for date in dates:
        rows = [a for a in archives if a['date'] == date]
        queued = progress['days'].get(date)
        sums = {k: sum(a[k] for a in rows) for k in ('indexed_recordings', 'preprocessed_recordings',
                'active_count', 'inactive_count', 'keyframes', 'video_files', 'audio_files', 'csv_files',
                'source_bytes', 'stt_transcribed_recordings', 'understood_segments', 'segment_count')}
        days.append({'date': date, **sums, 'archives': rows, 'queue': queued,
                     'discovered_recordings': queued['total'] if queued else None,
                     'camera_count': queued['camera_count'] if queued else len(rows),
                     'activity_seconds': union_seconds((s['start_us'], s['end_us']) for a in rows for s in a['active']),
                     'waiting': progress.get('waiting', {}).get(date, {}),
                     'read_errors': [e for e in errors if e['archive'].startswith(date)]})
    return {'updated_at': progress['observed_at'], 'days': days, 'progress': progress, 'errors': errors,
            'scope': 'published_indexes_and_queue_snapshots_not_human_ground_truth',
            'evidence_status': 'PARTIAL_EVIDENCE', 'refresh_seconds': 30}


class ArchiveOverview:
    def __init__(self):
        self.cache = {}
        self.repair_after = {}

    def publish(self, runner):
        progress = snapshot(runner.config)
        # Local observations run on this background worker, never the GPU or
        # capture discovery lanes. An unavailable sampler cannot block output.
        from .device_day_acceptance import observe, render as render_acceptance
        import sqlite3
        try:
            progress['acceptance_observations'] = observe(runner.config, progress)
            progress['latency_html'] = progress.get('latency_html', '') + render_acceptance(progress['acceptance_observations'])
        except (OSError, sqlite3.Error, ValueError, KeyError):
            progress.setdefault('errors', []).append('持续运行观察记录暂不可用')
        archives, errors, seen = [], [], set()
        # Indexes and file metadata only; never frames/media or model calls.
        for root in sorted(runner.archive_root.iterdir(), reverse=True):
            try:
                validate_archive_name(root.name)
            except ValueError:
                continue
            if root.is_symlink():
                continue
            seen.add(root.name)
            try:
                path = root / 'ProcessedClips/Index.json'
                stat = path.stat()
                version = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
                previous = self.cache.get(root.name)
                if not previous or previous[0] != version:
                    data = summarize(root.name, read_json(path), runner.config,
                        report_exists=(root / 'LaboratoryDailyReport/LaboratoryDailyReport.html').is_file(),
                        understanding_exists=(root / 'MultimodalUnderstanding/UnderstandingReport.html').is_file())
                    self.cache[root.name] = (version, data)
                archives.append(self.cache[root.name][1])
            except (OSError, ValueError, KeyError, TypeError) as exc:
                # Never silently substitute the last good value for an unreadable index.
                self.cache.pop(root.name, None)
                errors.append({'archive': root.name, 'reason': type(exc).__name__})
        self.cache = {k: v for k, v in self.cache.items() if k in seen}
        data = assemble(progress, archives, errors)
        data['archive_root'] = str(runner.archive_root.resolve())
        body = render(data).encode()
        atomic_json(runner.runtime_root / 'ArchiveOverview.json', data)
        atomic_bytes(runner.runtime_root / 'ArchiveOverview.html', body)
        atomic_bytes(runner.archive_root / 'Readme.html', body)
        # Rebuild one absent derived index from existing canonical receipts on
        # this independent worker. Never rerun a model or fabricate a completion.
        from .device_day_recovery import repair_missing_index
        from .device_day_schedule import processing_cutoff
        repair = None
        for error in ([] if processing_cutoff(runner.config.get('device_day', {})) else errors):
            if error['reason'] == 'FileNotFoundError' and self.repair_after.get(error['archive'], 0) < progress['observed_at']:
                self.repair_after[error['archive']] = progress['observed_at'] + 300
                repair = repair_missing_index(runner, error['archive'])
                break
        return {'status': 'completed', 'updated_at': data['updated_at'], 'archives': len(archives),
                'active_segments': sum(a['active_count'] for a in archives), 'errors': errors, 'index_repair': repair}


def render(data):
    def esc(value):
        return html.escape(str(value), quote=True)
    def stamp(us):
        return datetime.fromtimestamp(us / 1e6, ZoneInfo('Asia/Shanghai')).strftime('%H:%M:%S')
    def link(archive, path, label, fragment='', *, watch=False):
        # Restrict every link to the five existing archive directories.
        if (not path or '\\' in path or ':' in path or Path(path).is_absolute() or '..' in Path(path).parts
                or Path(path).parts[0] not in {'MetaVideo', 'ProcessedClips', 'MultimodalUnderstanding',
                                             'LaboratoryDailyReport', 'Comment'}):
            return '<span>素材引用不可用</span>'
        href = quote(archive + '/' + path, safe='/') + fragment
        return (f'<a data-artifact {"data-watch class=play-button" if watch else ""} data-archive="{esc(archive)}" data-path="{esc(path)}" '
                f'data-fragment="{esc(fragment)}" href="{esc(href)}" target="_blank" rel="noopener">{esc(label)}</a>')
    progress = data['progress']
    rows, details = [], []
    for day in data['days']:
        date, q = day['date'], day['queue']
        discovered = day['discovered_recordings']
        total = q.get('processing_total', discovered) if q else discovered
        missing = q.get('missing_input_count', 0) if q else 0
        processed = day['preprocessed_recordings']
        stages = q['stages'] if q else {}
        vision = stages.get('vision', {})
        pending = max(0, total-processed) if total is not None else None
        rows.append(f'''<tr data-day="{date}"><th><a href="#Day{date}">{date}</a><small>{day['camera_count']} 台设备</small></th>
<td>{discovered if discovered is not None else '未知'}<small>已发现分片；{missing} 个输入缺失，单独记录</small></td>
<td><b>{processed}</b> / {total if total is not None else '?'}<progress value="{processed}" max="{max(1,total or processed)}"></progress><small>已排除缺失输入；{pending if pending is not None else '未知'} 尚无发布结果</small></td>
<td><a class="activity" href="#Day{date}">{day['active_count']} 个</a><small>{day['activity_seconds']/60:.1f} 分钟，时间重叠去重</small></td>
<td>{day['keyframes']} 张</td><td>{day['audio_files']} / {day['stt_transcribed_recordings']}<small>原录音 / 已转写分片</small></td>
<td>{day['understood_segments']} / {day['segment_count']}<small>已理解 / 已发布区间</small></td>
<td>{vision.get('running',0)} 运行 · {vision.get('failed',0)} 失败<small>{'；'.join(f'{n} {WAIT_LABELS[k]}' for k,n in day['waiting'].get('vision',{}).items()) or '无排队项'}</small>{'<small class="error">索引读取不全</small>' if day['read_errors'] else ''}</td></tr>''')
        stage_rows = ''.join(f'<tr><th>{label}</th><td>{stages.get(stage,{}).get("completed",0)}</td>'
            f'<td>{stages.get(stage,{}).get("running",0)}</td><td>{stages.get(stage,{}).get("failed",0)}</td>'
            f'<td>{stages.get(stage,{}).get("not_enqueued",0)}</td><td>'
            + ('；'.join(f'{n} {WAIT_LABELS[k]}' for k,n in day['waiting'].get(stage,{}).items()) or '无') + '</td></tr>'
            for stage,label in STAGE_NAMES.items())
        device_rows, clip_rows = [], []
        for a in day['archives']:
            name = a['archive']
            refs = [link(name, 'ProcessedClips/Index.json', '总索引')]
            if a['report_exists']:
                refs.append(link(name,'LaboratoryDailyReport/LaboratoryDailyReport.html','日报（已生成部分）'))
            if a['understanding_exists']:
                refs.append(link(name,'MultimodalUnderstanding/UnderstandingReport.html','多模态报告'))
            role = {'first_person':'第一人称','third_person':'第三人称'}.get(a['role'],'视角待配置')
            device_rows.append(f'<tr><th>{esc(a["camera"])}<small>{role}</small></th><td>{a["video_files"]}</td>'
                f'<td>{a["active_count"]} / {a["inactive_count"]}</td><td>{a["csv_files"]}</td><td>{a["audio_files"]}</td>'
                f'<td>{" · ".join(refs)}<small>索引更新：{esc(a["updated_at"])}</small></td></tr>')
            for s in a['active']:
                ref = s.get('video') or s.get('source_ref') or {}
                fragment = '' if s.get('video') else f'#t={(s.get("start_ms") or 0)/1000:.3f},{(s.get("end_ms") or 0)/1000:.3f}'
                frames = ''.join(f'<li>{link(name,f["path"],"动作关键帧 " + str(i+1))}</li>' for i,f in enumerate(s.get('key_frames') or []))
                nas_path = ''
                if ref.get('path') and '素材引用不可用' not in link(name, ref['path'], ''):
                    nas_path = str(Path(data.get('archive_root', 'VisionCortexExperimentArchive'))/name/ref['path'])
                location = (f'<details class="nas-location"><summary>在 NAS 中的位置</summary>'
                            f'<input readonly aria-label="NAS 视频路径" value="{esc(nas_path)}">'
                            '<button data-copy-path>复制路径</button></details>') if nas_path else ''
                clip_rows.append((s['start_us'], f'<tr><td>{stamp(s["start_us"])}—{stamp(s["end_us"])}</td>'
                    f'<td>{esc(a["camera"])}<small>{role}</small></td><td>{link(name,ref.get("path"),"打开实验视频",fragment,watch=True)} '
                    f'<button data-multiview data-app data-day-value="{date}" data-archive="{esc(name)}" data-segment="{esc(s["segment_id"])}">同一时段 · 多视角</button>'
                    f'{location}<small>{link(name,s.get("json_path"),"片段索引")}</small></td><td><details><summary>{len(s.get("key_frames") or [])} 张动作关键帧</summary><ul>{frames}</ul></details></td></tr>'))
        clips = ''.join(row for _,row in sorted(clip_rows))
        timing = progress.get('vision_timings', {}).get(date)
        timing_text = (f'最近 {timing["samples"]} 个有耗时记录的 YOLO 分片任务：中位数 {timing["median_seconds"]:.1f} 秒，'
                       f'P90 {timing["p90_seconds"]:.1f} 秒。包含读取、供帧与后处理，含历史版本，不是纯 GPU 推理耗时。'
                       if timing else '尚无可统计的 YOLO 分片任务耗时。')
        details.append(f'''<section id="Day{date}" data-day="{date}" data-activity-count="{day['active_count']}"><h2>{date} · 实验片段</h2>
<p>已发布 {day['active_count']} 个活动区间、{day['inactive_count']} 个无活动／无关区间。原片与资料 {day['source_bytes']/1024**3:.2f} GiB。</p>
<div class="table-wrap"><table><thead><tr><th>采集时间</th><th>设备与视角</th><th>观看实验片段</th><th>关键帧入口</th></tr></thead><tbody>{clips or '<tr><td colspan="4">该日期暂无已发布的活动片段。请选择上方标有片段数量的日期；尚未处理或漏检的可能性不能排除。</td></tr>'}</tbody></table></div>
<details><summary>查看各设备的原片、采集资料与报告</summary><div class="table-wrap"><table><thead><tr><th>设备</th><th>原视频</th><th>活动 / 无活动区间</th><th>CSV</th><th>原录音</th><th>查阅入口</th></tr></thead><tbody>{''.join(device_rows) or '<tr><td colspan="6">尚无可读取索引</td></tr>'}</tbody></table></div></details>
<details><summary>查看五个阶段的处理状态与 YOLO 耗时</summary><div class="table-wrap"><table><thead><tr><th>阶段</th><th>任务完成</th><th>运行</th><th>失败</th><th>尚未入队</th><th>入队后的等待原因</th></tr></thead><tbody>{stage_rows}</tbody></table></div><p>{timing_text}</p><p>任务完成数来自执行队列，可能含历史版本及无录音的正常结束；日报任务逐分片更新同一份报告，不代表同等数量的日报。</p></details></section>''')
    running = ''.join(f'<li>{j["date"]} · {esc(j["camera"])} · {STAGE_NAMES[j["stage"]]} · {stamp(j["start_us"])}—{stamp(j["end_us"])} · {esc(j.get("phase","处理中"))}</li>' for j in progress['running'])
    updated = datetime.fromtimestamp(data['updated_at'],ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
    schedule = progress['night_schedule']
    night = f'多模态和日报在 {schedule["start"]}–次日 {schedule["end"]} 运行。' if schedule['enabled'] else '夜间时段限制未启用。'
    cloud = '<p class="error">云端账户当前不可用（Arrearage），录音识别和多模态等待服务恢复；原片归档与 YOLO 独立运行。</p>' if progress.get('provider',{}).get('active') else ''
    errors = ''.join(f'<li>{esc(e["archive"])}：{esc("总索引待恢复" if e["reason"] == "FileNotFoundError" else "总索引读取异常")}</li>' for e in data['errors'])
    errors += ''.join(f'<li>{esc(e)}</li>' for e in progress['errors'])
    activity_dates = ''.join(f'<a class="date-chip" href="#Day{d["date"]}">{d["date"]}<b>{d["active_count"]} 个片段</b></a>'
                            for d in data['days'] if d['active_count'])
    ordered_details = [row for day, row in zip(data['days'], details, strict=True) if day['active_count']]
    ordered_details += [row for day, row in zip(data['days'], details, strict=True) if not day['active_count']]
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>VisionCortex 实验数据总览</title>
<style>body{{margin:0;background:#eaf2f3;color:#112d32;font:15px/1.6 system-ui}}main{{max-width:1480px;margin:auto;padding:28px}}header,section{{background:#fbfdfd;padding:24px;margin:0 0 20px;border:1px solid #cbdadc;border-radius:14px}}header{{border-top:6px solid #1e4a52}}header img{{width:48px;float:left;margin-right:14px}}h1{{font-size:27px;margin:0}}h2{{font-size:21px}}h3{{font-size:17px}}a{{color:#216d77;text-underline-offset:3px}}small{{display:block;color:#526d73;font-size:12px}}.table-wrap{{overflow:auto}}table{{border-collapse:collapse;width:100%;text-align:left}}th,td{{padding:12px;border-bottom:1px solid #dbe6e8;vertical-align:top}}th{{font-weight:600}}td{{min-width:70px}}progress{{display:block;width:130px;height:9px;accent-color:#208a68}}.activity{{font-size:19px;font-weight:700}}details{{margin:15px 0}}td details{{margin:0}}summary{{cursor:pointer;color:#216d77}}.error{{color:#a34435}}select,button{{font:inherit;padding:8px 12px;border:1px solid #cbdadc;border-radius:6px;background:white}}nav{{display:flex;gap:16px;align-items:center;flex-wrap:wrap}}[hidden]{{display:none!important}}:focus-visible{{outline:3px solid #cf8337}}@media(max-width:700px){{main{{padding:10px}}header,section{{padding:16px}}h1{{font-size:22px}}}}@media print{{nav{{display:none}}section{{break-inside:avoid}}}}</style></head><body><main>
<header><img src="{brand_logo_data_url()}" alt="VisionCortex"><h1>实验数据总览</h1><p>数据更新：<time data-updated="{data['updated_at']}">{updated}</time>（北京时间） · 后台每 30 秒汇总，页面每 60 秒刷新</p>
<nav><label>采集日期 <select id="DateFilter"><option value="">全部日期</option>{''.join(f'<option>{d["date"]}</option>' for d in data['days'])}</select></label><button id="Refresh">刷新</button><a href="#Operations">后台处理进度</a><a data-app href="/#/day-timeline">返回应用时间线</a></nav></header>
<section id="Experiments"><h2>实验片段，从这里看</h2><nav class="date-chips">{activity_dates or '当前暂无已发布活动片段。'}</nav><p>选择日期，然后点击“打开实验视频”；“同一时段 · 多视角”可查看其他相机，包括未检出活动的对照画面。</p><small>以下是模型筛选结果，未作人工确认。每个片段都存放在 NAS 对应设备日的 ProcessedClips/Clips 中，展开“在 NAS 中的位置”可复制完整路径。</small></section>
{''.join(ordered_details)}
<div id="Operations"></div>
<section><h2>当前运行情况</h2><p>原片、YOLO 按新分片持续处理。{night}无新增采集的日期不会被当作“无实验活动”。</p>{cloud}
<ul>{running or '<li>当前没有有效运行任务，具体等待原因见各日期的阶段状态。</li>'}</ul>
{progress.get('latency_html', '')}
<p>采集端原视频软链接替换：{'已启用' if progress['capture_link_cleanup']['enabled'] else '尚未启用，原片可能仍有两份'}。</p>{'<details class="error"><summary>查看索引与队列读取异常（正在后台核对恢复）</summary><ul>'+errors+'</ul></details>' if errors else ''}</section>
<section><h2>每天处理到哪里了</h2><p>发布结果按原始分片去重；活动时长按同一天全部视角的采集时间合并，不重复相加，也不代表已确认的实验员工作时长。</p>
<div class="table-wrap"><table><thead><tr><th>采集日期</th><th>采集量</th><th>已有预处理结果</th><th>检出活动</th><th>动作关键帧</th><th>录音 / 转写</th><th>多模态理解</th><th>YOLO 当前状态</th></tr></thead><tbody>{''.join(rows) or '<tr><td colspan="8">尚无可读取的采集数据</td></tr>'}</tbody></table></div></section>
<p>下方活动入口来自当前发布索引中的模型筛选结果，供核查实际实验视频；未作人工确认，不能等同人工真值。无活动区间的抽样图片不计入动作关键帧。统计来源：各设备的 ProcessedClips/Index.json、采集清单及本地阶段队列。</p>
<footer>五目录保持不变：MetaVideo / ProcessedClips / MultimodalUnderstanding / LaboratoryDailyReport / Comment。ExperimentActivity 为活动片段；NoExperimentActivity 为无活动／无关区间，引用原片对应时间。JSON 是可展开的溯源数据，直接阅读请选择 HTML 报告。</footer></main>
<dialog id="Playback"><form method="dialog"><button aria-label="关闭播放器">关闭</button></form><h2 id="PlaybackTitle">实验视频</h2><div id="PlaybackBody"></div></dialog>
<style>header{{padding:16px 24px}}header p{{margin:6px 0 12px}}h2{{margin-top:0}}.date-chip{{border:1px solid #b9d4d7;padding:6px 12px;border-radius:8px;text-decoration:none;background:#edf7f5}}.date-chip b{{display:block;font-size:12px}}.play-button{{display:inline-block;padding:8px 14px;background:#1e4a52;color:white;border-radius:6px;text-decoration:none;white-space:nowrap;margin-bottom:6px}}button{{cursor:pointer}}.nas-location input{{font:12px monospace;max-width:100%;width:420px;box-sizing:border-box;padding:8px}}td small,td input{{overflow-wrap:anywhere}}dialog{{border:1px solid #bfd3d6;border-radius:12px;width:min(1320px,94vw);max-height:90vh;box-sizing:border-box;color:#112d32}}dialog::backdrop{{background:#102e35b3}}dialog form{{float:right}}dialog video{{display:block;width:100%;max-height:65vh;background:#132328}}.view-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,310px),1fr));gap:16px}}.view-grid article{{min-width:0;border:1px solid #d5e3e6;padding:12px;border-radius:8px}}.view-grid h3{{overflow-wrap:anywhere}}.view-grid video{{aspect-ratio:1.6;max-height:45vh}}.playback-controls{{display:flex;align-items:center;flex-wrap:wrap;gap:10px;margin:14px 0}}.playback-controls input{{flex:1;min-width:150px}}section[id]{{scroll-margin-top:12px}}</style>
<script>{Path(__file__).with_name('web').joinpath('archive-playback.js').read_text()}</script>
<script>{Path(__file__).with_name('web').joinpath('archive-overview.js').read_text()}</script></body></html>'''
