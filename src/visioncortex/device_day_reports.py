"""One coherent daily report, rendered only from persisted stage outputs."""
from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

from .device_day_contract import TIMEZONE, VERSION, atomic_bytes, atomic_json, digest, file_hash, read_json, safe_child


def text(value):
    return html.escape(str(value or ""), quote=True)


def clock(value):
    return datetime.fromtimestamp(value / 1e6, ZoneInfo(TIMEZONE)).strftime("%H:%M:%S")


def material_link(reference, label):
    path = reference.get("path") if isinstance(reference, dict) else reference
    if not path:
        return ""
    return f'<a href="../{quote(path, safe="/")}">{text(label)}</a>'


def template_contract(identifier):
    path = Path(__file__).with_name("templates") / f"{identifier}.json"
    template = read_json(path)
    contract = {"template_id": template["template_id"], "template_schema_version": template["schema_version"],
                "template_sha256": file_hash(path), "renderer_sha256": file_hash(Path(__file__))}
    if template.get("base_template_id"):
        base_path = path.with_name(template["base_template_id"] + ".json")
        base = read_json(base_path)
        contract["base_template_id"] = base["template_id"]
        contract["base_template_sha256"] = file_hash(base_path)
        template["sections"] = base["sections"]
    return template, contract


def preserve_prior_synthesis(layout, replacement):
    """Keep a completed contextual report when later inputs make it stale."""
    path = layout.reports/'LaboratoryDailyReport.json'
    if not path.is_file():
        return
    try:
        previous = read_json(path)
        prior = previous.get('全天综合理解') or {}
        if not isinstance(prior, dict):
            raise ValueError('Current daily synthesis is not an object')
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        import hashlib
        def identity():
            stat = path.stat()
            return {field: getattr(stat, field) for field in (
                'st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')}
        before = identity()
        raw = path.read_bytes()
        if identity() != before or len(raw) != before['st_size']:
            raise ValueError('Unverified current report changed during preservation') from exc
        checksum = hashlib.sha256(raw).hexdigest()
        saved = safe_child(layout.root, 'LaboratoryDailyReport/.Evidence/History/UnverifiedReport' + checksum + '.json')
        if saved.exists():
            if saved.stat().st_size != len(raw) or file_hash(saved) != checksum:
                raise ValueError('Unverified report history collision; preserving both originals') from exc
        else:
            atomic_bytes(saved, raw)
        if (saved.stat().st_size != len(raw) or file_hash(saved) != checksum or identity() != before):
            raise ValueError('Unverified report preservation failed byte/source verification') from exc
        recovery = {'schema_version': 'visioncortex-unverified-report-preservation/1',
            'source_path': layout.relative(path), 'source_stat': before,
            'preserved_reference': {'path': layout.relative(saved), 'size_bytes': len(raw), 'sha256': checksum},
            'error_type': type(exc).__name__, 'reason': 'current_report_unreadable_or_invalid',
            'evidence_status': 'NOT_PROVEN', 'model_invoked': False,
            'recovery_action': 'preserve_exact_unverified_bytes_then_publish_verified_history_or_explicit_unverified_state'}
        record = safe_child(layout.root, 'LaboratoryDailyReport/.Evidence/Recovery/Report' + digest(recovery) + '.json')
        if record.exists():
            if read_json(record) != recovery:
                raise ValueError('Unverified report recovery record collision') from exc
        else:
            atomic_json(record, recovery)
        if identity() != before:
            raise ValueError('Unverified current report changed before recovery publication') from exc
        return
    if prior.get('status') == 'completed_for_available_inputs' and prior != replacement:
        atomic_json(layout.reports/'.Evidence/History'/('Report'+digest(previous)+'.json'), previous)


def render_page(title, template, contract, contents, *, scope_label='单设备观察记录'):
    from .report_brand import BRAND_COLORS, brand_logo_data_url
    color = BRAND_COLORS
    sections = ''.join(f'<section id="{section["id"]}"><h2>{text(section["title"])}</h2>{contents[section["id"]]}</section>' for section in template["sections"])
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{text(title)}</title>
<style>body{{background:{color['background']};color:{color['text']};font:16px/1.7 system-ui;margin:0}}main{{max-width:1100px;margin:32px auto;padding:0 20px}}header,section{{background:{color['surface']};border:1px solid {color['border']};border-radius:16px;padding:24px;margin-bottom:20px}}header{{border-top:8px solid {color['brand']}}}header img{{width:56px;float:left;margin-right:16px}}h1{{font-size:26px;overflow-wrap:anywhere}}h2,h3,a{{color:{color['brand']}}}article{{border-top:1px solid {color['border']};margin-top:20px;padding-top:12px}}.frames{{display:flex;gap:16px;flex-wrap:wrap}}figure{{margin:0;max-width:280px}}figure img{{width:100%;border-radius:8px}}figcaption,small,.uncertain{{font-size:13px;color:{color['text_muted']}}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid {color['border']};text-align:left}}@media print{{body{{background:white}}section{{break-inside:avoid}}}}</style>
<main><header><img src="{brand_logo_data_url()}" alt="VisionCortex"><h1>{text(title)}</h1><p>{text(scope_label)} · PARTIAL_EVIDENCE</p></header>{sections}<footer>{text(contract['template_id'])} · 模板 SHA-256：{text(contract['template_sha256'])}<p>本次页面排版不新增模型调用；片段理解与全天综合的实际调用及用量见各自回执。</p></footer></main></html>'''


def day_entries(index):
    """Pure persisted-window projection shared by the day publisher and proof."""
    from .device_day_narrative import effective_meaning, report_candidates
    meanings = {row['segment_id']: effective_meaning(row) for row in index['understandings']}
    records = {row['recording_id']: row for row in index['recordings']}
    result = []
    for segment in report_candidates(index):
        meaning = meanings.get(segment['segment_id'])
        recording = records.get(segment['recording_id'], {})
        label = f'{clock(segment["start_us"])}—{clock(segment["end_us"])} · {segment["activity_label"]}'
        result.append({'segment_id': segment['segment_id'], 'label': label,
            'start_us': segment['start_us'], 'end_us': segment['end_us'],
            'understanding_status': (meaning or {}).get('status', 'completed' if meaning else 'pending'),
            'understanding': meaning, 'key_frames': segment['key_frames'], 'scene_frames': segment['scene_frames'],
            'source_ref': segment['source_ref'], 'segment_json': segment['json_path'],
            'transcription': recording.get('transcription'), 'audio': recording.get('audio')})
    return result


def day_document(layout, index, synthesis, *, readable=None, entries=None, report_generated_at=None):
    """Pure JSON projection; only report_generated_at is a publication clock."""
    from .device_day_narrative import day_record
    if readable is None:
        readable = day_record(index, layout.root)
    day_template, day_contract = template_contract('VC-DEVICE-DAY-REPORT-V1')
    return {'全天综合理解': synthesis, '全天记录': readable, '视频留存统计': readable['视频留存统计'],
        'report_generated_at': report_generated_at or datetime.now(ZoneInfo(TIMEZONE)).isoformat(timespec='microseconds'),
        'schema_version': VERSION, 'archive': layout.name, 'logical_report_count': 1, **day_contract,
        'evidence_status': 'PARTIAL_EVIDENCE', 'entries': day_entries(index) if entries is None else entries,
        'additional_mllm_calls': False, 'additional_model_tokens': 0,
        'sections': day_template['sections'], 'input_index_digest': digest(index)}


def render_day(layout, index):
    from .device_day_narrative import day_record, effective_meaning, report_candidates
    readable = day_record(index, layout.root)
    segments = report_candidates(index)
    from .device_day_synthesis import current_synthesis
    synthesis = current_synthesis(layout, index)
    preserve_prior_synthesis(layout, synthesis)
    day_scope = '跨设备全天上下文' if len(synthesis.get('context_archives', [])) > 1 else '单设备全天上下文'
    from .runtime_control import check_yield, defer_yield
    check_yield(checkpoint=False)
    by_segment = {item["segment_id"]: effective_meaning(item) for item in index["understandings"]}
    by_recording = {item["recording_id"]: item for item in index["recordings"]}
    sections, briefs, context_rows, uncertainty_rows, execution_rows = [], [], [], [], []
    daily_entries = day_entries(index)
    provider_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    usage_unknown = 0
    for segment in segments:
        check_yield(checkpoint=False)
        understanding = by_segment.get(segment["segment_id"])
        label = f'{clock(segment["start_us"])}—{clock(segment["end_us"])} · {segment["activity_label"]}'
        paragraphs = []
        recording = by_recording.get(segment["recording_id"], {})
        for ref in recording.get("audio", {}).get("artifacts", []):
            if ref.get("kind") == "audio_audio":
                paragraphs.append('<p>' + material_link(ref, "播放原录音") + '</p>')
        transcription = recording.get("transcription") or {}
        for comment in transcription.get("comments") or []:
            if comment["start_us"] < segment["end_us"] and comment["end_us"] > segment["start_us"]:
                paragraphs.append(f'<p><strong>{clock(comment["start_us"])} 录音识别</strong> {text(comment["text"])} '
                                  f'{material_link(comment["transcript_path"], "STT溯源")} <small>自动语音识别</small></p>')
        for window in (understanding or {}).get("windows", []):
            check_yield(checkpoint=False)
            if window.get("input"):
                metadata = read_json(safe_child(layout.root, window["input"])).get("metadata", {})
                for comment in metadata.get("comments", []):
                    if comment.get("source") == "human_comment":
                        paragraphs.append(f'<p><strong>{clock(comment["start_us"])} comment</strong> {text(comment["text"])}</p>')
                protocol = metadata.get("protocol")
                if protocol:
                    paragraphs.append(f'<p><strong>补充参考材料</strong> {text(protocol.get("text", ""))}</p>')
                context_rows.append('<li>' + text(label) + ' · ' + material_link(window["input"], "本次 comment、protocol、原片和采样来源") + '</li>')
            if window.get("model_receipt"):
                result = read_json(safe_child(layout.root, window["model_receipt"])).get("model_result", {})
                execution_rows.append('<tr><td>' + text(label) + '</td><td>' + text(result.get("response_model") or result.get("model")) + '</td><td>' + ('校验复用已有响应' if window.get("response_cache_reused") else '模型实际调用') + '</td><td>' + material_link(window["model_receipt"], "执行与用量回执") + '</td></tr>')
            usage = window.get("usage") or {}
            if any(usage.get(k) is None for k in provider_usage):
                usage_unknown += 1
            else:
                for key in provider_usage:
                    provider_usage[key] += usage[key]
            paragraphs.append(f'<p>{text(window["summary"])}</p>')
            if not window.get('step_schema'):
                paragraphs.append('<p><small>旧版场景与行为观察；尚未生成结构化实验步骤。</small></p>')
            origin = segment["start_us"] - round(segment["start_ms"] * 1000)
            for step in window.get("steps", []):
                when = clock(step.get("start_us", origin + round(step["start_ms"] * 1000)))
                basis = {"observed": "画面观察", "inferred": "推断", "uncertain": "不确定"}.get(step["basis"], step["basis"])
                paragraphs.append(f'<p><strong>{when}</strong> {text(step["description"])} '
                                  f'<small>（{text(basis)}）</small></p>')
            if window.get("activity_observed") != segment["activity"]:
                paragraphs.append('<p>抽帧理解与筛选分类存在差异或不确定，需保留两者依据，不能认定完整实验事实。</p>')
            for uncertainty in window.get("uncertainties", []):
                paragraphs.append(f'<p class="uncertain">{text(uncertainty)}</p>')
                uncertainty_rows.append(f'<li>{text(label)}：{text(uncertainty)}</li>')
        if not understanding:
            state = (recording.get('stages', {}).get('understanding') or {})
            paragraphs.append(f'<p>此片段的多模态理解尚未完成。状态：{text(state.get("status", "pending"))}；{text(state.get("message"))}</p>')
        elif understanding.get('status') == 'partial':
            paragraphs.append('<p>已展示完成的理解窗口；其余窗口仍待完成，不能视为全片理解完成。</p>')
        frames = []
        for frame in [*segment["key_frames"], *segment["scene_frames"]]:
            check_yield(checkpoint=False)
            frame_label = "动作关键帧" if frame.get("frame_kind") == "action_keyframe" else "场景采样帧"
            url = "../" + quote(frame["path"], safe="/")
            frames.append(f'<figure><a href="{url}"><img loading="lazy" src="{url}" alt="{frame_label}"></a>'
                          f'<figcaption>{text(frame_label + "：" + (frame.get("understanding_text") or "理解待完成"))}</figcaption></figure>')
        video_link = material_link(segment.get("video") or segment["source_ref"], "打开视频")
        if not segment.get("video"):
            source_url = '../' + quote(segment['source_ref']['path'], safe='/')
            video_link = f'<a href="{source_url}#t={segment["start_ms"] / 1000},{segment["end_ms"] / 1000}">打开原片对应区间</a>'
        metadata_link = material_link(segment["json_path"], "片段JSON与素材索引")
        sections.append(f'<article><h3>{text(label)}</h3>{"".join(paragraphs)}'
                        f'<div class="frames">{"".join(frames)}</div><p>{video_link} · {metadata_link}</p></article>')
        briefs.append(f'<article><h3>{text(label)}</h3>{"".join(paragraphs)}<p>{video_link} · {metadata_link}</p>'
                      f'<details><summary>本时段动作关键帧与场景采样理解</summary><div class="frames">{"".join(frames)}</div></details></article>')
    pending = sum(1 for record in by_recording.values()
                  if not any(s["recording_id"] == record["recording_id"] for s in segments)
                  or any(s["recording_id"] == record["recording_id"] and (s["segment_id"] not in by_segment
                         or by_segment[s["segment_id"]].get('status') == 'partial')
                         for s in segments))
    retained = readable['视频留存统计']
    overview = f'<p>{retained["original_video_count"]} 个已索引原始视频，{retained["original_recording_count"]} 条采集记录；{len(segments)} 个分析候选；{pending} 条记录仍有阶段待完成。</p><p>{text(readable["覆盖说明"])}</p>'
    overview += (f'<p>选中 {retained["selected_interval_count"]} 个精确片段，实际可读 '
        f'{retained["available_selected_interval_count"]} 个，已验证视频总时长 '
        f'{retained["actual_video_duration_seconds"]} 秒；待判定 {retained["pending_interval_count"]} 个区间，'
        f'确认无价值 {retained["confirmed_no_value_interval_count"]} 个区间。'
        '全部分区、未发布原因、采样覆盖与缺口直接见日报 JSON 的“视频留存统计”。</p>')
    overview += '<article><h3>模型全天综合理解</h3>'
    if synthesis.get('status') in {'completed_for_available_inputs', 'updating_for_new_inputs'}:
        overview += '<p>' + text(synthesis.get('说明', '模型结合当前已保存全天证据综合理解；完整性和未处理范围见理解文件。')) + '</p>'
        if synthesis.get('generated_at'):
            overview += '<p>最近综合时间：' + text(synthesis['generated_at']) + '</p>'
        for finding in synthesis['findings']:
            overview += f'<p><strong>{text(finding["title"])}</strong> {text(finding["text"])}</p>'
            overview += '<small>' + text('；'.join(finding['uncertainties'])) + '</small>'
    else:
        overview += '<p>' + text(synthesis['说明']) + '</p>'
    overview += '</article>'
    overview += '<details><summary>全天录音时间线（包括没有对应视频的语音）</summary>' + ''.join(
        f'<p><strong>{text(s["开始时间"])}</strong> {text(s["原话"])}</p>' for s in readable['录音时间线']) + '</details>'
    overview += '<details><summary>全天现场事件时间线</summary>' + ''.join(
        f'<p>{text(e["记录时间"])} · {text(e["类型"])} · 详见日报 JSON 的现场事件时间线及原始结果。</p>'
        for e in readable['现场事件时间线']) + '</details>'
    readable_html = []
    for row in readable['全天时间线']:
        body = [f'<article><h3>{text(row["开始时间"])} — {text(row["结束时间"])}</h3>'
                f'<p><strong>实验员：{text(row["实验员"]["姓名"])}</strong> · {text(row["实验员"]["状态"])}</p>']
        for field in ('实验步骤', '非实验工作'):
            for step in row[field]:
                body.append(f'<p><strong>{text(field)} · {text(step["动作"])}</strong> {text(step["内容"])} '
                            f'<small>{text(step["开始时间"])}；操作前：{text(step["操作前"])}；操作后：{text(step["操作后"])}</small></p>')
        if not row['实验步骤'] and not row['非实验工作']:
            body.extend(f'<p>{text(w["内容"])}</p>' for w in row['分窗口观察'])
        for reading in row['仪器与现场记录']:
            import json
            body.append(f'<p><strong>{text(reading["记录时间"])} 现场记录</strong> '
                        f'{text(json.dumps(reading["识别内容"], ensure_ascii=False))}<small>{text(reading["关联依据"])}</small></p>')
        for speech in row['录音记录']:
            body.append(f'<p><strong>{text(speech["开始时间"])} 录音识别</strong> {text(speech["原话"])}</p>')
        body.append(f'<p><small>留存：{text(row["留存判断"]["决定"])}；{text(row["质量说明"])}</small></p></article>')
        readable_html.append(''.join(body))
    uncertainties = '<ul>' + ''.join(uncertainty_rows) + '</ul><p>单设备筛选不等于跨视角物理动作验收；稀疏抽帧不证明完整观察了相邻采样之间的过程。</p>'
    cost = f'<p>理解结果对应的已知用量：输入 {provider_usage["input_tokens"]}，输出 {provider_usage["output_tokens"]}，合计 {provider_usage["total_tokens"]} Token。{usage_unknown} 个窗口用量未完整提供。复用响应的用量属于原执行记录，不代表本轮新增消耗。</p><table><tr><th>时段</th><th>模型</th><th>结果来源</th><th>耗时与 Token</th></tr>' + ''.join(execution_rows) + '</table>'
    if synthesis.get('model_receipts'):
        receipts = synthesis['model_receipts']
        actual = [r for r in receipts if not r.get('response_cache_reused')]
        known = sum((r.get('usage') or {}).get('total_tokens') or 0 for r in actual)
        missing = sum((r.get('usage') or {}).get('total_tokens') is None for r in actual)
        cost += f'<p>全天综合单独记录：{len(receipts)} 个回执，本轮新增 {len(actual)} 次调用，已知新增 {known} Token，{missing} 项用量未知。失败尝试和未确认结果另保留在实际用量账本。</p>'
    day_template, day_contract = template_contract("VC-DEVICE-DAY-REPORT-V1")
    daily_contents = {"daily_overview": overview, "experiment_briefs": ''.join(readable_html) + '<details><summary>逐片段画面、模型执行与来源</summary>' + ''.join(briefs) + '</details>',
                      "attention_and_handoff": uncertainties, "cost_and_timing": cost}
    check_yield(checkpoint=False)
    # Once publication begins, finish this report generation. Each existing
    # atomic writer retains its old file until replacement; the journal covers
    # process death between files. Arrivals yield before this commit group.
    with defer_yield():
        atomic_bytes(layout.reports / "LaboratoryDailyReport.html", render_page(f"{layout.name} 实验室日报", day_template, day_contract, daily_contents, scope_label=day_scope).encode())
        atomic_json(layout.reports / "LaboratoryDailyReport.json",
                    day_document(layout, index, synthesis, readable=readable, entries=daily_entries))
