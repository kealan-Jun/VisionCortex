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


def render_page(title, template, contract, contents):
    from .report_brand import BRAND_COLORS, brand_logo_data_url
    color = BRAND_COLORS
    sections = ''.join(f'<section id="{section["id"]}"><h2>{text(section["title"])}</h2>{contents[section["id"]]}</section>' for section in template["sections"])
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{text(title)}</title>
<style>body{{background:{color['background']};color:{color['text']};font:16px/1.7 system-ui;margin:0}}main{{max-width:1100px;margin:32px auto;padding:0 20px}}header,section{{background:{color['surface']};border:1px solid {color['border']};border-radius:16px;padding:24px;margin-bottom:20px}}header{{border-top:8px solid {color['brand']}}}header img{{width:56px;float:left;margin-right:16px}}h1{{font-size:26px;overflow-wrap:anywhere}}h2,h3,a{{color:{color['brand']}}}article{{border-top:1px solid {color['border']};margin-top:20px;padding-top:12px}}.frames{{display:flex;gap:16px;flex-wrap:wrap}}figure{{margin:0;max-width:280px}}figure img{{width:100%;border-radius:8px}}figcaption,small,.uncertain{{font-size:13px;color:{color['text_muted']}}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid {color['border']};text-align:left}}@media print{{body{{background:white}}section{{break-inside:avoid}}}}</style>
<main><header><img src="{brand_logo_data_url()}" alt="VisionCortex"><h1>{text(title)}</h1><p>单设备观察记录 · PARTIAL_EVIDENCE</p></header>{sections}<footer>{text(contract['template_id'])} · 模板 SHA-256：{text(contract['template_sha256'])}<p>报告整理复用已落盘理解，新增模型调用 0，新增 Token 0。</p></footer></main></html>'''


def render_day(layout, index):
    by_segment = {item["segment_id"]: item for item in index["understandings"]}
    by_recording = {item["recording_id"]: item for item in index["recordings"]}
    sections, briefs, context_rows, uncertainty_rows, execution_rows = [], [], [], [], []
    daily_entries = []
    provider_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    usage_unknown = 0
    for segment in index["segments"]:
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
                                  f'{material_link(comment["transcript_path"], "STT溯源")} <small>机器识别，未经人工复核</small></p>')
        for window in (understanding or {}).get("windows", []):
            if window.get("input"):
                metadata = read_json(safe_child(layout.root, window["input"])).get("metadata", {})
                for comment in metadata.get("comments", []):
                    if comment.get("source") == "human_comment":
                        paragraphs.append(f'<p><strong>{clock(comment["start_us"])} comment</strong> {text(comment["text"])}</p>')
                protocol = metadata.get("protocol")
                if protocol:
                    paragraphs.append(f'<p><strong>protocol</strong> {text(protocol.get("text", ""))}</p>')
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
        daily_entries.append({"segment_id": segment["segment_id"], "label": label,
                              "start_us": segment["start_us"], "end_us": segment["end_us"],
                              "understanding_status": (understanding or {}).get('status', 'completed' if understanding else 'pending'),
                              "understanding": understanding, "key_frames": segment["key_frames"], "scene_frames": segment["scene_frames"],
                              "source_ref": segment["source_ref"], "segment_json": segment["json_path"],
                              "transcription": recording.get("transcription"), "audio": recording.get("audio")})
    pending = sum(1 for record in index["recordings"]
                  if not any(s["recording_id"] == record["recording_id"] for s in index["segments"])
                  or any(s["recording_id"] == record["recording_id"] and (s["segment_id"] not in by_segment
                         or by_segment[s["segment_id"]].get('status') == 'partial')
                         for s in index["segments"]))
    overview = f'<p>{len(index["recordings"])} 个采集分片；{len(index["segments"])} 个保留区间；{pending} 个分片仍有阶段待完成。</p><p>有实验活动 {sum(s["activity"] == "active" for s in index["segments"])} 段，无活动／无关区间 {sum(s["activity"] == "inactive" for s in index["segments"])} 段。未通过活动筛选不证明人员没有其他行为。</p>'
    uncertainties = '<ul>' + ''.join(uncertainty_rows) + '</ul><p>单设备筛选不等于跨视角物理动作验收；稀疏抽帧不证明完整观察了相邻采样之间的过程。</p>'
    cost = f'<p>理解结果对应的已知用量：输入 {provider_usage["input_tokens"]}，输出 {provider_usage["output_tokens"]}，合计 {provider_usage["total_tokens"]} Token。{usage_unknown} 个窗口用量未完整提供。复用响应的用量属于原执行记录，不代表本轮新增消耗。</p><table><tr><th>时段</th><th>模型</th><th>结果来源</th><th>耗时与 Token</th></tr>' + ''.join(execution_rows) + '</table>'
    day_template, day_contract = template_contract("VC-DEVICE-DAY-REPORT-V1")
    daily_contents = {"daily_overview": overview, "experiment_briefs": ''.join(briefs) or '<p>等待片段产出。</p>',
                      "attention_and_handoff": uncertainties, "cost_and_timing": cost}
    atomic_bytes(layout.reports / "LaboratoryDailyReport.html", render_page(f"{layout.name} 实验室日报", day_template, day_contract, daily_contents).encode())
    atomic_json(layout.reports / "LaboratoryDailyReport.json", {
        "schema_version": VERSION, "archive": layout.name, "logical_report_count": 1, **day_contract,
        "evidence_status": "PARTIAL_EVIDENCE", "entries": daily_entries,
        "additional_mllm_calls": False, "additional_model_tokens": 0,
        "sections": day_template["sections"], "input_index_digest": digest(index)})
    semantic_template, semantic_contract = template_contract("VC-DEVICE-UNDERSTANDING-REPORT-V1")
    semantic_contents = {"coverage": overview, "timeline": ''.join(sections) or '<p>等待理解产出。</p>',
                         "context": '<ul>' + ''.join(context_rows) + '</ul>', "uncertainty": uncertainties, "execution": cost}
    atomic_bytes(layout.understanding / "UnderstandingReport.html", render_page(f"{layout.name} 多模态理解报告", semantic_template, semantic_contract, semantic_contents).encode())
    atomic_json(layout.understanding / "Understanding.json", {"schema_version": VERSION, "archive": layout.name,
                **semantic_contract, "understandings": index["understandings"], "sections": semantic_template["sections"],
                "additional_mllm_calls": False, "additional_model_tokens": 0})
